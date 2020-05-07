import numpy as np
import os
import base64
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from typing import Optional
import itertools

from ddfa.execution_parameters import ModelExecutionParams
from ddfa.ddfa_model_hyper_parameters import DDFAModelTrainingHyperParams
from ddfa.code_tasks.code_task_base import CodeTaskBase
from ddfa.dataset_properties import DataFold
from ddfa.nn_utils.train_loop import fit


def create_optimizer(model: nn.Module, train_hps: DDFAModelTrainingHyperParams) -> Optimizer:
    # TODO: fully implement (choose optimizer and lr)!
    return torch.optim.Adam(model.parameters())


def main():
    exec_params: ModelExecutionParams = ModelExecutionParams.factory(
        load_from_args=True, load_from_yaml=True, verify_confclass=True)
    use_gpu = exec_params.use_gpu_if_available and torch.cuda.is_available()  # TODO: fix confclass issues (the bool default to true doesn't work)
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f'Using device: {device} (is CUDA available: {torch.cuda.is_available()})')

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    expr_settings_hash_base64 = base64.b64encode(str(hash(exec_params.experiment_setting)).encode('utf8'))\
        .strip().decode('ascii').strip('=')

    loaded_checkpoint = None
    if exec_params.should_load_model:
        ckpt_filepath = None
        if os.path.isfile(exec_params.model_load_path):
            ckpt_filepath = exec_params.model_load_path
        elif not os.path.isdir(exec_params.model_load_path):
            raise ValueError(f'No model to load ')
        else:
            for epoch_nr in itertools.count():
                # FIXME: it won't be found if the `epoch_nr` is encoded in the experiment setting...
                #  we should change the `epoch_nr` param in the `exec_params.experiment_setting` so it would be found.
                # TODO: consider getting all the models and find the model whose hps are 'adaptable' with the parsed
                #  `exec_params.experiment_setting` of the current run.
                tst_ckpt_filename = f'model_{expr_settings_hash_base64}_ep={epoch_nr}_.ckpt'
                if os.path.isfile(exec_params.model_load_path):
                    ckpt_filename = tst_ckpt_filename
                    ckpt_filepath = os.path.join(exec_params.model_load_path, ckpt_filename)
                else:
                    break
        if ckpt_filepath is None:
            raise ValueError(
                f'No model to load in dir {exec_params.model_load_path} that matches the chosen experiment setting.')
        with open(exec_params.model_load_path, 'bw') as checkpoint_file:
            loaded_checkpoint = torch.load(checkpoint_file)
        # TODO: Modify `exec_params.experiment_setting` according to `loaded_checkpoint['experiment_setting']`.
        #       Verify overridden arguments and raise ArgumentException if needed.
        expr_settings_hash_base64 = base64.b64encode(str(hash(exec_params.experiment_setting)).encode('utf8')) \
            .strip().decode('ascii').strip('=')

    # TODO: print the `experiment_setting`.

    if exec_params.seed is not None:
        np.random.seed(exec_params.seed)
        torch.manual_seed(exec_params.seed)

    task = CodeTaskBase.load_task(exec_params.experiment_setting.task)

    if exec_params.perform_preprocessing:
        os.makedirs(exec_params.pp_data_dir_path, exist_ok=True)
        task.preprocess(
            model_hps=exec_params.experiment_setting.model_hyper_params,
            pp_data_path=exec_params.pp_data_dir_path,
            raw_train_data_path=exec_params.raw_train_data_path,
            raw_eval_data_path=exec_params.raw_eval_data_path,
            raw_test_data_path=exec_params.raw_test_data_path)

    model = task.build_model(
        model_hps=exec_params.experiment_setting.model_hyper_params,
        pp_data_path=exec_params.pp_data_dir_path)

    print(f'Model built. #params: {sum(weight.nelement() for weight in model.parameters()):,}')

    if loaded_checkpoint:
        model.load_state_dict(loaded_checkpoint['model_state_dict'])

    if exec_params.perform_training:
        optimizer = create_optimizer(model, exec_params.experiment_setting.train_hyper_params)
        if loaded_checkpoint:
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

        saved_ckpts = []
        def save_checkpoint(model: nn.Module, optimizer: Optimizer, epoch_nr: int, step_nr: Optional[int] = None):
            assert exec_params.should_save_model
            os.makedirs(exec_params.model_save_path, exist_ok=True)
            ckpt_filepath = os.path.join(exec_params.model_save_path, f'model_{expr_settings_hash_base64}_ep={epoch_nr}_.ckpt')
            with open(ckpt_filepath, 'bw') as checkpoint_file:
                model.state_dict()
                torch.save({  # FIXME: we might want to modify these params
                    'experiment_setting': exec_params.experiment_setting,
                    'epoch_nr': epoch_nr,
                    'step_nr': step_nr,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, checkpoint_file)
            saved_ckpts.append((epoch_nr, ckpt_filepath))
            if exec_params.max_latest_checkpoints_to_keep is not None and \
                    len(saved_ckpts) > exec_params.max_latest_checkpoints_to_keep:
                for _ in range(len(saved_ckpts) - exec_params.max_latest_checkpoints_to_keep):
                    _, ckpt_filepath = saved_ckpts.pop(0)
                    os.remove(ckpt_filepath)

        train_dataset = task.create_dataset(
            model_hps=exec_params.experiment_setting.model_hyper_params,
            dataset_props=exec_params.experiment_setting.dataset,
            datafold=DataFold.Train,
            pp_data_path=exec_params.pp_data_dir_path)
        dataloader_cuda_kwargs = {'num_workers': 3, 'pin_memory': True} if use_gpu else {}  # TODO: play with `num_workers` and `pin_memory`; add these to `exec_params`
        train_loader = DataLoader(
            train_dataset, batch_size=exec_params.experiment_setting.train_hyper_params.batch_size,
            collate_fn=task.collate_examples, shuffle=True, **dataloader_cuda_kwargs)
        eval_loader = None
        if exec_params.perform_evaluation:
            eval_dataset = task.create_dataset(
                model_hps=exec_params.experiment_setting.model_hyper_params,
                dataset_props=exec_params.experiment_setting.dataset,
                datafold=DataFold.Validation,
                pp_data_path=exec_params.eval_data_path)
            eval_loader = DataLoader(
                eval_dataset, batch_size=exec_params.experiment_setting.train_hyper_params.batch_size * 2,
                **dataloader_cuda_kwargs)

        criterion = task.build_loss_criterion(model_hps=exec_params.experiment_setting.model_hyper_params)

        print('Starting training.')
        fit(
            nr_epochs=exec_params.experiment_setting.train_hyper_params.nr_epochs,
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=eval_loader,
            optimizer=optimizer,
            criterion=criterion,
            minibatch_size=8,  # TODO: make a train HP
            save_checkpoint_fn=save_checkpoint if exec_params.should_save_model else None)

    if exec_params.perform_evaluation:  # TODO: consider adding `and not exec_params.perform_training`
        raise NotImplementedError()  # TODO: implement!

    if exec_params.perform_prediction:
        raise NotImplementedError()  # TODO: implement!


if __name__ == '__main__':
    main()
