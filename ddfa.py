import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from ddfa.execution_parameters import ModelExecutionParams
from ddfa.ddfa_model_hyper_parameters import DDFAModelTrainingHyperParams
from ddfa.code_tasks.code_task_base import CodeTaskBase
from ddfa.dataset_properties import DataFold
from ddfa.nn_utils import fit


def create_optimizer(model: nn.Module, train_hps: DDFAModelTrainingHyperParams) -> Optimizer:
    # TODO: fully implement (choose optimizer and lr)!
    return torch.optim.Adam(model.parameters())


def main():
    exec_params: ModelExecutionParams = ModelExecutionParams.factory(
        load_from_args=True, load_from_yaml=True, verify_confclass=True)
    device = torch.device("cuda" if exec_params.use_gpu_if_available and torch.cuda.is_available() else "cpu")

    loaded_checkpoint = None
    if exec_params.should_load_model:
        loaded_checkpoint = torch.load(exec_params.model_load_path)
        # TODO: Modify `exec_params.experiment_setting` according to `loaded_checkpoint['experiment_setting']`.
        #       Verify overridden arguments and raise ArgumentException if needed.

    if exec_params.seed is not None:
        np.random.seed(exec_params.seed)
        torch.manual_seed(exec_params.seed)

    task = CodeTaskBase.load_task(exec_params.experiment_setting.task)

    if exec_params.perform_preprocessing:
        os.makedirs(exec_params.raw_train_data_path, exist_ok=True)
        if exec_params.raw_eval_data_path is not None:
            os.makedirs(exec_params.raw_eval_data_path, exist_ok=True)
        if exec_params.raw_test_data_path is not None:
            os.makedirs(exec_params.raw_test_data_path, exist_ok=True)
        task.preprocess(
            model_hps=exec_params.experiment_setting.model_hyper_params,
            pp_data_path=exec_params.pp_data_dir_path,
            raw_train_data_path=exec_params.raw_train_data_path,
            raw_eval_data_path=exec_params.raw_eval_data_path,
            raw_test_data_path=exec_params.raw_test_data_path)

    model = task.build_model(
        model_hps=exec_params.experiment_setting.model_hyper_params,
        pp_data_path=exec_params.pp_data_dir_path)

    if loaded_checkpoint:
        model.load_state_dict(loaded_checkpoint['model_state_dict'])

    if exec_params.perform_training:
        optimizer = create_optimizer(model, exec_params.experiment_setting.train_hyper_params)
        if loaded_checkpoint:
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

        train_dataset = task.create_dataset(
            model_hps=exec_params.experiment_setting.model_hyper_params,
            dataset_props=exec_params.experiment_setting.dataset,
            datafold=DataFold.Train,
            pp_data_path=exec_params.pp_data_dir_path)
        train_loader = DataLoader(
            train_dataset, batch_size=exec_params.experiment_setting.train_hyper_params.batch_size,
            collate_fn=task.collate_examples)  # FIXME: add shuffle=True
        eval_loader = None
        if exec_params.perform_evaluation:
            eval_dataset = task.create_dataset(
                model_hps=exec_params.experiment_setting.model_hyper_params,
                dataset_props=exec_params.experiment_setting.dataset,
                datafold=DataFold.Validation,
                pp_data_path=exec_params.eval_data_path)
            eval_loader = DataLoader(
                eval_dataset, batch_size=exec_params.experiment_setting.train_hyper_params.batch_size * 2)

        criterion = task.build_loss_criterion(model_hps=exec_params.experiment_setting.model_hyper_params)

        fit(
            nr_epochs=exec_params.experiment_setting.train_hyper_params.nr_epochs,
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=eval_loader,
            optimizer=optimizer,
            criterion=criterion)

    if exec_params.perform_evaluation:  # TODO: consider adding `and not exec_params.perform_training`
        raise NotImplementedError()  # TODO: implement!

    if exec_params.perform_prediction:
        raise NotImplementedError()  # TODO: implement!


if __name__ == '__main__':
    main()
