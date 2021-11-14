import io
import os
import sys
import itertools
import functools
import multiprocessing
from warnings import warn
from typing import Optional, Tuple

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader, RandomSampler, BatchSampler

from ndfa.execution_parameters import ModelExecutionParams
from ndfa.experiment_setting import ExperimentSetting
from ndfa.ndfa_model_hyper_parameters import NDFAModelTrainingHyperParams
from ndfa.code_tasks.code_task_base import CodeTaskBase
from ndfa.nn_utils.model_wrapper.dataset_properties import DataFold
from ndfa.nn_utils.model_wrapper.train_loop import fit, evaluate
from ndfa.code_tasks.preprocess_code_task_dataset import PreprocessLimitExceedError
from ndfa.misc.configurations_utils import create_argparser_from_dataclass_conf_structure, \
    reinstantiate_omegaconf_container, create_conf_dotlist_from_parsed_args, HasDispatchableField
from ndfa.code_tasks.create_preprocess_params_from_model_hps import create_preprocess_params_from_model_hps
from ndfa.code_tasks.method_code_preprocess_params import NDFAModelPreprocessedDataParams


def create_optimizer(model: nn.Module, train_hps: NDFAModelTrainingHyperParams) -> Optimizer:
    # TODO: fully implement (choose optimizer and lr)!
    return torch.optim.AdamW(
        params=model.parameters(),
        lr=train_hps.learning_rate,
        weight_decay=train_hps.weight_decay)
    # return torch.optim.Adam(model.parameters(), lr=0.0005)


def create_lr_schedulers(model: nn.Module, train_hps: NDFAModelTrainingHyperParams, optimizer: Optimizer) \
        -> Tuple[torch.optim.lr_scheduler._LRScheduler, ...]:
    # FIXME: should we load `last_epoch` from `loaded_checkpoint` or is it loaded on `load_state_dict()`?
    schedulers = []
    if train_hps.learning_rate_decay:
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch_nr: (1 - train_hps.learning_rate_decay) ** epoch_nr,
            last_epoch=-1))
    if train_hps.reduce_lr_on_plateau:
        # TODO: load these params from `train_hps`
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=0.8, patience=4, verbose=True,
            threshold=0.1, threshold_mode='rel'))
    return tuple(schedulers)


def load_exec_params() -> ModelExecutionParams:
    conf = OmegaConf.structured(ModelExecutionParams)
    argparser = create_argparser_from_dataclass_conf_structure(ModelExecutionParams)
    args = argparser.parse_args()
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(create_conf_dotlist_from_parsed_args(args)))
    if os.path.isfile('ndfa_conf.yaml'):
        with open('ndfa_conf.yaml', 'r') as conf_file:
            conf = OmegaConf.merge(conf, OmegaConf.load(conf_file))
    exec_params = reinstantiate_omegaconf_container(conf, ModelExecutionParams)
    HasDispatchableField.fix_dispatch_fields(exec_params)
    return exec_params


def load_experiment_setting_from_yaml(yaml: str) -> ExperimentSetting:
    conf = OmegaConf.structured(ExperimentSetting)
    with io.StringIO(yaml) as yaml_file:
        yaml_file.seek(0)
        conf = OmegaConf.merge(conf, OmegaConf.load(yaml_file))
    exec_params = reinstantiate_omegaconf_container(conf, ExperimentSetting)
    HasDispatchableField.fix_dispatch_fields(exec_params)
    return exec_params


def create_task_dataset(task: CodeTaskBase, exec_params: ModelExecutionParams, datafold: DataFold) -> Dataset:
    return task.create_dataset(
        model_hps=exec_params.experiment_setting.model_hyper_params,
        dataset_props=exec_params.experiment_setting.dataset,
        datafold=datafold,
        pp_data_path=exec_params.pp_data_dir_path,
        pp_storage_method=exec_params.pp_storage_method,
        pp_compression_method=exec_params.pp_compression_method)


def main():
    exec_params = load_exec_params()

    experiment_setting_yaml = OmegaConf.to_yaml(OmegaConf.structured(exec_params.experiment_setting))
    expr_settings_hash_base64 = exec_params.experiment_setting.get_sha1_base64()
    model_hps_yaml = OmegaConf.to_yaml(OmegaConf.structured(exec_params.experiment_setting.model_hyper_params))
    model_hps_hash_base64 = exec_params.experiment_setting.model_hyper_params.get_sha1_base64()

    loaded_checkpoint = None
    if exec_params.should_load_model:
        ckpt_filepath = None
        if os.path.isfile(exec_params.model_load_path):
            ckpt_filepath = exec_params.model_load_path
        elif not os.path.isdir(exec_params.model_load_path):
            raise ValueError(f'The model to load path provided does not exist (`{exec_params.model_load_path}`).')
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
        with open(exec_params.model_load_path, 'br') as checkpoint_file:
            loaded_checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        exec_params.experiment_setting = load_experiment_setting_from_yaml(loaded_checkpoint['experiment_setting_yaml'])
        experiment_setting_yaml = OmegaConf.to_yaml(OmegaConf.structured(exec_params.experiment_setting))
        expr_settings_hash_base64 = exec_params.experiment_setting.get_sha1_base64()
        model_hps_yaml = OmegaConf.to_yaml(OmegaConf.structured(exec_params.experiment_setting.model_hyper_params))
        model_hps_hash_base64 = exec_params.experiment_setting.model_hyper_params.get_sha1_base64()
        warn(f'Using experiment settings from loaded checkpoint [hash=`{expr_settings_hash_base64}`]. '
             f'Ignoring experiment settings from other inputs.')

    preprocess_params = create_preprocess_params_from_model_hps(
        model_hps=exec_params.experiment_setting.model_hyper_params)
    preprocessed_data_params = NDFAModelPreprocessedDataParams(
        preprocess_params=preprocess_params, dataset_props=exec_params.experiment_setting.dataset)
    if exec_params.get_pp_data_params_hash:
        print(preprocessed_data_params.get_sha1_base64())
        exit(0)

    use_gpu = exec_params.use_gpu_if_available and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f'Using device: {device} (is CUDA available: {torch.cuda.is_available()})')
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'Experiment setting [hash=`{expr_settings_hash_base64}`]:')
    print(experiment_setting_yaml)

    if exec_params.seed is not None:
        np.random.seed(exec_params.seed)
        torch.manual_seed(exec_params.seed)

    task = CodeTaskBase.load_task(exec_params.experiment_setting.task)

    """
    This is a POC for finding the relevant tensor fields that the model actually needs.
    The idea is to remove the rest while performing the preprocessing, so that the pp_data would be smaller.
    However, if a some (used) TensorDataClass has self_index_group, we should find the matching field that
      has the same tgt_index_group and add it's relevant sub-fields so that the collate() could use it.
    Also, in some places in the model, we might access a tensor (for example to pass it as a parameter to 
      some other model), and it marks this field as used. We should replace these with "proxies" that touches
      this field only if really needed.
    Additionally, we should handle the ngrams dynamic dictionaries somehow (each example may have different ngram-ns).
    """
    # if exec_params.perform_preprocessing:
    #     os.makedirs(exec_params.pp_data_dir_path, exist_ok=True)
    #
    #     model = task.build_model(
    #         model_hps=exec_params.experiment_setting.model_hyper_params,
    #         pp_data_path=exec_params.pp_data_dir_path)
    #
    #     code_task_vocabs = task.create_or_load_code_task_vocabs(
    #         model_hps=exec_params.experiment_setting.model_hyper_params,
    #         pp_data_path=exec_params.pp_data_dir_path,
    #         raw_train_data_path=exec_params.raw_train_data_path)
    #
    #     for raw_example in task.iterate_raw_examples(
    #             model_hps=exec_params.experiment_setting.model_hyper_params,
    #             raw_extracted_data_dir=exec_params.raw_train_data_path):
    #         try:
    #             from ndfa.code_tasks.method_code_preprocess_params import NDFAModelPreprocessParams
    #             pp_example = task.preprocess_raw_example(
    #                 model_hps=exec_params.experiment_setting.model_hyper_params,
    #                 preprocess_params=NDFAModelPreprocessParams.all(),
    #                 code_task_vocabs=code_task_vocabs,
    #                 raw_example=raw_example, add_tag=True)
    #             if isinstance(pp_example, PreprocessLimitExceedError):
    #                 continue
    #         except PreprocessLimitExceedError:
    #             continue
    #
    #         code_task_input = pp_example.code_task_input
    #         model.to(device)
    #         model.eval()
    #         example_hashes = [pp_example.example_hash]
    #         from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
    #         from ndfa.misc.tensors_data_class import CollateData
    #         code_task_input = MethodCodeInputTensors.collate(
    #             [code_task_input], collate_data=CollateData(example_hashes=example_hashes, model_hps=model.model_hps))
    #         lazy_usage_history = {}
    #         identity_map_fn = lambda field_val: field_val
    #         code_task_input = code_task_input.deep_lazy_map(
    #             map_fn=identity_map_fn,
    #             mapper_override_group='identity_to_check_hist',
    #             lazy_map_usage_history=lazy_usage_history)
    #         # print(lazy_usage_history)
    #         assert sum(1 for key, tensors in lazy_usage_history.items() if tensors) == 0
    #         output = model(code_task_input=code_task_input)
    #         # TODO: add the loss criterion calculation with the ground-truth tensor.
    #         print(['.'.join(tuple(str(a) for a in key) + (str(tensor),)) for key, tensors in lazy_usage_history.items() if tensors for tensor in tensors])
    #         break
    #
    #     # TODO: make a nested structured dict with the required fields.
    #     # TODO: find the additional relevant indexing fixing fields (with `tgt_index_group`) and add them to the required fields dict.
    #     # TODO: perform the preprocess for all the dataset
    #     raise NotImplementedError

    if exec_params.perform_preprocessing:
        os.makedirs(exec_params.pp_data_dir_path, exist_ok=True)
        task.preprocess_dataset(
            model_hps=exec_params.experiment_setting.model_hyper_params,
            dataset_props=exec_params.experiment_setting.dataset,
            pp_data_path=exec_params.pp_data_dir_path,
            raw_train_data_path=exec_params.raw_train_data_path,
            raw_validation_data_path=exec_params.raw_validation_data_path,
            raw_test_data_path=exec_params.raw_test_data_path,
            pp_nr_processes=exec_params.pp_nr_processes,
            pp_override=exec_params.pp_override,
            storage_method=exec_params.pp_storage_method,
            compression_method=exec_params.pp_compression_method)

    model = task.build_model(
        model_hps=exec_params.experiment_setting.model_hyper_params,
        pp_data_path=exec_params.pp_data_dir_path)

    print(f'Model built. #params: {sum(weight.nelement() for weight in model.parameters()):,}')
    print(model)

    if loaded_checkpoint:
        model.load_state_dict(loaded_checkpoint['model_state_dict'])

    dataloader_num_workers = \
        multiprocessing.cpu_count() \
            if exec_params.dataloader_num_workers is None else \
            exec_params.dataloader_num_workers
    print(f'Using {dataloader_num_workers} dataloader workers.')
    dataloader_cuda_kwargs = {
        'num_workers': dataloader_num_workers,
        'pin_memory': exec_params.dataloader_pin_memory,
        'persistent_workers': False} if use_gpu else {}
    torch_version = tuple(int(v) for v in torch.__version__.split('.')[:2])
    if use_gpu and dataloader_num_workers > 0 and torch_version >= (1, 8):
        dataloader_prefetch_factor = 20  # TODO: pass `prefetch_factor` from a param
        dataloader_cuda_kwargs['prefetch_factor'] = dataloader_prefetch_factor

    if exec_params.perform_training:
        optimizer = create_optimizer(model, exec_params.experiment_setting.train_hyper_params)
        if loaded_checkpoint:
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        # FIXME: should we load `last_epoch` from `loaded_checkpoint` or is it loaded on `load_state_dict()`?
        lr_schedulers = create_lr_schedulers(model, exec_params.experiment_setting.train_hyper_params, optimizer)
        if loaded_checkpoint:
            for lr_scheduler_idx, lr_scheduler in enumerate(lr_schedulers):
                lr_scheduler.load_state_dict(loaded_checkpoint[f'lr_scheduler_{lr_scheduler_idx}_state_dict'])

        saved_ckpts = []
        def save_checkpoint(model: nn.Module, optimizer: Optimizer, epoch_nr: int, step_nr: Optional[int] = None):
            assert exec_params.should_save_model
            os.makedirs(exec_params.model_save_path, exist_ok=True)
            ckpt_filepath = os.path.join(exec_params.model_save_path, f'model_{expr_settings_hash_base64}_ep={epoch_nr}_.ckpt')
            with open(ckpt_filepath, 'bw') as checkpoint_file:
                model.state_dict()
                # FIXME: we might want to modify these params
                new_ckpt_state_dict = {
                    'experiment_setting_yaml': experiment_setting_yaml,
                    'epoch_nr': epoch_nr,
                    'step_nr': step_nr,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}
                for lr_scheduler_idx, lr_scheduler in enumerate(lr_schedulers):
                    new_ckpt_state_dict[f'lr_scheduler_{lr_scheduler_idx}_state_dict'] = lr_scheduler.state_dict()
                torch.save(new_ckpt_state_dict, checkpoint_file)
            saved_ckpts.append((epoch_nr, ckpt_filepath))
            if exec_params.max_latest_checkpoints_to_keep is not None and \
                    len(saved_ckpts) > exec_params.max_latest_checkpoints_to_keep:
                for _ in range(len(saved_ckpts) - exec_params.max_latest_checkpoints_to_keep):
                    _, ckpt_filepath = saved_ckpts.pop(0)
                    try:
                        os.remove(ckpt_filepath)
                    except OSError as err:  # Note: we could also use `FileNotFoundError` here.
                        warn(f'Error while trying to remove the checkpoint at `{ckpt_filepath}` '
                             f'(because `max_latest_checkpoints_to_keep` '
                             f'[{exec_params.max_latest_checkpoints_to_keep}] is reached): {err}')

        train_dataset = create_task_dataset(task=task, exec_params=exec_params, datafold=DataFold.Train)
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=exec_params.batch_size,
        #     collate_fn=functools.partial(
        #         task.collate_examples,
        #         model_hps=exec_params.experiment_setting.model_hyper_params),
        #     shuffle=True, **dataloader_cuda_kwargs)
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            sampler=BatchSampler(
                RandomSampler(range(len(train_dataset))),
                batch_size=exec_params.batch_size, drop_last=False),
            collate_fn=functools.partial(
                task.collate_examples,
                model_hps=exec_params.experiment_setting.model_hyper_params),
            shuffle=False,
            **dataloader_cuda_kwargs)
        eval_loader = None
        if exec_params.perform_evaluation:
            eval_dataset = create_task_dataset(task=task, exec_params=exec_params, datafold=DataFold.Validation)
            # eval_loader = DataLoader(
            #     eval_dataset, batch_size=exec_params.batch_size,
            #     collate_fn=functools.partial(
            #         task.collate_examples,
            #         model_hps=exec_params.experiment_setting.model_hyper_params),
            #     shuffle=True, **dataloader_cuda_kwargs)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=None,
                sampler=BatchSampler(
                    RandomSampler(range(len(eval_dataset))),
                    batch_size=exec_params.batch_size, drop_last=False),
                collate_fn=functools.partial(
                    task.collate_examples,
                    model_hps=exec_params.experiment_setting.model_hyper_params),
                shuffle=False,
                **dataloader_cuda_kwargs)

        criterion = task.build_loss_criterion(model_hps=exec_params.experiment_setting.model_hyper_params)

        train_callbacks = []
        if exec_params.use_notify:
            from ndfa.nn_utils.model_wrapper.notify_train_callback import NotifyCallback
            train_callbacks.append(NotifyCallback())

        gdrive_logger = None
        if exec_params.use_gdrive_logger:
            from ndfa.nn_utils.model_wrapper.gdrive_train_logger_callback import GDriveTrainLoggerCallback
            from ndfa.nn_utils.model_wrapper.gdrive_train_logger import GDriveTrainLogger
            gdrive_logger = GDriveTrainLogger(
                gdrive_base_folder_id=exec_params.train_results_gdrive_folder_id,
                model_hps_hash=model_hps_hash_base64,
                experiment_settings_hash=expr_settings_hash_base64)
            gdrive_logger.upload_string_as_text_file(
                experiment_setting_yaml, filename='experiment_settings.yaml')
            gdrive_logger.upload_string_as_text_file(
                expr_settings_hash_base64, filename='experiment_settings_hash.txt')
            gdrive_logger.upload_string_as_text_file(
                model_hps_yaml, filename='model_hps.yaml')
            gdrive_logger.upload_string_as_text_file(
                model_hps_hash_base64, filename='model_hps_hash.txt')
            gdrive_logger.upload_string_as_text_file(
                OmegaConf.to_yaml(OmegaConf.structured(preprocessed_data_params)),
                filename='preprocessed_data_params.yaml')
            gdrive_logger.upload_string_as_text_file(
                preprocessed_data_params.get_sha1_base64(), filename='preprocessed_data_params_hash.txt')
            gdrive_logger.upload_string_as_text_file(
                ' '.join(sys.argv), filename='exec_command.txt')
            gdrive_logger.upload_string_as_text_file(
                f'{exec_params.experiment_setting.dataset.name}\n{exec_params.experiment_setting.dataset.folding}',
                filename='dataset_name.txt')
            gdrive_logger.upload_string_as_text_file(
                f'{repr(model)}\n#params: {sum(weight.nelement() for weight in model.parameters()):,}',
                filename='model_architecture_description.txt')
            gdrive_logger.run_subprocess_and_upload_stdout_as_text_file(
                subprocess_args=['git', 'log', '--name-status', 'HEAD^..HEAD'], filename='git_commit.txt')
            gdrive_logger.run_subprocess_and_upload_stdout_as_text_file(
                subprocess_args=['printenv'], filename='environment.txt', ignore_fault=True)
            gdrive_logger.run_subprocess_and_upload_stdout_as_text_file(
                subprocess_args=['nvidia-smi'], filename='nvidia-smi.txt', ignore_fault=True)
            train_callbacks.append(GDriveTrainLoggerCallback(gdrive_logger))

        print('Starting training.')
        try:
            fit(
                nr_epochs=exec_params.experiment_setting.train_hyper_params.nr_epochs,
                model=model,
                device=device,
                train_loader=train_loader,
                valid_loader=eval_loader,
                optimizer=optimizer,
                lr_schedulers=lr_schedulers,
                criterion=criterion,
                nr_gradient_accumulation_steps=
                exec_params.experiment_setting.train_hyper_params.eff_batch_size // exec_params.batch_size,
                save_checkpoint_fn=save_checkpoint if exec_params.should_save_model else None,
                evaluation_metrics_types=task.evaluation_metrics(
                    model_hps=exec_params.experiment_setting.model_hyper_params),
                callbacks=train_callbacks,
                gradient_clip_param=exec_params.experiment_setting.train_hyper_params.gradient_clip,
                progress_bar_min_interval_sec=exec_params.progress_bar_min_interval_sec)
        finally:
            if gdrive_logger is not None:
                gdrive_logger.close()

    if exec_params.perform_evaluation:  # TODO: consider adding `and not exec_params.perform_training`
        print('Performing evaluation (over the validation set) ..')
        eval_dataset = create_task_dataset(task=task, exec_params=exec_params, datafold=DataFold.Validation)
        eval_loader = DataLoader(
            eval_dataset, batch_size=exec_params.batch_size,
            collate_fn=functools.partial(
                task.collate_examples,
                model_hps=exec_params.experiment_setting.model_hyper_params),
            shuffle=True, **dataloader_cuda_kwargs)
        criterion = task.build_loss_criterion(model_hps=exec_params.experiment_setting.model_hyper_params)
        val_loss, metrics_results = evaluate(
            model=model,
            device=device,
            valid_loader=eval_loader,
            criterion=criterion,
            evaluation_metrics_types=task.evaluation_metrics(
                model_hps=exec_params.experiment_setting.model_hyper_params),
            progress_bar_min_interval_sec=exec_params.progress_bar_min_interval_sec)
        # TODO: For pretty printing the evaluation metric results:
        #       https://stackoverflow.com/questions/44356693/pprint-with-custom-float-formats
        print(f'Completed performing evaluation.'
              f'\n\t validation loss: {val_loss:.4f}'
              f'\n\t validation metrics: {metrics_results}')

    if exec_params.perform_prediction:
        if exec_params.predict_raw_data_path:
            print(f'Performing prediction (over raw data in `{exec_params.predict_raw_data_path}`) ..')
            # TODO: consider getting the vocabs from `model`
            code_task_vocabs = task.create_or_load_code_task_vocabs(
                model_hps=exec_params.experiment_setting.model_hyper_params,
                pp_data_path=exec_params.pp_data_dir_path,
                raw_train_data_path=exec_params.raw_train_data_path)
            os.makedirs(exec_params.predict_output_path, exist_ok=True)
            with open(os.path.join(exec_params.predict_output_path, 'predictions.txt'), 'w') as predictions_output_file, \
                    open(os.path.join(exec_params.predict_output_path, 'predictions_hashes.txt'), 'w') as predictions_hashes_output_file:
                for raw_example, pp_example in task.preprocess_raw_examples_generator(
                        model_hps=exec_params.experiment_setting.model_hyper_params,
                        raw_extracted_data_dir=exec_params.predict_raw_data_path,
                        code_task_vocabs=code_task_vocabs, add_tag=False):
                    if isinstance(pp_example, PreprocessLimitExceedError):
                        continue
                    prediction = task.predict(
                        model=model, device=device, raw_example=raw_example, pp_example=pp_example)
                    predictions_output_file.write(' '.join(word for word in prediction))
                    predictions_output_file.write('\n')
                    predictions_hashes_output_file.write(f'{pp_example.example_hash}\n')
            print(f'Completed performing prediction.')
        elif exec_params.predict_pp_data_path:
            print(f'Performing prediction (over preprocessed data in `{exec_params.predict_pp_data_path}`) ..')
            raise NotImplementedError
            dataset = create_task_dataset(task=task, exec_params=exec_params, datafold=DataFold.Test)
            # data_loader = DataLoader(
            #     dataset, batch_size=exec_params.batch_size,
            #     collate_fn=functools.partial(
            #         task.collate_examples,
            #         model_hps=exec_params.experiment_setting.model_hyper_params),
            #     **dataloader_cuda_kwargs)
            # predictions = task.predict(
            #     model=model,
            #     device=device,
            #     data_loader=data_loader)
            print(f'Completed performing prediction.')
        else:
            assert False

    if exec_params.dbg_validate_batch_separation:
        """
        Check whether the model produces the same outputs of a given batch when running twice on the 
          exact same batching and third time on the same batch but when ordered in reversed.
        """
        train_dataset = create_task_dataset(task=task, exec_params=exec_params, datafold=DataFold.Train)
        from torch.utils.data.dataloader import Sampler

        class MockSampler(Sampler):
            def __init__(self, data_source):
                super(MockSampler, self).__init__(data_source=data_source)
                self.data_source = data_source
                self.nr_calls = 0

            def __iter__(self):
                if self.nr_calls % 3 < 2:
                    yield from self.data_source
                else:
                    yield from list(reversed(list(self.data_source)))
                self.nr_calls += 1

        from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors

        batch_size = exec_params.batch_size
        # batch_size = 8
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            sampler=BatchSampler(
                MockSampler(range(batch_size)),
                batch_size=batch_size, drop_last=False),
            collate_fn=functools.partial(
                task.collate_examples,
                model_hps=exec_params.experiment_setting.model_hyper_params),
            shuffle=False,
            **dataloader_cuda_kwargs)
        results = []
        model.to(device)
        model.eval()  # to cancel dropouts (as we expect same results from different forward() calls)
        for epoch_nr in range(3):
            print(f'forward() call #{epoch_nr}')
            for x_batch, y_batch in iter(train_loader):
                assert isinstance(x_batch, MethodCodeInputTensors)
                print('#AST nodes per example',
                      x_batch.ast.ast_node_types.nr_items_per_example)
                print('#CFG/AST l2l paths per example',
                      x_batch.pdg.cfg_nodes_expressions_ast.ast_leaf_to_leaf_paths_node_indices.nr_items_per_example)
                print('#expressions per example',
                      x_batch.pdg.cfg_nodes_expressions_ast.pdg_node_idx_to_sub_ast_root_idx_mapping_key.nr_items_per_example)
                print('#CFG nodes per example',
                      x_batch.pdg.cfg_nodes_control_kind.nr_items_per_example)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)  # PredictLoggingCallVarsModelOutput
                y_pred = y_pred.decoder_outputs
                results.append(y_pred.cpu().detach().numpy())
        first_inf_mask = np.isinf(results[0])
        assert np.all(first_inf_mask == np.isinf(results[1]))
        assert np.allclose(results[0][~first_inf_mask] - results[1][~first_inf_mask], .0)

        reversed_res_2 = results[2][::-1]
        assert np.all(first_inf_mask == np.isinf(reversed_res_2))
        nonclose_places = ~np.isclose(results[0][~first_inf_mask], reversed_res_2[~first_inf_mask], atol=1.e-7)
        abs_err = np.abs(results[0][~first_inf_mask] - reversed_res_2[~first_inf_mask])
        print(f'abs_err: mean(nonclose): {np.mean(abs_err[nonclose_places]) if len(nonclose_places) else np.nan}, '
              f'mean(all): {np.mean(abs_err)}, max: {np.max(abs_err)}')
        print(f'#nonclose {np.sum(nonclose_places)} / {results[0][~first_inf_mask].size}')
        assert not np.any(nonclose_places)


if __name__ == '__main__':
    main()
