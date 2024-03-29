__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-04-04"

import abc
import torch
import functools
from torch.utils.data.dataset import Dataset
from typing import Optional, List, Any, Type, Iterable, Tuple
import torch.nn as nn

from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams
from ndfa.nn_utils.model_wrapper.dataset_properties import DatasetProperties, DataFold
from ndfa.code_tasks.preprocess_code_task_dataset import preprocess_code_task_dataset, PreprocessLimitExceedError
from ndfa.code_tasks.evaluation_metric_base import EvaluationMetric
from ndfa.code_tasks.code_task_properties import CodeTaskProperties, task_names
from ndfa.misc.iter_raw_extracted_data_files import RawExtractedExample
from ndfa.code_tasks.method_code_preprocess_params import NDFAModelPreprocessParams
from ndfa.nn_utils.model_wrapper.train_loop import TrainProgressInfo


__all__ = ['CodeTaskBase']


class CodeTaskBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, task_props: CodeTaskProperties):
        pass

    def preprocess_dataset(
            self, model_hps: NDFAModelHyperParams, dataset_props: DatasetProperties, pp_data_path: str,
            raw_train_data_path: str, raw_validation_data_path: Optional[str] = None,
            raw_test_data_path: Optional[str] = None, pp_nr_processes: int = 4, pp_override: bool = False,
            storage_method: str = 'dbm', compression_method: str = 'none',
            keep_entire_preprocessed_dataset: bool = False, use_compatible_pp_data_if_exists: bool = True):
        code_task_vocabs = self.create_or_load_code_task_vocabs(
            model_hps=model_hps, pp_data_path=pp_data_path, raw_train_data_path=raw_train_data_path)
        preprocess_code_task_dataset(
            model_hps=model_hps, dataset_props=dataset_props, pp_data_path=pp_data_path,
            raw_extracted_examples_generator=functools.partial(self.iterate_raw_examples, model_hps=model_hps),
            nr_raw_extracted_examples_getter=functools.partial(self.get_nr_raw_extracted_examples, model_hps=model_hps),
            pp_example_fn=self.preprocess_raw_example, code_task_vocabs=code_task_vocabs,
            raw_train_data_path=raw_train_data_path, raw_validation_data_path=raw_validation_data_path,
            raw_test_data_path=raw_test_data_path, nr_processes=pp_nr_processes, pp_override=pp_override,
            storage_method=storage_method, compression_method=compression_method,
            keep_entire_preprocessed_dataset=keep_entire_preprocessed_dataset,
            use_compatible_pp_data_if_exists=use_compatible_pp_data_if_exists)

    @abc.abstractmethod
    def iterate_raw_examples(
            self, model_hps: NDFAModelHyperParams, raw_extracted_data_dir: str,
            show_progress_bar: bool = False) -> Iterable[Any]:
        ...

    @abc.abstractmethod
    def get_nr_raw_extracted_examples(
            self, model_hps: NDFAModelHyperParams, raw_extracted_data_dir: str) -> int:
        ...

    @abc.abstractmethod
    def preprocess_raw_example(
            self, model_hps: NDFAModelHyperParams,
            preprocess_params: NDFAModelPreprocessParams,
            code_task_vocabs: CodeTaskVocabs,
            raw_example: Any, add_tag: bool = True) -> Any:
        ...

    def preprocess_raw_examples_generator(
            self, model_hps: NDFAModelHyperParams,
            raw_extracted_data_dir: str,
            code_task_vocabs: CodeTaskVocabs,
            add_tag: bool = True) \
            -> Iterable[Tuple[Any, Any]]:
        for raw_example in self.iterate_raw_examples(
                model_hps=model_hps, raw_extracted_data_dir=raw_extracted_data_dir, show_progress_bar=True):
            try:
                pp_example = self.preprocess_raw_example(
                    model_hps=model_hps, code_task_vocabs=code_task_vocabs,
                    raw_example=raw_example, add_tag=add_tag)
                assert pp_example is not None
                yield raw_example, pp_example
            except PreprocessLimitExceedError as err:
                yield raw_example, err

    @abc.abstractmethod
    def build_model(self, model_hps: NDFAModelHyperParams, pp_data_path: str) -> nn.Module:
        ...

    @abc.abstractmethod
    def predict(
            self, model: nn.Module, device: torch.device, raw_example: RawExtractedExample, pp_example: Any) -> Any:
        ...

    @abc.abstractmethod
    def create_dataset(
            self, model_hps: NDFAModelHyperParams, dataset_props: DatasetProperties,
            datafold: DataFold, pp_data_path: str, pp_storage_method: str = 'dbm',
            pp_compression_method: str = 'none', use_compatible_pp_data_if_exists: bool = True) -> Dataset:
        ...

    @abc.abstractmethod
    def build_loss_criterion(self, model_hps: NDFAModelHyperParams) -> nn.Module:
        ...

    @staticmethod
    def load_task(task_props: CodeTaskProperties) -> 'CodeTaskBase':
        assert task_props.name in task_names
        task_class = None
        if task_props.name == 'pred-log-vars':
            from ndfa.code_tasks.predict_log_variables_task import PredictLogVarsTask
            task_class = PredictLogVarsTask
        return task_class(task_props)

    @abc.abstractmethod
    def collate_examples(
            self, examples: List[Any], model_hps: NDFAModelHyperParams,
            is_training: bool, train_progress_info: Optional[TrainProgressInfo] = None):
        ...

    @abc.abstractmethod
    def evaluation_metrics(self, model_hps: NDFAModelHyperParams) -> List[Type[EvaluationMetric]]:
        ...

    @abc.abstractmethod
    def create_or_load_code_task_vocabs(
            self, model_hps: NDFAModelHyperParams,
            pp_data_path: str,
            raw_train_data_path: Optional[str] = None) -> CodeTaskVocabs:
        ...
