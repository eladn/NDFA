import abc
import torch
import functools
from torch.utils.data.dataset import Dataset
from typing import Optional, List, Any, Type, Dict, Iterable, Tuple
from confclass import confclass, confparam
import torch.nn as nn

from ddfa.code_nn_modules.code_task_vocabs import CodeTaskVocabs
from ddfa.ddfa_model_hyper_parameters import DDFAModelHyperParams
from ddfa.dataset_properties import DatasetProperties, DataFold
from ddfa.code_tasks.preprocess_code_task_dataset import preprocess_code_task_dataset, PreprocessLimitExceedError


__all__ = ['task_names', 'CodeTaskProperties', 'CodeTaskBase', 'EvaluationMetric']


task_names = ('pred-log-vars',)


class EvaluationMetric(abc.ABC):
    @abc.abstractmethod
    def update(self, y_hat, target):
        ...

    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        ...


@confclass
class CodeTaskProperties:
    name: str = confparam(
        default=task_names[0],
        choices=list(task_names))


class CodeTaskBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, task_props: CodeTaskProperties):
        pass

    def preprocess_dataset(
            self, model_hps: DDFAModelHyperParams, pp_data_path: str, raw_train_data_path: str,
            raw_eval_data_path: Optional[str] = None, raw_test_data_path: Optional[str] = None):
        code_task_vocabs = self.create_or_load_code_task_vocabs(
            model_hps=model_hps, pp_data_path=pp_data_path, raw_train_data_path=raw_train_data_path)
        preprocess_code_task_dataset(
            model_hps=model_hps, pp_data_path=pp_data_path,
            raw_extracted_examples_generator=functools.partial(self.iterate_raw_examples, model_hps=model_hps),
            pp_example_fn=self.preprocess_raw_example, code_task_vocabs=code_task_vocabs,
            raw_train_data_path=raw_train_data_path, raw_eval_data_path=raw_eval_data_path,
            raw_test_data_path=raw_test_data_path)

    @abc.abstractmethod
    def iterate_raw_examples(self, model_hps: DDFAModelHyperParams, raw_extracted_data_dir: str) -> Iterable[Any]:
        ...

    @abc.abstractmethod
    def preprocess_raw_example(
            self, model_hps: DDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs,
            raw_example: Any, add_tag: bool = True) -> Any:
        ...

    def preprocess_raw_examples_generator(
            self, model_hps: DDFAModelHyperParams,
            raw_extracted_data_dir: str,
            code_task_vocabs: CodeTaskVocabs,
            add_tag: bool = True) \
            -> Iterable[Tuple[Any, Any]]:
        for raw_example in self.iterate_raw_examples(
                model_hps=model_hps, raw_extracted_data_dir=raw_extracted_data_dir):
            try:
                pp_example = self.preprocess_raw_example(
                    model_hps=model_hps, code_task_vocabs=code_task_vocabs,
                    raw_example=raw_example, add_tag=add_tag)
                assert pp_example is not None
                yield raw_example, pp_example
            except PreprocessLimitExceedError as err:
                yield raw_example, err

    @abc.abstractmethod
    def build_model(self, model_hps: DDFAModelHyperParams, pp_data_path: str) -> nn.Module:
        ...

    @abc.abstractmethod
    def predict(self, model: nn.Module, device: torch.device, pp_example: Any) -> Any:
        ...

    @abc.abstractmethod
    def create_dataset(
            self, model_hps: DDFAModelHyperParams, dataset_props: DatasetProperties,
            datafold: DataFold, pp_data_path: str) -> Dataset:
        ...

    @abc.abstractmethod
    def build_loss_criterion(self, model_hps: DDFAModelHyperParams) -> nn.Module:
        ...

    @staticmethod
    def load_task(task_props: CodeTaskProperties) -> 'CodeTaskBase':
        assert task_props.name in task_names
        task_class = None
        if task_props.name == 'pred-log-vars':
            from ddfa.code_tasks.predict_log_variables_task import PredictLogVarsTask
            task_class = PredictLogVarsTask
        return task_class(task_props)

    @abc.abstractmethod
    def collate_examples(self, examples: List[Any]):
        ...

    @abc.abstractmethod
    def evaluation_metrics(self, model_hps: DDFAModelHyperParams) -> List[Type[EvaluationMetric]]:
        ...

    @abc.abstractmethod
    def create_or_load_code_task_vocabs(
            self, model_hps: DDFAModelHyperParams,
            pp_data_path: str,
            raw_train_data_path: Optional[str] = None) -> CodeTaskVocabs:
        ...
