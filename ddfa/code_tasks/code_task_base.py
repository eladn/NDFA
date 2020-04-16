import abc
import torch
from torch.utils.data.dataset import Dataset
from typing import Optional
from confclass import confclass, confparam
import torch.nn as nn

from ddfa.ddfa_model_hyper_parameters import DDFAModelHyperParams
from ddfa.dataset_properties import DatasetProperties, DataFold


__all__ = ['task_names', 'CodeTaskProperties', 'CodeTaskBase']


task_names = ('pred-log-vars',)


@confclass
class CodeTaskProperties:
    name: str = confparam(
        default=task_names[0],
        choices=list(task_names))


class CodeTaskBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, task_props: CodeTaskProperties):
        pass

    @abc.abstractmethod
    def preprocess(self, model_hps: DDFAModelHyperParams, pp_data_path: str, raw_train_data_path: str,
                   raw_eval_data_path: Optional[str] = None, raw_test_data_path: Optional[str] = None):
        ...

    @abc.abstractmethod
    def build_model(self, model_hps: DDFAModelHyperParams, pp_data_path: str) -> nn.Module:
        ...

    @abc.abstractmethod
    def create_dataset(
            self, model_hps: DDFAModelHyperParams, dataset_props: DatasetProperties,
            datafold: DataFold, dataset_path: str) -> Dataset:
        ...

    @abc.abstractmethod
    def build_loss_criterion(self, model_hps: DDFAModelHyperParams) -> nn.Module:
        ...

    @staticmethod
    def load_task(task_props: CodeTaskProperties) -> 'CodeTaskBase':
        assert task_props.name in task_names
        task_class = None
        if task_props.name == 'pred-log-vars':
            from ddfa.code_tasks.predict_log_variables import PredictLogVariablesTask
            task_class = PredictLogVariablesTask
        return task_class(task_props)
