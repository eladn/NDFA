import abc
from torch.utils.data.dataset import Dataset

from ddfa_model_hyper_parameters import DDFAModelHyperParams
from dataset_properties import DatasetProperties, DataFold


class CodeTaskBase(abc.ABC):
    @abc.abstractmethod
    def build_model(self, model_hps: DDFAModelHyperParams):
        ...

    @abc.abstractmethod
    def create_dataset(
            self, model_hps: DDFAModelHyperParams, dataset_props: DatasetProperties,
            datafold: DataFold, dataset_path: str) -> Dataset:
        ...
