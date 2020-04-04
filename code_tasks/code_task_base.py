import abc

from ddfa_model_hyper_parameters import DDFAModelHyperParams


class CodeTaskBase(abc.ABC):
    @abc.abstractmethod
    def build_model(self, model_hps: DDFAModelHyperParams):
        ...

    @abc.abstractmethod
    def create_dataset(self, model_hps: DDFAModelHyperParams):
        ...
