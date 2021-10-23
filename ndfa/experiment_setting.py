from dataclasses import dataclass

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams, NDFAModelTrainingHyperParams
from ndfa.code_tasks.code_task_properties import CodeTaskProperties
from ndfa.nn_utils.model_wrapper.dataset_properties import DatasetProperties
from ndfa.misc.configurations_utils import conf_field, DeterministicallyHashable


__all__ = ['ExperimentSetting']


@dataclass
class ExperimentSetting(DeterministicallyHashable):
    task: CodeTaskProperties = conf_field(
        default_factory=CodeTaskProperties,
        description="Parameters of the code-related task to tackle.",
        arg_prefix='task')

    model_hyper_params: NDFAModelHyperParams = conf_field(
        default_factory=NDFAModelHyperParams,
        description="NDFA model hyper-parameters.",
        arg_prefix='hp')

    train_hyper_params: NDFAModelTrainingHyperParams = conf_field(
        default_factory=NDFAModelTrainingHyperParams,
        description="NDFA model training hyper-parameters.",
        arg_prefix='trn')

    dataset: DatasetProperties = conf_field(
        default_factory=DatasetProperties,
        description="Dataset properties.",
        arg_prefix='ds')
