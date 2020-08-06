from confclass import confclass, confparam

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams, NDFAModelTrainingHyperParams
from ndfa.code_tasks.code_task_properties import CodeTaskProperties
from ndfa.dataset_properties import DatasetProperties


__all__ = ['ExperimentSetting']


@confclass
class ExperimentSetting:
    task: CodeTaskProperties = confparam(
        default_factory=CodeTaskProperties,
        description="Parameters of the code-related task to tackle.",
        arg_prefix='task')

    model_hyper_params: NDFAModelHyperParams = confparam(
        default_factory=NDFAModelHyperParams,
        description="NDFA model hyper-parameters.",
        arg_prefix='hp')

    train_hyper_params: NDFAModelTrainingHyperParams = confparam(
        default_factory=NDFAModelTrainingHyperParams,
        description="NDFA model training hyper-parameters.",
        arg_prefix='trn')

    dataset: DatasetProperties = confparam(
        default_factory=DatasetProperties,
        description="Dataset properties.",
        arg_prefix='ds')
