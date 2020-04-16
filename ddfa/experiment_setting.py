from confclass import confclass, confparam

from ddfa.ddfa_model_hyper_parameters import DDFAModelHyperParams, DDFAModelTrainingHyperParams
from ddfa.code_tasks.code_task_base import CodeTaskProperties
from ddfa.dataset_properties import DatasetProperties


__all__ = ['ExperimentSetting']


@confclass
class ExperimentSetting:
    task: CodeTaskProperties = confparam(
        default_factory=CodeTaskProperties,
        description="Parameters of the code-related task to tackle.",
        arg_prefix='task')

    model_hyper_params: DDFAModelHyperParams = confparam(
        default_factory=DDFAModelHyperParams,
        description="DDFA model hyper-parameters.",
        arg_prefix='hp')

    train_hyper_params: DDFAModelTrainingHyperParams = confparam(
        default_factory=DDFAModelTrainingHyperParams,
        description="DDFA model training hyper-parameters.",
        arg_prefix='trn')

    dataset: DatasetProperties = confparam(
        default_factory=DatasetProperties,
        description="Dataset properties.",
        arg_prefix='ds')
