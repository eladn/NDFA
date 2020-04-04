from confclass import confclass, confparam

from .code_task_base import CodeTaskBase
from .predict_log_variables import PredictLogVariablesTask

__all__ = ['CodeTasks', 'CodeTaskProperties', 'load_task']

CodeTasks = [
    PredictLogVariablesTask
]


@confclass
class CodeTaskProperties:
    pass


def load_task(task_props: CodeTaskProperties) -> CodeTaskBase:
    raise NotImplementedError()  # TODO: implement!
