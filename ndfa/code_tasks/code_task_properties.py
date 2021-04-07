from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['task_names', 'CodeTaskProperties']


task_names = ('pred-log-vars',)


@dataclass
class CodeTaskProperties:
    name: str = conf_field(
        default=task_names[0],
        choices=list(task_names))