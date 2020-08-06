from confclass import confclass, confparam


__all__ = ['task_names', 'CodeTaskProperties']


task_names = ('pred-log-vars',)


@confclass
class CodeTaskProperties:
    name: str = confparam(
        default=task_names[0],
        choices=list(task_names))