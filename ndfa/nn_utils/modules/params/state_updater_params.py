from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['StateUpdaterParams']


@dataclass
class StateUpdaterParams:
    method: str = conf_field(
        default='cat-project',
        choices=('cat-project', 'gate', 'add', 'pass-through'))
