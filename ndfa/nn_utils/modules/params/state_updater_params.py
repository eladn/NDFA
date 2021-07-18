import enum
from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['StateUpdaterParams']


@dataclass
class StateUpdaterParams:
    class Method(enum.Enum):
        CatProject = 'CatProject'
        Gate = 'Gate'
        Add = 'Add'
        PassThrough = 'PassThrough'

    method: Method = conf_field(
        default=Method.CatProject)
