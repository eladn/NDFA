from dataclasses import dataclass
import omegaconf
from enum import Enum
from typing import Optional

from ndfa.misc.configurations_utils import conf_field


__all__ = ['DataFold', 'DatasetProperties']


class DataFold(Enum):
    Train = 'Train'
    Validation = 'Validation'
    Test = 'Test'


@dataclass
class DatasetProperties:
    name: str = conf_field(
        default=omegaconf.MISSING)
    folding: str = conf_field(
        default='orig',
        choices=('orig', 'kfold_by_proj_1oo3', 'kfold_by_proj_2oo3', 'kfold_by_proj_3oo3',
                 'kfold_by_proj_1oo4', 'kfold_by_proj_2oo4', 'kfold_by_proj_3oo4', 'kfold_by_proj_4oo4'))

    # TODO: add filters properties ...
