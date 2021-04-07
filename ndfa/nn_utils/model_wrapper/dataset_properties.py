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
    datafold: Optional[str] = conf_field(
        default=None,
        choices=('train', 'validation', 'test'))

    # TODO: add filters properties ...
