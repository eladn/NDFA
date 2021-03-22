from dataclasses import dataclass
import omegaconf
from confclass import confparam
from enum import Enum
from typing import Optional


__all__ = ['DataFold', 'DatasetProperties']


class DataFold(Enum):
    Train = 'Train'
    Validation = 'Validation'
    Test = 'Test'


@dataclass
class DatasetProperties:
    name: str = omegaconf.MISSING
    datafold: Optional[str] = confparam(
        default=None,
        choices=('train', 'validation', 'test'))

    # TODO: add filters properties ...
