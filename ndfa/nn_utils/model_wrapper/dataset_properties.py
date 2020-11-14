from confclass import confclass, confparam
from enum import Enum
from typing import Optional


__all__ = ['DataFold', 'DatasetProperties']


class DataFold(Enum):
    Train = 'Train'
    Validation = 'Validation'
    Test = 'Test'


@confclass
class DatasetProperties:
    name: str
    datafold: Optional[str] = confparam(
        default=None,
        choices=('train', 'validation', 'test'))

    # TODO: add filters properties ...
