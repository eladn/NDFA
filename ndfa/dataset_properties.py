from confclass import confclass, confparam
from enum import Enum
from typing import Optional


class DataFold(Enum):
    Train = 'Train'
    Validation = 'Validation'
    Test = 'Test'


@confclass
class DatasetProperties:
    name: str
    datafold: Optional[str] = confparam(
        default=None,
        choices=('train', 'eval', 'test'))

    # TODO: add filters properties ...
