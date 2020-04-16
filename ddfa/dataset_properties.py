from confclass import confclass, confparam
from enum import Enum


class DataFold(Enum):
    Train = 'Train',
    Validation = 'Validation',
    Test = 'Test'


@confclass
class DatasetProperties:
    name: str
    datafold: str = confparam(
        choices=('train', 'eval', 'test'))

    # TODO: add filters properties ...
