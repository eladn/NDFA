import dataclasses
from enum import Enum

from ndfa.misc.configurations_utils import conf_field


__all__ = ['NormWrapperParams']


@dataclasses.dataclass
class NormWrapperParams:
    class NormType(Enum):
        Layer = 'Layer'
        Batch = 'Batch'
        PassThrough = 'PassThrough'

    affine: bool = conf_field(default=False)
    norm_type: NormType = conf_field(default=NormType.PassThrough)
