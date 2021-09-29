import dataclasses
from typing import Literal

from ndfa.misc.configurations_utils import conf_field


__all__ = ['NormWrapperParams']


@dataclasses.dataclass
class NormWrapperParams:
    affine: bool = conf_field(default=True)
    norm_type: Literal['later', 'batch'] = conf_field(default='layer')
