from typing import Optional
from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['NGramsParams']


@dataclass
class NGramsParams:
    min_n: Optional[int] = conf_field(
        default=None)
    max_n: Optional[int] = conf_field(
        default=3)
    create_sub_grams_from_long_gram: Optional[bool] = conf_field(
        default=False)
