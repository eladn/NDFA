from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

from ndfa.misc.configurations_utils import conf_field

__all__ = ['DistributionInfoParams', 'SamplingParams']


@dataclass
class DistributionInfoParams:
    class DistributionType(Enum):
        Normal = 'Normal'
        Gamma = 'Gamma'
        Uniform = 'Uniform'
    distribution_type: DistributionType = conf_field(
        default=DistributionType.Normal)
    distribution_params: Tuple[float, ...] = conf_field(
        default=(0, 1))


@dataclass
class SamplingParams:
    max_nr_items: Optional[int]
    distribution_for_rate_to_sample_by: Optional[DistributionInfoParams] = conf_field(
        default_factory=DistributionInfoParams)
    min_nr_items_to_sample_by_rate: Optional[int] = conf_field(
        default=None)
    sample_in_eval: bool = False
