from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['ScatterCombinerParams']


@dataclass
class ScatterCombinerParams:
    method: str = conf_field(
        default='mean',
        choices=('attn', 'sum', 'mean'))
    nr_attn_heads: int = conf_field(
        default=8)
    project_attn_values: bool = conf_field(
        default=False)
    nr_dim_reduction_layers: int = conf_field(
        default=0)
