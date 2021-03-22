from dataclasses import dataclass
from confclass import confparam


__all__ = ['ScatterCombinerParams']


@dataclass
class ScatterCombinerParams:
    method: str = confparam(
        default='mean',
        choices=('attn', 'sum', 'mean'))
    nr_attn_heads: int = confparam(
        default=8)
    project_attn_values: bool = confparam(
        default=False)
    nr_dim_reduction_layers: int = confparam(
        default=0)
