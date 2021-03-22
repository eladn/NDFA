from dataclasses import dataclass
from confclass import confparam


__all__ = ['SequenceCombinerParams']


@dataclass
class SequenceCombinerParams:
    method: str = confparam(
        default='ends',
        choices=('attn', 'sum', 'mean', 'last', 'ends'))
    nr_attn_heads: int = confparam(
        default=8)
    nr_dim_reduction_layers: int = confparam(
        default=1)
