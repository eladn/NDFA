from dataclasses import dataclass, field


__all__ = ['SequenceCombinerParams']


@dataclass
class SequenceCombinerParams:
    method: str = field(
        default='ends',)
        # choices=('attn', 'sum', 'mean', 'last', 'ends'),
        # description="...")
    nr_attn_heads: int = field(
        default=8)
    nr_dim_reduction_layers: int = field(
        default=1)
