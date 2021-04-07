from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['SequenceCombinerParams']


@dataclass
class SequenceCombinerParams:
    method: str = conf_field(
        default='ends',
        choices=('attn', 'sum', 'mean', 'last', 'ends'))
    nr_attn_heads: int = conf_field(
        default=8)
    project_attn_values: bool = conf_field(
        default=False)
    nr_dim_reduction_layers: int = conf_field(
        default=1)
