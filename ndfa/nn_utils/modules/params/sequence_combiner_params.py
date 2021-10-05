from dataclasses import dataclass
from typing import Optional

from ndfa.misc.configurations_utils import conf_field, HasDispatchableField, DispatchField


__all__ = ['SequenceCombinerParams']


@dataclass
class SequenceCombinerParams(HasDispatchableField):
    method: str = conf_field(
        default='ends',
        choices=('attn', 'sum', 'mean', 'last', 'ends'))
    nr_attn_heads: Optional[int] = conf_field(
        default=8)
    project_attn_values: Optional[bool] = conf_field(
        default=False)
    nr_dim_reduction_layers: Optional[int] = conf_field(
        default=1)

    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'method', {
                'attn': ['nr_attn_heads', 'project_attn_values', 'nr_dim_reduction_layers'],
                'sum': [], 'mean': [], 'last': [], 'ends': []}))
