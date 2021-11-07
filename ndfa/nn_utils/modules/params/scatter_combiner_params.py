from dataclasses import dataclass
from typing import Optional

from ndfa.misc.configurations_utils import conf_field, HasDispatchableField, DispatchField


__all__ = ['ScatterCombinerParams']


@dataclass
class ScatterCombinerParams(HasDispatchableField):
    method: str = conf_field(
        default='mean',
        choices=('attn', 'sum', 'mean', 'general_attn'))
    nr_attn_heads: Optional[int] = conf_field(
        default=8)
    project_attn_values: Optional[bool] = conf_field(
        default=False)

    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'method', {
                'attn': ['nr_attn_heads', 'project_attn_values'],
                'general_attn': ['project_attn_values'],
                'sum': [], 'mean': []}))
