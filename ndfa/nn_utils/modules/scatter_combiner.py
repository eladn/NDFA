import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean
from typing import Optional

from ndfa.nn_utils.modules.scatter_attention import ScatterAttention


__all__ = ['ScatterCombiner']


class ScatterCombiner(nn.Module):
    def __init__(self, encoding_dim: int, combining_method: str = 'attn'):
        super(ScatterCombiner, self).__init__()
        self.encoding_dim = encoding_dim
        self.combining_method = combining_method
        if combining_method not in {'mean', 'sum', 'attn'}:
            raise ValueError(f'Unsupported combining method `{combining_method}`.')
        if self.combining_method == 'attn':
            self.scatter_attn_layer = ScatterAttention(values_dim=encoding_dim)

    def forward(self, scattered_input: torch.Tensor, indices=torch.LongTensor,
                dim_size: Optional[int] = None, attn_keys: Optional[torch.Tensor] = None):
        if self.combining_method == 'mean':
            combined = scatter_mean(
                src=scattered_input, index=indices,
                dim=0, dim_size=dim_size)
        elif self.combining_method == 'sum':
            combined = scatter_sum(
                src=scattered_input, index=indices,
                dim=0, dim_size=dim_size)
        elif self.combining_method == 'attn':
            assert attn_keys is not None
            assert dim_size is None or attn_keys.size(0) == dim_size
            _, combined = self.scatter_attn_layer(
                scattered_values=scattered_input, indices=indices,
                attn_keys=attn_keys)
        else:
            raise ValueError(f'Unsupported combining method `{self.combining_method}`.')
        assert combined.ndim == 2
        assert dim_size is None or combined.size(0) == dim_size
        assert combined.size(1) == scattered_input.size(1)
        return combined
