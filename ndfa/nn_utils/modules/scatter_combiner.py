import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean
from typing import Optional

from ndfa.nn_utils.modules.scatter_attention import ScatterAttention
from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams


__all__ = ['ScatterCombiner']


class ScatterCombiner(nn.Module):
    def __init__(self, encoding_dim: int,
                 combiner_params: ScatterCombinerParams,
                 applied_attn_output_dim: Optional[int] = None):
        super(ScatterCombiner, self).__init__()
        self.encoding_dim = encoding_dim
        self.combiner_params = combiner_params
        if self.combiner_params.method not in {'mean', 'sum', 'attn'}:
            raise ValueError(f'Unsupported combining method `{self.combiner_params.method}`.')
        if self.combiner_params.method == 'attn':
            head_embed_dim = encoding_dim // self.combiner_params.nr_attn_heads
            assert head_embed_dim * self.combiner_params.nr_attn_heads == encoding_dim
            if applied_attn_output_dim is None:
                head_out_values_dim = head_embed_dim
            else:
                head_out_values_dim = applied_attn_output_dim // self.combiner_params.nr_attn_heads
                assert head_out_values_dim * self.combiner_params.nr_attn_heads == applied_attn_output_dim
            self.scatter_attn_layers = nn.ModuleList([
                ScatterAttention(
                    in_embed_dim=encoding_dim, qk_proj_dim=head_embed_dim,
                    project_values=self.combiner_params.project_attn_values,
                    out_values_dim=head_out_values_dim)
                for _ in range(self.combiner_params.nr_attn_heads)])
            actual_attn_output_dim = head_out_values_dim * self.combiner_params.nr_attn_heads
            self.applied_attn_linear_proj = None \
                if applied_attn_output_dim is None or applied_attn_output_dim == actual_attn_output_dim else \
                nn.Linear(
                    in_features=head_out_values_dim * self.combiner_params.nr_attn_heads,
                    out_features=applied_attn_output_dim)

    def forward(self, scattered_input: torch.Tensor, indices=torch.LongTensor,
                dim_size: Optional[int] = None, attn_queries: Optional[torch.Tensor] = None):
        if self.combiner_params.method == 'mean':
            combined = scatter_mean(
                src=scattered_input, index=indices,
                dim=0, dim_size=dim_size)
        elif self.combiner_params.method == 'sum':
            combined = scatter_sum(
                src=scattered_input, index=indices,
                dim=0, dim_size=dim_size)
        elif self.combiner_params.method == 'attn':
            assert attn_queries is not None
            assert dim_size is None or attn_queries.size(0) == dim_size
            applied_attn = [
                scatter_attn_layer(
                    scattered_values=scattered_input,
                    indices=indices, queries=attn_queries)[1]
                for scatter_attn_layer in self.scatter_attn_layers]
            combined = torch.cat(applied_attn, dim=-1) if len(applied_attn) > 1 else applied_attn[0]
            if self.applied_attn_linear_proj is not None:
                combined = self.applied_attn_linear_proj(combined)
        else:
            raise ValueError(f'Unsupported combining method `{self.combiner_params.method}`.')
        assert combined.ndim == 2
        assert dim_size is None or combined.size(0) == dim_size
        assert (self.combiner_params.method == 'attn' and self.applied_attn_linear_proj is not None) or \
               combined.size(1) == scattered_input.size(1)
        return combined
