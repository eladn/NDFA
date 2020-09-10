import torch
import torch.nn as nn

from ndfa.nn_utils.attention import Attention


__all__ = ['ExpressionCombiner']


class ExpressionCombiner(nn.Module):
    def __init__(self, expression_encoding_dim: int, combined_expression_dim: int,
                 nr_attn_heads: int = 1, nr_dim_reduction_layers: int = 1):
        super(ExpressionCombiner, self).__init__()
        self.expression_encoding_dim = expression_encoding_dim
        self.combined_expression_dim = combined_expression_dim
        self.nr_attn_heads = nr_attn_heads
        self.nr_dim_reduction_layers = nr_dim_reduction_layers
        self.attn_layers = nn.ModuleList([
            Attention(nr_features=self.expression_encoding_dim, project_key=True)
            for _ in range(nr_attn_heads)])
        # TODO: implement `projection_dimensions` list using `nr_dim_reduction_layers`!
        #       its length should be nr_dim_reduction_layers+1.
        assert expression_encoding_dim * nr_attn_heads > combined_expression_dim
        projection_dimensions = [expression_encoding_dim * nr_attn_heads, combined_expression_dim]
        self.dim_reduction_projection_layers = [
            nn.Linear(in_features=projection_dimensions[layer_idx],
                      out_features=projection_dimensions[layer_idx + 1])
            for layer_idx in range(nr_dim_reduction_layers)]

    def forward(self, expressions_encodings: torch.Tensor,
                expressions_mask: torch.BoolTensor,
                batch_first: bool = True):
        assert batch_first
        attn_heads = torch.cat([
            attn_layer(sequences=expressions_encodings,
                       attn_key_from=expressions_encodings[:, -1, :],
                       mask=expressions_mask)
            for attn_layer in self.attn_layers], dim=-1)
        projected = attn_heads
        for dim_reduction_projection_layer in self.dim_reduction_projection_layers:
            projected = dim_reduction_projection_layer(projected)
        assert projected.size(-1) == self.combined_expression_dim
        return projected
