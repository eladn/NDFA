import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_scatter import scatter_sum, scatter_softmax


__all__ = ['ScatterGeneralAttention']


class ScatterGeneralAttention(nn.Module):
    def __init__(
            self,
            in_embed_dim: int,
            key_proj_dim: Optional[int] = None,
            project_keys: bool = True,
            project_values: bool = False,
            out_values_dim: Optional[int] = None):
        super(ScatterGeneralAttention, self).__init__()
        self.in_embed_dim = in_embed_dim
        self.key_proj_dim = self.in_embed_dim if key_proj_dim is None else key_proj_dim
        assert self.key_proj_dim == self.in_embed_dim or project_keys

        self.general_attention_weight = \
            nn.Parameter(torch.empty(size=(self.key_proj_dim,), dtype=torch.float))
        self.key_projection_layer = nn.Linear(
            in_features=self.in_embed_dim, out_features=self.key_proj_dim, bias=True) \
            if project_keys else None
        self.out_values_dim = self.in_embed_dim if out_values_dim is None else out_values_dim
        assert self.out_values_dim == self.in_embed_dim or project_values
        self.value_projection_layer = nn.Linear(
            in_features=self.in_embed_dim, out_features=self.out_values_dim, bias=True) \
            if project_values else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(super(ScatterGeneralAttention, self), 'reset_parameters'):
            super(ScatterGeneralAttention, self).register_parameters()
        if self.general_attention_weight is not None:
            bound = 1 / math.sqrt(self.key_proj_dim)
            nn.init.uniform_(self.general_attention_weight, -bound, bound)

    def forward(
            self, scattered_values: torch.Tensor, indices: torch.LongTensor, nr_elements: int) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert scattered_values.ndim == 2
        assert scattered_values.size(1) == self.in_embed_dim
        assert indices.size() == (scattered_values.size(0),)

        # scores(i) = softmax(<scattered_values[j] * general_attention_weight | j's s.t. indices[j]=i>)
        # out[i] = scores(i) ELEM_MUL <scattered_values[j] | j's s.t. indices[j]=i>

        scattered_keys = scattered_values if self.key_projection_layer is None else \
            self.key_projection_layer(scattered_values)
        assert scattered_keys.size() == (scattered_values.size(0), self.key_proj_dim)
        probs_scale_factor = self.key_proj_dim ** (-0.5)
        scattered_probs = torch.sum(scattered_keys * self.general_attention_weight, dim=1) * probs_scale_factor
        assert scattered_probs.ndim == 1 and scattered_probs.size() == (scattered_values.size(0),)
        # TODO: Is it ok calling `scatter_softmax()` when not all of the range
        #       [0, 1, 2, ..., attn_queries.size(0) - 1] occurs in `indices`?
        #       What is entered to the missing-indices-entries in the output vector?
        #       Are they zeroed (the good case) or just skipped (bad case because of
        #       incorrect offsetting of the following entries)?
        scattered_scores = scatter_softmax(src=scattered_probs, index=indices, dim=-1)
        assert scattered_scores.ndim == 1 and scattered_scores.size() == (scattered_values.size(0),)
        scattered_values_projected = scattered_values if self.value_projection_layer is None else \
            self.value_projection_layer(scattered_values)
        scattered_values_weighed_by_scores = \
            scattered_values_projected * scattered_scores.unsqueeze(-1).expand(scattered_values_projected.size())
        assert scattered_values_weighed_by_scores.size() == scattered_values_projected.size()
        # Note: Passing `dim_size` to `scatter_sum()` is important for the case where
        #       not all of the range [0, 1, 2, ..., attn_queries.size(0) - 1] occurs in `indices`.
        attn_applied = scatter_sum(
            scattered_values_weighed_by_scores, index=indices, dim=0, dim_size=nr_elements)
        return scattered_scores, attn_applied