__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-09-30"

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_scatter import scatter_sum, scatter_softmax


__all__ = ['ScatterAttention']


class ScatterAttention(nn.Module):
    def __init__(self, in_embed_dim: int, in_queries_dim: Optional[int] = None, qk_proj_dim: Optional[int] = None,
                 project_queries: bool = True, project_keys: bool = True, project_values: bool = False,
                 out_values_dim: Optional[int] = None, attn_scores_dropout_rate: float = 0.0):
        super(ScatterAttention, self).__init__()
        self.in_embed_dim = in_embed_dim
        self.in_queries_dim = self.in_embed_dim if in_queries_dim is None else in_queries_dim
        assert self.in_queries_dim == self.in_embed_dim or project_queries
        self.qk_proj_dim = self.in_embed_dim if qk_proj_dim is None else qk_proj_dim
        assert self.qk_proj_dim == self.in_embed_dim or project_keys
        assert self.qk_proj_dim == self.in_queries_dim or project_queries
        self.query_projection_layer = nn.Linear(
            in_features=self.in_queries_dim, out_features=self.qk_proj_dim, bias=True) \
            if project_queries else None
        self.key_projection_layer = nn.Linear(
            in_features=self.in_embed_dim, out_features=self.qk_proj_dim, bias=True) \
            if project_keys else None
        self.out_values_dim = self.in_embed_dim if out_values_dim is None else out_values_dim
        assert self.out_values_dim == self.in_embed_dim or project_values
        self.value_projection_layer = nn.Linear(
            in_features=self.in_embed_dim, out_features=self.out_values_dim, bias=False) \
            if project_values else None
        self.attn_scores_dropout = nn.Dropout(attn_scores_dropout_rate)

    def forward(self, scattered_values: torch.Tensor, indices: torch.LongTensor,
                queries: torch.Tensor, queries_indices: Optional[torch.LongTensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert scattered_values.ndim == 2
        assert scattered_values.size(1) == self.in_embed_dim
        assert indices.size() == (scattered_values.size(0),)
        assert queries.ndim == 2
        assert queries.size(1) == self.in_queries_dim
        assert queries_indices is None or queries_indices.size() == (queries.size(0),)

        attn_queries_projected = self.query_projection_layer(queries)
        assert attn_queries_projected.size() == (queries.size(0), self.qk_proj_dim)

        if queries_indices is None:
            # means that queries[i] consists the query vector of the i-th element.
            #   or in other words: attn_queries_indices is [0, 1, 2, ..., max(indices)]
            # scores(i) = softmax(<scattered_values[j] * W_attn * attn_queries[i] | j's s.t. indices[j]=i>)
            # out[i] = scores(i) ELEM_MUL <scattered_values[j] | j's s.t. indices[j]=i>
            nr_elements = queries.size(0)
            scattered_attn_queries_projected = torch.gather(
                attn_queries_projected, dim=0,
                index=indices.unsqueeze(-1).expand(indices.size(0), attn_queries_projected.size(1)))
            assert scattered_attn_queries_projected.size() == (scattered_values.size(0), self.qk_proj_dim)
            scattered_keys = scattered_values if self.key_projection_layer is None else \
                self.key_projection_layer(scattered_values)
            assert scattered_keys.size() == (scattered_values.size(0), self.qk_proj_dim)
            probs_scale_factor = self.qk_proj_dim ** (-0.5)
            # scattered_probs = torch.sum(scattered_keys * scattered_attn_queries_projected, dim=1) * probs_scale_factor
            scattered_probs = \
                torch.einsum('ij,ij->i', scattered_keys, scattered_attn_queries_projected) * probs_scale_factor
            assert scattered_probs.ndim == 1 and scattered_probs.size() == (scattered_values.size(0),)
            # TODO: Is it ok calling `scatter_softmax()` when not all of the range
            #       [0, 1, 2, ..., attn_queries.size(0) - 1] occurs in `indices`?
            #       What is entered to the missing-indices-entries in the output vector?
            #       Are they zeroed (the good case) or just skipped (bad case because of
            #       incorrect offsetting of the following entries)?
            scattered_scores = scatter_softmax(src=scattered_probs, index=indices, dim=-1)
            scattered_scores = self.attn_scores_dropout(scattered_scores)
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
        else:
            # means that queries[i] consists the query vector of the j-th element, where j = attn_queries_indices[i].
            # scores(i) = softmax(<scattered_values[j] * W_attn * attn_queries[i] | j's s.t. indices[j]=attn_queries_indices[i]>)
            # out[i] = scores(i) ELEM_MUL <scattered_values[j] | j's s.t. indices[j]=attn_queries_indices[i]>
            # Note: This case is much more complex, as there might be multiple different queries for a certain index.
            #       Hence, we have to calculate a scores-vector (of all occurrences of an index) per each given query.
            raise NotImplementedError  # TODO: implement
