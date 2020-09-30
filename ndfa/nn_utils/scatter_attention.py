import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_scatter import scatter_sum, scatter_softmax


__all__ = ['ScatterAttention']


class ScatterAttention(nn.Module):
    def __init__(self, values_dim: int, keys_dim: Optional[int] = None):
        super(ScatterAttention, self).__init__()
        self.values_dim = values_dim
        self.keys_dim = values_dim if keys_dim is None else keys_dim
        self.scores_linear_key_projection_layer = nn.Linear(
            in_features=self.keys_dim, out_features=self.values_dim)

    def forward(self, scattered_values: torch.Tensor, indices: torch.LongTensor,
                attn_keys: torch.Tensor, attn_keys_indices: Optional[torch.LongTensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert scattered_values.ndim == 2
        assert scattered_values.size(1) == self.values_dim
        assert indices.size() == (scattered_values.size(0),)
        assert attn_keys.ndim == 2
        assert attn_keys.size(1) == self.keys_dim
        assert attn_keys_indices is None or attn_keys_indices.size() == (attn_keys.size(0),)

        attn_keys_projected = self.scores_linear_key_projection_layer(attn_keys)

        if attn_keys_indices is None:
            # means attn_keys_indices is [0, 1, 2, ..., max(indices)]
            # scores(i) = softmax(<scattered_values[j] * W_attn * attn_keys[i] | j's s.t. indices[j]=i>)
            # out[i] = scores(i) ELEM_MUL <scattered_values[j] | j's s.t. indices[j]=i>
            scattered_attn_keys_projected = torch.gather(attn_keys_projected, dim=0, index=indices)
            assert scattered_attn_keys_projected.size() == scattered_values.size()
            scattered_probs = torch.tensordot(scattered_values, scattered_attn_keys_projected, dims=([1], [1]))
            assert scattered_probs.ndim == 1 and scattered_probs.size() == (scattered_values.size(0),)
            scattered_scores = scatter_softmax(src=scattered_probs, index=indices, dim=-1)
            assert scattered_scores.ndim == 1 and scattered_scores.size() == (scattered_values.size(0),)
            attn_applied = scatter_sum(torch.mul(scattered_values, scattered_scores), index=indices, dim=0)
            return scattered_scores, attn_applied
        else:
            # scores(i) = softmax(<scattered_values[j] * W_attn * attn_keys[i] | j's s.t. indices[j]=attn_keys_indices[i]>)
            # out[i] = scores(i) ELEM_MUL <scattered_values[j] | j's s.t. indices[j]=attn_keys_indices[i]>
            # Note: This case is much more complex, as there might be multiple different keys for a certain index.
            #       Hence, we have to calculate a scores-vector (of all occurrences of an index) per each given key.
            raise NotImplementedError  # TODO: implement
