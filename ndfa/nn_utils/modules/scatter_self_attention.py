import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_scatter import scatter_sum, scatter_softmax


__all__ = ['ScatterSelfAttention']


class ScatterSelfAttention(nn.Module):
    def __init__(self, dim: int):
        super(ScatterSelfAttention, self).__init__()
        self.dim = dim
        self.K = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.Q = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.V = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)

    def forward(self, scattered_values: torch.Tensor, indices: torch.LongTensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert scattered_values.ndim == 2
        assert scattered_values.size(1) == self.values_dim
        assert indices.size() == (scattered_values.size(0),)

        scattered_K = self.K(scattered_values)
        scattered_Q = self.Q(scattered_values)
        scattered_V = self.V(scattered_values)

        # scores(r,s) = softmax(<scattered_values[j] * W_attn * attn_keys[i] | j's s.t. indices[j]=attn_keys_indices[i]>)
        # out[i] = scores(i) ELEM_MUL <scattered_values[j] | j's s.t. indices[j]=attn_keys_indices[i]>
        # Note: This case is much more complex, as there might be multiple different keys for a certain index.
        #       Hence, we have to calculate a scores-vector (of all occurrences of an index) per each given key.

        raise NotImplementedError  # TODO: implement!
