import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ndfa.nn_utils.misc.misc import get_activation_layer, seq_lengths_to_mask


__all__ = ['Attention']


class Attention(nn.Module):
    def __init__(self, in_embed_dim: int, out_embed_dim: Optional[int] = None,
                 project_key: bool = True, project_query: bool = True,
                 query_in_embed_dim: Optional[int] = None, project_values: bool = False,
                 activation_fn: str = 'relu'):
        super(Attention, self).__init__()
        self.activation_layer = get_activation_layer(activation_fn)()
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = self.in_embed_dim if out_embed_dim is None else out_embed_dim
        self.query_in_embed_dim = self.in_embed_dim if query_in_embed_dim is None else query_in_embed_dim
        assert (self.in_embed_dim == self.query_in_embed_dim) or project_query
        self.query_linear_projection_layer = \
            nn.Linear(in_features=self.query_in_embed_dim, out_features=self.in_embed_dim) if project_query else None
        self.key_linear_projection_layer = \
            nn.Linear(in_features=self.in_embed_dim, out_features=self.in_embed_dim) if project_key else None
        assert (self.in_embed_dim == self.out_embed_dim) or project_values
        self.value_linear_projection_layer = \
            nn.Linear(in_features=self.in_embed_dim, out_features=self.out_embed_dim) if project_values else None

    def forward(self, sequences: torch.Tensor, attn_query_from: Optional[torch.Tensor] = None,
                attn_weights: Optional[torch.Tensor] = None, mask: Optional[torch.BoolTensor] = None,
                lengths: Optional[torch.LongTensor] = None):
        assert (attn_query_from is None) ^ (attn_weights is None)
        assert attn_weights is None or self.query_linear_projection_layer is None
        assert sequences.ndim == 3  # (bsz, seq_len, in_embed_dim)
        batch_size, seq_len, in_embed_dim = sequences.size()
        assert attn_query_from is None or attn_query_from.size() == (batch_size, self.query_in_embed_dim)
        assert attn_weights is None or attn_weights.size() == (batch_size, seq_len)
        assert in_embed_dim == self.in_embed_dim

        assert mask is None or mask.size() == (batch_size, seq_len)
        assert lengths is None or lengths.size() == (batch_size,)
        if lengths is not None and mask is None:
            mask = seq_lengths_to_mask(seq_lengths=lengths, max_seq_len=seq_len)

        if attn_query_from is not None:
            attn_query_vector = self.activation_layer(self.query_linear_projection_layer(attn_query_from)) \
                if self.query_linear_projection_layer is not None else attn_query_from  # (bsz, in_embed_dim)
            seq_keys = self.activation_layer(self.key_linear_projection_layer(sequences.flatten(0, 1))) \
                if self.key_linear_projection_layer is not None else sequences.flatten(0, 1)  # (bsz * seq_len, in_embed_dim)
            assert seq_keys.size() == (batch_size * seq_len, in_embed_dim)
            attn_weights = torch.bmm(
                seq_keys.unsqueeze(dim=1),  # (bsz * seq_len, 1, in_embed_dim)
                attn_query_vector.unsqueeze(dim=1).expand(batch_size, seq_len, self.in_embed_dim)
                    .flatten(0, 1).unsqueeze(dim=-1))  # (bsz * seq_len, in_embed_dim, 1)
            assert attn_weights.size() == (batch_size * seq_len, 1, 1)
            attn_weights = attn_weights.view(batch_size, seq_len)
            # scale attn_weights by 1/sqrt(d_v) as done by Attention Is All You Need
            attn_weights_scale_factor = self.in_embed_dim ** (-0.5)
            attn_weights = attn_weights * attn_weights_scale_factor

        if mask is not None:
            attn_weights = attn_weights + torch.where(
                mask,  # (bsz, seq_len)
                attn_weights.new_zeros(size=(1,)),
                attn_weights.new_full(size=(1,), fill_value=float('-inf')))
        attn_probs = F.softmax(attn_weights, dim=1)  # (bsz, seq_len)
        values = sequences if self.value_linear_projection_layer is None else \
            self.value_linear_projection_layer(sequences)
        # (bsz, 1, seq_len) * (bsz, seq_len, out_embed_dim) -> (bsz, 1, out_embed_dim)
        attn_applied = torch.bmm(attn_probs.unsqueeze(1), values).squeeze(1)  # (bsz, out_embed_dim)
        assert attn_applied.size() == (batch_size, self.out_embed_dim)
        return attn_applied
