import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from ndfa.nn_utils.misc import get_activation_layer
from ndfa.nn_utils.attention import Attention


__all__ = ['SequenceCombiner']


class SequenceCombiner(nn.Module):
    def __init__(self, encoding_dim: int, combined_dim: int,
                 nr_attn_heads: int = 1, nr_dim_reduction_layers: int = 1,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(SequenceCombiner, self).__init__()
        self.activation_layer = get_activation_layer(activation_fn)()
        self.encoding_dim = encoding_dim
        self.combined_dim = combined_dim
        self.nr_attn_heads = nr_attn_heads
        self.nr_dim_reduction_layers = nr_dim_reduction_layers
        assert nr_dim_reduction_layers >= 1
        self.attn_layers = nn.ModuleList([
            Attention(nr_features=self.encoding_dim,
                      project_key=True, activation_fn=activation_fn)
            for _ in range(nr_attn_heads)])
        assert encoding_dim * nr_attn_heads >= combined_dim
        attn_cat_dim = encoding_dim * nr_attn_heads
        projection_dimensions = np.linspace(
            start=attn_cat_dim, stop=combined_dim, num=nr_dim_reduction_layers + 1, dtype=int)
        self.dim_reduction_projection_layers = nn.ModuleList([
            nn.Linear(in_features=projection_dimensions[layer_idx],
                      out_features=projection_dimensions[layer_idx + 1])
            for layer_idx in range(nr_dim_reduction_layers)])
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, sequence_encodings: torch.Tensor,
                sequence_mask: Optional[torch.BoolTensor] = None,
                sequence_lengths: Optional[torch.LongTensor] = None,
                batch_first: bool = True):
        if not batch_first:
            raise NotImplementedError
        assert sequence_mask is not None or sequence_lengths is not None
        if sequence_mask is not None or sequence_lengths is None:
            raise NotImplementedError  # TODO: calc `sequence_lengths` from `sequence_mask`
        # sequence_encodings: (bsz, max_seq_len, encoding_dim)
        # sequence_lengths: (bsz, )
        last_word_indices = (sequence_lengths - 1).view(sequence_encodings.size(0), 1, 1).expand(sequence_encodings.size(0), 1, sequence_encodings.size(2))
        last_word_encoding = torch.gather(
            sequence_encodings, dim=1, index=last_word_indices).squeeze(1)
        attn_heads = torch.cat([
            attn_layer(sequences=sequence_encodings,
                       attn_key_from=last_word_encoding,
                       mask=sequence_mask, lengths=sequence_lengths)
            for attn_layer in self.attn_layers], dim=-1)
        projected = attn_heads
        for dim_reduction_projection_layer in self.dim_reduction_projection_layers:
            projected = self.dropout_layer(self.activation_layer(dim_reduction_projection_layer(projected)))
        assert projected.size(-1) == self.combined_dim
        return projected
