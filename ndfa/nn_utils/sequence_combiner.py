import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from ndfa.nn_utils.misc import get_activation_layer
from ndfa.nn_utils.attention import Attention
from ndfa.ndfa_model_hyper_parameters import SequenceCombinerParams


__all__ = ['SequenceCombiner']


class SequenceCombiner(nn.Module):
    def __init__(self, encoding_dim: int, combined_dim: int,
                 combiner_params: SequenceCombinerParams,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(SequenceCombiner, self).__init__()
        self.activation_layer = get_activation_layer(activation_fn)()
        self.encoding_dim = encoding_dim
        self.combined_dim = combined_dim
        self.combiner_params = combiner_params
        if self.combiner_params.method != 'attn':
            raise NotImplementedError  # TODO: impl!
        assert self.combiner_params.nr_dim_reduction_layers >= 1
        self.attn_layers = nn.ModuleList([
            Attention(nr_features=self.encoding_dim,
                      project_key=True, activation_fn=activation_fn)
            for _ in range(self.combiner_params.nr_attn_heads)])
        assert encoding_dim * self.combiner_params.nr_attn_heads >= combined_dim
        attn_cat_dim = encoding_dim * self.combiner_params.nr_attn_heads
        projection_dimensions = np.linspace(
            start=attn_cat_dim, stop=combined_dim,
            num=self.combiner_params.nr_dim_reduction_layers + 1, dtype=int)
        self.dim_reduction_projection_layers = nn.ModuleList([
            nn.Linear(in_features=projection_dimensions[layer_idx],
                      out_features=projection_dimensions[layer_idx + 1])
            for layer_idx in range(self.combiner_params.nr_dim_reduction_layers)])
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, sequence_encodings: torch.Tensor,
                sequence_mask: Optional[torch.BoolTensor] = None,
                sequence_lengths: Optional[torch.LongTensor] = None,
                batch_first: bool = True):
        if self.combiner_params.method != 'attn':
            raise NotImplementedError  # TODO: impl!
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
        first_word_encoding = sequence_encodings[:, 0, :]
        attn_key_from = first_word_encoding + last_word_encoding
        attn_heads = torch.cat([
            attn_layer(sequences=sequence_encodings,
                       attn_key_from=attn_key_from,
                       mask=sequence_mask, lengths=sequence_lengths)
            for attn_layer in self.attn_layers], dim=-1)
        projected = attn_heads
        for dim_reduction_projection_layer in self.dim_reduction_projection_layers:
            projected = self.dropout_layer(self.activation_layer(dim_reduction_projection_layer(projected)))
        assert projected.size(-1) == self.combined_dim
        return projected
