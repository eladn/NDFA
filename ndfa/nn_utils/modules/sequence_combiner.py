import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import numpy as np
from typing import Optional

from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.modules.attention import Attention
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams


__all__ = ['SequenceCombiner']


class SequenceCombiner(nn.Module):
    def __init__(self, encoding_dim: int,
                 combiner_params: SequenceCombinerParams,
                 combined_dim: Optional[int] = None,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(SequenceCombiner, self).__init__()
        self.encoding_dim = encoding_dim
        self.combined_dim = self.encoding_dim if combined_dim is None else combined_dim
        self.combiner_params = combiner_params

        # self.multihead_attn_layer = nn.MultiheadAttention(
        #     embed_dim=self.encoding_dim,
        #     num_heads=self.combiner_params.nr_attn_heads,
        #     dropout=0.1)  # TODO: get dropout rate from HPs

        if self.combiner_params.method == 'attn':
            self.multihead_attn_layer = Attention(
                in_embed_dim=self.encoding_dim,
                out_embed_dim=self.combined_dim,
                project_key=True, project_query=True,
                project_values=self.combiner_params.project_attn_values,
                nr_heads=self.combiner_params.nr_attn_heads,
                activation_fn=activation_fn)
            assert encoding_dim * self.combiner_params.nr_attn_heads >= self.combined_dim
            assert self.combiner_params.nr_dim_reduction_layers >= 1
            projection_dimensions = np.linspace(
                start=self.combined_dim, stop=self.combined_dim,
                num=self.combiner_params.nr_dim_reduction_layers + 1, dtype=int)
            self.dim_reduction_projection_layers = nn.ModuleList([
                nn.Linear(in_features=projection_dimensions[layer_idx],
                          out_features=projection_dimensions[layer_idx + 1])
                for layer_idx in range(self.combiner_params.nr_dim_reduction_layers)])
            self.layer_norms = nn.ModuleList([
                LayerNorm(projection_dimensions[layer_idx], elementwise_affine=False)
                for layer_idx in range(self.combiner_params.nr_dim_reduction_layers)])
        elif self.combiner_params.method in {'mean', 'sum', 'last', 'ends'}:
            assert self.combined_dim == self.encoding_dim
        else:
            raise ValueError(f'Unsupported combining method `{self.combiner_params.method}`.')

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, sequence_encodings: torch.Tensor,
                sequence_mask: Optional[torch.BoolTensor] = None,
                sequence_lengths: Optional[torch.LongTensor] = None,
                batch_first: bool = True):
        if self.combiner_params.method in {'mean', 'sum'}:
            if sequence_mask is not None:
                sequence_encodings = sequence_encodings.masked_fill(~sequence_mask.unsqueeze(-1), 0)
            elif sequence_lengths is not None:
                raise NotImplementedError  # TODO: impl!
            seq_dim = 1 if batch_first else 0
            agg_fns = {'sum': torch.sum, 'mean': torch.mean}
            return agg_fns[self.combiner_params.method](sequence_encodings, dim=seq_dim)

        if not batch_first:
            raise NotImplementedError
        assert sequence_mask is not None or sequence_lengths is not None
        if sequence_mask is not None and sequence_lengths is None:
            raise NotImplementedError  # TODO: calc `sequence_lengths` from `sequence_mask`
        # sequence_encodings: (bsz, max_seq_len, encoding_dim)
        # sequence_lengths: (bsz, )
        last_word_indices = (sequence_lengths - 1).view(sequence_encodings.size(0), 1, 1)\
            .expand(sequence_encodings.size(0), 1, sequence_encodings.size(2))
        last_word_encoding = torch.gather(
            sequence_encodings, dim=1, index=last_word_indices).squeeze(1)
        if self.combiner_params.method == 'last':
            return last_word_encoding
        first_word_encoding = sequence_encodings[:, 0, :]
        both_end_words_encoding = (first_word_encoding + last_word_encoding)
        if self.combiner_params.method == 'ends':
            return both_end_words_encoding
        # attn_output = self.multihead_attn_layer(
        #     query=both_end_words_encoding.unsqueeze(0),
        #     key=sequence_encodings.transpose(0, 1),
        #     value=sequence_encodings.transpose(0, 1),
        #     key_padding_mask=None if sequence_mask is None else ~sequence_mask)[0].view(batch_size, self.encoding_dim)
        attn_output = self.multihead_attn_layer(
            sequences=sequence_encodings,
            query=both_end_words_encoding,
            mask=sequence_mask,
            lengths=sequence_lengths)
        projected = attn_output
        for dim_reduction_projection_layer, layer_norm in zip(self.dim_reduction_projection_layers, self.layer_norms):
            projected = layer_norm(projected)
            projected = self.dropout_layer(self.activation_layer(dim_reduction_projection_layer(projected)))
        assert projected.size(-1) == self.combined_dim
        return projected
