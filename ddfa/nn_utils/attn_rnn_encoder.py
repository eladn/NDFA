import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional

from ddfa.nn_utils.attention import Attention


__all__ = ['AttnRNNEncoder']


class AttnRNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, rnn_type: str = 'lstm',
                 nr_rnn_layers: int = 2, rnn_bi_direction: bool = True):
        assert rnn_type in {'lstm', 'gru'}
        super(AttnRNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.nr_rnn_layers = nr_rnn_layers
        self.nr_rnn_directions = 2 if rnn_bi_direction else 1
        rnn_type = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn_layer = rnn_type(
            input_size=self.input_dim, hidden_size=self.hidden_dim,
            bidirectional=rnn_bi_direction, num_layers=self.nr_rnn_layers)
        self.attn_layer = Attention(nr_features=self.hidden_dim, project_key=True)

    def forward(self, sequence_input: torch.Tensor, mask: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None, batch_first: bool = False):
        assert len(sequence_input.size()) == 3 and sequence_input.size()[2] == self.input_dim
        if batch_first:
            # TODO: instead of permute in input / output, pass `batch_first` to `pack_padded_sequence()` &
            #  `pad_packed_sequence()` and fix rest of the code to adapt it.
            sequence_input = sequence_input.permute(1, 0, 2)  # (seq_len, bsz, input_dim)
        seq_len, batch_size = sequence_input.size()[:2]
        assert mask is None or mask.size() == (batch_size, seq_len) and mask.dtype == torch.bool
        assert mask is None or lengths is None
        assert lengths is None or lengths.size() == (batch_size,) and lengths.dtype == torch.long
        if mask is not None:
            lengths = None if mask is None else mask.long().sum(dim=1)
            lengths = torch.where(lengths <= torch.zeros(1, dtype=torch.long, device=lengths.device),
                                  torch.ones(1, dtype=torch.long, device=lengths.device), lengths)
        elif lengths is not None:
            batched_ranges = torch.arange(start=1, end=seq_len + 1)\
                .unsqueeze(0).expand(batch_size, seq_len)
            mask = (batched_ranges <= lengths.unsqueeze(-1).expand(batch_size, seq_len))

        packed_input = pack_padded_sequence(sequence_input, lengths=lengths, enforce_sorted=False)
        rnn_outputs, (last_hidden_out, _) = self.rnn_layer(packed_input)
        assert last_hidden_out.size() == \
               (self.nr_rnn_layers * self.nr_rnn_directions, batch_size, self.hidden_dim)
        last_hidden_out = last_hidden_out\
            .view(self.nr_rnn_layers, self.nr_rnn_directions, batch_size, self.hidden_dim)[-1, :, :, :].squeeze(0)
        last_hidden_out = last_hidden_out.sum(dim=0) if self.nr_rnn_directions > 1 else last_hidden_out.squeeze(0)
        assert last_hidden_out.size() == (batch_size, self.hidden_dim)

        rnn_outputs, _ = pad_packed_sequence(sequence=rnn_outputs)
        assert rnn_outputs.size() == (seq_len, batch_size, self.nr_rnn_directions * self.hidden_dim)
        if self.nr_rnn_directions > 1:
            rnn_outputs = rnn_outputs \
                .view(seq_len, batch_size, self.nr_rnn_directions, self.hidden_dim).sum(dim=-2)
        rnn_outputs = rnn_outputs.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        assert rnn_outputs.size() == (batch_size, seq_len, self.hidden_dim)

        outputs = self.attn_layer(sequences=rnn_outputs, attn_key_from=last_hidden_out, mask=mask)
        assert outputs.size() == (batch_size, self.hidden_dim)

        if not batch_first:
            # TODO: instead of permute in input / output, pass `batch_first` to `pack_padded_sequence()` &
            #  `pad_packed_sequence()` and fix rest of the code to adapt it.
            outputs = outputs.permute(1, 0, 2)  # (seq_len, batch_size, self.input_dim)
        return outputs
