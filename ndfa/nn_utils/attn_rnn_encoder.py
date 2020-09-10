import torch
import torch.nn as nn
from typing import Optional

from ndfa.nn_utils.rnn_encoder import RNNEncoder
from ndfa.nn_utils.attention import Attention


__all__ = ['AttnRNNEncoder']


class AttnRNNEncoder(RNNEncoder):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, rnn_type: str = 'lstm',
                 nr_rnn_layers: int = 2, rnn_bi_direction: bool = True):
        super(AttnRNNEncoder, self).__init__(
            input_dim=input_dim, hidden_dim=hidden_dim, rnn_type=rnn_type,
            nr_rnn_layers=nr_rnn_layers, rnn_bi_direction=rnn_bi_direction)
        self.attn_layer = Attention(nr_features=self.hidden_dim, project_key=True)

    def forward(self, sequence_input: torch.Tensor, mask: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None, batch_first: bool = False):
        last_hidden_out, rnn_outputs = super(AttnRNNEncoder, self).forward(
            sequence_input=sequence_input, mask=mask, lengths=lengths)

        batch_size = rnn_outputs.size(0 if batch_first else 1)
        seq_len = rnn_outputs.size(1 if batch_first else 0)
        if lengths is not None and mask is None:
            batched_ranges = torch.arange(start=1, end=seq_len + 1, dtype=torch.long, device=lengths.device) \
                .unsqueeze(0).expand(batch_size, seq_len)
            mask = (batched_ranges <= lengths.unsqueeze(-1).expand(batch_size, seq_len))

        merged_rnn_outputs = self.attn_layer(
            sequences=rnn_outputs, attn_key_from=last_hidden_out, mask=mask)
        assert merged_rnn_outputs.size() == (batch_size, self.hidden_dim)

        return merged_rnn_outputs, rnn_outputs
