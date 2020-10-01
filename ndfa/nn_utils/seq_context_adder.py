import torch
import torch.nn as nn
from typing import Optional

from ndfa.nn_utils.misc import get_activation_layer


__all__ = ['SeqContextAdder']


class SeqContextAdder(nn.Module):
    def __init__(self, main_dim: int, ctx_dim: int,
                 method: str = 'parallel', ctx_dim_reduction_rate: float = 0.25,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(SeqContextAdder, self).__init__()
        self.main_dim = main_dim
        self.ctx_dim = ctx_dim
        assert method in {'parallel', 'series'}
        self.method = method
        if self.method == 'parallel':
            self.ctx_reductioned_dim = min(self.ctx_dim, int(self.main_dim * ctx_dim_reduction_rate))
            self.ctx_dim_reduction_layer = nn.Linear(
                in_features=self.ctx_dim, out_features=self.ctx_reductioned_dim)
            projections_inbetween_dim = int(((self.main_dim + self.ctx_reductioned_dim) + self.main_dim) / 2)
            self.first_common_projection_layer = nn.Linear(
                in_features=self.main_dim + self.ctx_reductioned_dim, out_features=projections_inbetween_dim)
            self.second_common_linear_layer = nn.Linear(
                in_features=projections_inbetween_dim, out_features=self.main_dim)
        elif self.method == 'series':
            self.ctx_to_seq_token_projection_layer = nn.Linear(
                in_features=self.ctx_dim, out_features=self.main_dim)
        else:
            raise ValueError(f'Unsupported method {self.method}.')
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, sequence: torch.Tensor, context: torch.Tensor,
                sequence_mask: Optional[torch.BoolTensor] = None,
                sequence_lengths: Optional[torch.LongTensor] = None,
                batch_first: bool = True):
        assert batch_first
        assert sequence.ndim == 3
        assert sequence.size(2) == self.main_dim
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        assert (batch_size, self.ctx_dim) == context.size()
        if self.method == 'parallel':
            context_reductioned = self.ctx_dim_reduction_layer(context)
            ctx_parallely_expanded = context_reductioned.unsqueeze(1).expand(
                batch_size, seq_len, self.ctx_reductioned_dim)
            sequence_with_ctx = torch.cat([sequence, ctx_parallely_expanded], dim=-1)
            projected = self.dropout_layer(self.activation_layer(self.first_common_projection_layer(sequence_with_ctx)))
            final = self.second_common_linear_layer(projected)
            assert final.size() == sequence.size()
            if sequence_mask is None and sequence_lengths is not None:
                raise NotImplementedError  # TODO: calc sequence_mask from sequence_lengths
            if sequence_mask is not None:
                final = torch.zeros_like(final).masked_scatter(sequence_mask.unsqueeze(-1).expand(final.size()), final)
            # TODO: use AddNorm for this skip-connection
            final = final + sequence  # skip connection
            return final
        elif self.method == 'series':
            raise NotImplementedError  # TODO: impl
        else:
            assert False
