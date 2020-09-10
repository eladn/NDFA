import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SeqContextAdder']


class SeqContextAdder(nn.Module):
    def __init__(self, main_dim: int, ctx_dim: int, dropout_p: float = 0.3):
        super(SeqContextAdder, self).__init__()
        self.main_dim = main_dim
        self.ctx_dim = ctx_dim
        self.first_projection_layer = nn.Linear(
            in_features=self.main_dim + self.ctx_dim, out_features=self.main_dim)
        self.second_linear_layer = nn.Linear(in_features=self.main_dim, out_features=self.main_dim)
        self.dropout_layer = nn.Dropout(p=dropout_p)

    def forward(self, sequence: torch.Tensor, context: torch.Tensor, batch_first: bool = True):
        assert batch_first
        assert sequence.ndim == 3
        assert sequence.size(2) == self.main_dim
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        assert (batch_size, self.ctx_dim) == context.size()
        ctx_expanded = context.unsqueeze(1).expand(batch_size, seq_len, self.ctx_dim)
        sequence_with_ctx = torch.cat([sequence, ctx_expanded], dim=-1)
        projected = self.dropout_layer(F.relu(self.first_projection_layer(sequence_with_ctx)))
        return self.dropout_layer(F.relu(self.second_linear_layer(projected)))
