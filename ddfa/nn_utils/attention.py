import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Attention(nn.Module):
    def __init__(self, nr_features: int, project_key: bool = True, key_in_features: Optional[int] = None):
        super(Attention, self).__init__()
        self.nr_features = nr_features
        self.key_linear_projection_layer = \
            nn.Linear(in_features=nr_features if key_in_features is None else key_in_features,
                      out_features=nr_features) if project_key else None

    def forward(self, sequences: torch.Tensor, attn_key_from: Optional[torch.Tensor] = None,
                attn_weights: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        assert (attn_key_from is None) ^ (attn_weights is None)
        assert attn_weights is None or self.key_linear_projection_layer is None
        assert len(sequences.size()) == 3  # (bsz, seq_len, nr_features)
        batch_size, seq_len, nr_features = sequences.size()
        assert attn_key_from is None or attn_key_from.size() == (batch_size, nr_features)
        assert attn_weights is None or attn_weights.size() == (batch_size, seq_len)
        assert nr_features == self.nr_features
        if mask.size() != (batch_size, seq_len):
            print(mask.size())
            print((batch_size, seq_len))
        assert mask is None or mask.size() == (batch_size, seq_len)

        if attn_key_from is not None:
            attn_key_vector = F.relu(self.key_linear_projection_layer(attn_key_from)) \
                if self.key_linear_projection_layer else attn_key_from  # (bsz, nr_features)
            attn_weights = torch.bmm(
                sequences.flatten(0, 1).unsqueeze(dim=1),  # (bsz * seq_len, 1, nr_features)
                attn_key_vector.unsqueeze(dim=1).expand(batch_size, seq_len, self.nr_features)
                    .flatten(0, 1).unsqueeze(dim=-1))  # (bsz * seq_len, nr_features, 1)
            assert attn_weights.size() == (batch_size * seq_len, 1, 1)
            attn_weights = attn_weights.view(batch_size, seq_len)

        if mask is not None:
            attn_weights = attn_weights + torch.where(
                mask,  # (bsz, seq_len)
                torch.zeros(1, dtype=torch.float, device=attn_weights.device),
                torch.full(size=(1,), fill_value=float('-inf'), dtype=torch.float, device=attn_weights.device))
        attn_probs = F.softmax(attn_weights, dim=1)  # (bsz, seq_len)
        # (bsz, 1, seq_len) * (bsz, seq_len, nr_features) -> (bsz, 1, nr_features)
        attn_applied = torch.bmm(attn_probs.unsqueeze(1), sequences).squeeze(1)  # (bsz, nr_features)
        return attn_applied
