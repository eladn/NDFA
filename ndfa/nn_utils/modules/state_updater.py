import torch
import torch.nn as nn
from typing import Optional

from ndfa.nn_utils.modules.gate import Gate
from ndfa.nn_utils.modules.params.state_updater_params import StateUpdaterParams


__all__ = ['StateUpdater']


class StateUpdater(nn.Module):
    def __init__(
            self, state_dim: int, params: StateUpdaterParams, update_dim: Optional[int] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(StateUpdater, self).__init__()
        self.state_dim = state_dim
        self.update_dim = self.state_dim if update_dim is None else update_dim
        self.params = params
        if self.params.method == 'cat-project':
            self.linear_projection_layer = nn.Linear(
                in_features=self.state_dim + self.update_dim, out_features=self.state_dim, bias=False)
        elif self.params.method == 'gate':
            self.gate = Gate(state_dim=self.state_dim, update_dim=self.update_dim,
                             dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.params.method == 'add':
            assert self.state_dim == self.update_dim
            pass  # we don't need anything in this case ...
        else:
            raise ValueError(f'Unsupported update method `{self.params.method}`.')

    def forward(self, previous_state: torch.Tensor, state_update: torch.Tensor):
        if self.params.method == 'cat-project':
            return self.linear_projection_layer(torch.cat([previous_state, state_update], dim=-1))
        elif self.params.method == 'gate':
            return self.gate(previous_state=previous_state, state_update=state_update)
        elif self.params.method == 'add':
            return previous_state + state_update
        else:
            assert False
