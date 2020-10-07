import torch
import torch.nn as nn

from ndfa.nn_utils.misc import get_activation_layer


__all__ = ['Gate']


class Gate(nn.Module):
    def __init__(self, state_dim: int, update_dim: int,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(Gate, self).__init__()
        self.state_dim = state_dim
        self.update_dim = update_dim
        self.forget_gate = nn.Linear(
            in_features=self.state_dim + self.update_dim,
            out_features=self.state_dim)
        self.update_gate = nn.Linear(
            in_features=self.state_dim + self.update_dim,
            out_features=self.state_dim)
        self.update_linear_project = nn.Linear(
            in_features=self.state_dim + self.update_dim,
            out_features=self.state_dim)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, previous_state: torch.Tensor, state_update: torch.Tensor):
        previous_state_with_update = torch.cat([previous_state, state_update], dim=-1)
        forget_gate = torch.sigmoid(self.forget_gate(previous_state_with_update))
        add_gate = torch.sigmoid(self.update_gate(previous_state_with_update))
        add_data = self.activation_layer(self.update_linear_project(previous_state_with_update))
        assert previous_state.shape == forget_gate.shape == add_gate.shape == add_data.shape
        new_state = (previous_state * forget_gate) + (add_data * add_gate)
        assert new_state.shape == previous_state.shape
        return new_state
