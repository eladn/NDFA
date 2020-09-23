import torch
import torch.nn.functional as F


__all__ = ['get_activation']


def get_activation(activation_str: str):
    activations = {'relu': F.relu, 'prelu': F.prelu, 'sigmoid': torch.sigmoid}
    if activation_str not in activations:
        raise ValueError(
            f'`{activation_str}` is not a valid activation function (use one of these: {activations.keys()}).')
    return activations[activation_str]
