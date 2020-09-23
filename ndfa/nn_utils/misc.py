import torch


__all__ = ['get_activation']


def get_activation(activation_str: str):
    activations = {'relu': torch.relu, 'prelu': torch.prelu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh}
    if activation_str not in activations:
        raise ValueError(
            f'`{activation_str}` is not a valid activation function (use one of these: {activations.keys()}).')
    return activations[activation_str]
