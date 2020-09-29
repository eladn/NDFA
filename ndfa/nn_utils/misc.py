import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['get_activation_layer', 'get_activation_fn']


def get_activation_layer(activation_str: str):
    activation_layers = {
        'relu': nn.ReLU, 'prelu': nn.PReLU, 'leaky-relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'none': nn.Identity}
    if activation_str not in activation_layers:
        raise ValueError(
            f'`{activation_str}` is not a valid activation function (use one of these: {activation_layers.keys()}).')
    return activation_layers[activation_str]


def get_activation_fn(activation_str: str):
    activation_fns = {
        'relu': torch.relu, 'prelu': torch.prelu, 'leaky-relu': F.leaky_relu,
        'sigmoid': torch.sigmoid, 'tanh': torch.tanh, 'none': lambda x: x}
    if activation_str not in activation_fns:
        raise ValueError(
            f'`{activation_str}` is not a valid activation function (use one of these: {activation_fns.keys()}).')
    return activation_fns[activation_str]
