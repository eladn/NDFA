import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Optional


__all__ = [
    'get_activation_layer', 'get_activation_fn', 'seq_lengths_to_mask',
    'EncodingsUpdater', 'apply_disjoint_updates_to_encodings_tensor']


def get_activation_layer(activation_str: str):
    activation_layers = {
        'relu': nn.ReLU, 'prelu': nn.PReLU, 'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'none': nn.Identity}
    if activation_str not in activation_layers:
        raise ValueError(
            f'`{activation_str}` is not a valid activation function (use one of these: {activation_layers.keys()}).')
    return activation_layers[activation_str]


def get_activation_fn(activation_str: str):
    activation_fns = {
        'relu': torch.relu, 'prelu': torch.prelu, 'leaky_relu': F.leaky_relu,
        'sigmoid': torch.sigmoid, 'tanh': torch.tanh, 'none': lambda x: x}
    if activation_str not in activation_fns:
        raise ValueError(
            f'`{activation_str}` is not a valid activation function (use one of these: {activation_fns.keys()}).')
    return activation_fns[activation_str]


def seq_lengths_to_mask(seq_lengths: torch.LongTensor, max_seq_len: int, batch_first: bool = True):
    assert batch_first
    batch_size = seq_lengths.size(0)
    batched_ranges = torch.arange(start=1, end=max_seq_len + 1, dtype=torch.long, device=seq_lengths.device) \
        .unsqueeze(0).expand(batch_size, max_seq_len)
    mask = (batched_ranges <= seq_lengths.unsqueeze(-1).expand(batch_size, max_seq_len))
    return mask


EncodingsUpdater = namedtuple('EncodingsUpdater', ['new_embeddings', 'element_indices', 'linear_projection_layer'])


def apply_disjoint_updates_to_encodings_tensor(
        original_encodings: torch.Tensor, *updaters: EncodingsUpdater,
        otherwise_linear: Optional[nn.Linear] = None):
    assert original_encodings.ndim == 2
    assert all(updater.new_embeddings.ndim == 2 for updater in updaters)
    assert all(updater.element_indices.ndim == 1 for updater in updaters)
    updated_encodings = torch.cat([
        updater.linear_projection_layer(torch.cat([
            original_encodings[updater.element_indices], updater.new_embeddings], dim=1))
        for updater in updaters], dim=0)
    all_element_indices_to_update = torch.cat([updater.element_indices for updater in updaters], dim=0)
    assert updated_encodings.size(0) == all_element_indices_to_update.size(0)
    all_element_indices_to_update = all_element_indices_to_update.unsqueeze(-1).expand(updated_encodings.shape)
    if otherwise_linear is not None:
        original_encodings = otherwise_linear(original_encodings)
    return original_encodings.scatter(
        dim=0, index=all_element_indices_to_update, src=updated_encodings)
