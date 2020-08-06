import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Optional


__all__ = ['unflatten_batch', 'unflatten_batch2']


def unflatten_batch(flattened_data: torch.Tensor, examples_indices: torch.LongTensor) \
        -> Tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
    assert len(flattened_data.size()) == 1
    assert len(examples_indices.size()) == 1
    batch_range = torch.arange(1, examples_indices.size()[0] + 1, device=flattened_data.device)
    indices_of_last_item_per_example = batch_range[
        examples_indices != torch.cat((examples_indices[1:], examples_indices[-1].unsqueeze(-1) + 1))]
    nr_items_per_example = indices_of_last_item_per_example - torch.cat((torch.tensor([0]), indices_of_last_item_per_example[:-1]))
    # performance note: here we move data GPU->CPU and later back to GPU
    list_of_tensor_seq_per_example = torch.split(flattened_data, nr_items_per_example.tolist())
    batch_size = nr_items_per_example.size()[0]
    # packed_seq = pack_sequence(list_of_tensor_seq_per_example, enforce_sorted=False)
    # pad_packed_sequence(packed_seq, batch_first=True)
    unflattened_data = pad_sequence(list_of_tensor_seq_per_example, batch_first=True)
    max_nr_items_per_example = unflattened_data.size()[1]
    mask = nr_items_per_example.unsqueeze(-1).expand(batch_size, max_nr_items_per_example) >= \
           torch.arange(1, max_nr_items_per_example + 1, device=flattened_data.device).unsqueeze(0)\
               .expand(batch_size, max_nr_items_per_example)
    return pad_sequence(list_of_tensor_seq_per_example, batch_first=True), nr_items_per_example, mask


def unflatten_batch2(
        flattened_data: torch.Tensor, examples_indices: torch.LongTensor,
        batch_size: Optional[int] = None, batch_range: Optional[torch.Tensor] = None,
        max_nr_items_per_example: Optional[int] = None) \
        -> Tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
    assert len(flattened_data.size()) == 1
    assert len(examples_indices.size()) == 1
    if batch_size is None:
        batch_size = examples_indices[-1].item() + 1
    if batch_range is None:
        batch_range = torch.arange(1, examples_indices.size()[0] + 1, device=flattened_data.device)
    last_item_per_example_mask = \
        examples_indices != torch.cat((examples_indices[1:], examples_indices[-1].unsqueeze(-1) + 1))
    first_item_per_example_mask = \
        examples_indices != torch.cat((torch.tensor([-1]), examples_indices[:-1]))
    indices_of_last_item_per_example = batch_range[last_item_per_example_mask]  # alt: last_item_per_example_mask.nonzeros().unsqueeze(-1) + 1
    nr_items_per_example = \
        indices_of_last_item_per_example - \
        torch.cat((torch.tensor([0]), indices_of_last_item_per_example[:-1]))
    if max_nr_items_per_example is None:
        max_nr_items_per_example = torch.max(nr_items_per_example, dim=0).values.item()
    nr_items_per_last_example_in_first_item_of_each_example = torch.zeros_like(examples_indices).masked_scatter_(
        mask=first_item_per_example_mask,
        source=torch.cat((torch.tensor([0], dtype=torch.long), nr_items_per_example[:-1]), dim=0))
    range_per_example = batch_range - 1 + torch.cumsum(-nr_items_per_last_example_in_first_item_of_each_example, dim=0)
    new_scatter_indices = examples_indices * max_nr_items_per_example + range_per_example
    zeros = torch.zeros(
        (batch_size * max_nr_items_per_example,), device=flattened_data.device, dtype=flattened_data.dtype)
    return torch.scatter(
        zeros, dim=0, index=new_scatter_indices, src=flattened_data)\
        .view((batch_size, max_nr_items_per_example))
