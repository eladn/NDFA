import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple


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
