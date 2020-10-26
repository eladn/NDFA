import torch
import functools
import dataclasses
from typing import List, Union, Optional, Tuple, Dict, Set, Any
from typing_extensions import Protocol


from ndfa.nn_utils.misc import seq_lengths_to_mask


__all__ = ['seq_lengths_to_mask', 'compose_fns', 'collate_tensors_with_variable_shapes',
           'CollateData', 'CollatableValuesTuple', 'MapFn']


def compose_fns(*functions):
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def collate_tensors_with_variable_shapes(
        tensors: Tuple[torch.Tensor], create_collate_mask: bool = True,
        create_collate_lengths: bool = False, last_variable_dim: int = -1,
        padding_fill_value: Optional[Any] = None) \
        -> Union[torch.Tensor,
                 Tuple[torch.Tensor, torch.BoolTensor],
                 Tuple[torch.Tensor, torch.LongTensor],
                 Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]]:
    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors)
    nr_dims = tensors[0].ndim
    assert all(tensor.ndim == nr_dims for tensor in tensors)
    last_variable_dim = last_variable_dim % nr_dims
    for non_variable_dim in range(last_variable_dim + 1, nr_dims):
        if any(tensor.size(non_variable_dim) != tensors[0].size(non_variable_dim) for tensor in tensors):
            raise ValueError(
                f'Dimension #{non_variable_dim} is set to be non-variable '
                f'(last variable dim is #{last_variable_dim}), '
                f'but input tensors have different sizes for this dim.')

    max_dims = tuple(
        max(tensor.size(dim_idx) for tensor in tensors)
        for dim_idx in range(tensors[0].ndim))
    collate_size = (len(tensors),) + max_dims
    if create_collate_mask:
        collate_mask_size = (len(tensors),) + max_dims[:last_variable_dim + 1]
        collate_mask = torch.zeros(collate_mask_size, dtype=torch.bool, device=tensors[0].device)
    if create_collate_lengths:
        collate_lengths = torch.zeros((len(tensors), last_variable_dim + 1),
                                      dtype=torch.long, device=tensors[0].device)
    if padding_fill_value is None:
        collated_tensor = torch.zeros(collate_size, dtype=tensors[0].dtype, device=tensors[0].device)
    else:
        collated_tensor = torch.full(collate_size, fill_value=padding_fill_value,
                                     dtype=tensors[0].dtype, device=tensors[0].device)
    for idx, tensor in enumerate(tensors):
        if tensor.ndim == 0:
            collated_tensor[idx] = tensor
        if tensor.ndim == 1:
            collated_tensor[idx, :tensor.size(0)] = tensor
        elif tensor.ndim == 2:
            collated_tensor[idx, :tensor.size(0), :tensor.size(1)] = tensor
        elif tensor.ndim == 3:
            collated_tensor[idx, :tensor.size(0), :tensor.size(1), :tensor.size(2)] = tensor
        elif tensor.ndim == 4:
            collated_tensor[idx, :tensor.size(0), :tensor.size(1), :tensor.size(2), :tensor.size(3)] = tensor
        elif tensor.ndim == 5:
            collated_tensor[
                idx, :tensor.size(0), :tensor.size(1), :tensor.size(2), :tensor.size(3), :tensor.size(4)] = tensor
        else:
            raise ValueError(
                f'Cannot collate tensor with > 5 dims. Given input tensors with {tensor.ndim} dims.')
        if create_collate_mask:
            if last_variable_dim == 0:
                collate_mask[idx, :tensor.size(0)] = True
            elif last_variable_dim == 1:
                collate_mask[idx, :tensor.size(0), :tensor.size(1)] = True
            elif last_variable_dim == 2:
                collate_mask[idx, :tensor.size(0), :tensor.size(1), :tensor.size(2)] = True
            elif last_variable_dim == 3:
                collate_mask[idx, :tensor.size(0), :tensor.size(1), :tensor.size(2), :tensor.size(3)] = True
            elif last_variable_dim == 4:
                collate_mask[
                    idx, :tensor.size(0), :tensor.size(1), :tensor.size(2), :tensor.size(3), :tensor.size(4)] = True
            else:
                raise ValueError(
                    f'Cannot collate tensor with > 5 dims. Given input tensors with {tensor.ndim} dims.')
        if create_collate_lengths:
            collate_lengths[idx] = torch.tensor(tensor.size()[:last_variable_dim + 1],
                                                dtype=torch.long, device=tensors[0].device)
    ret = collated_tensor,
    if create_collate_mask:
        ret += collate_mask,
    if create_collate_lengths:
        ret += collate_lengths,
    return ret[0] if len(ret) == 1 else ret


@dataclasses.dataclass
class CollateData:
    example_hashes: Optional[List[str]] = None


CollatableValuesTuple = Union[
    Tuple[str, ...], Tuple[bool, ...], Tuple[Dict, ...], Tuple[List, ...], Tuple[Tuple, ...],
    Tuple['TensorsDataClass', ...], Tuple[torch.Tensor, ...]]


class MapFn(Protocol):
    def __call__(self, value: Optional[CollatableValuesTuple]) -> Optional[CollatableValuesTuple]: ...
