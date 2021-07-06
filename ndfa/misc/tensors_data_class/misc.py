import torch
import typing
import hashlib
import functools
import dataclasses
from typing import List, Union, Optional, Tuple, Dict, Any
from typing_extensions import Protocol


__all__ = ['seq_lengths_to_mask', 'compose_fns', 'collate_tensors_with_variable_shapes',
           'CollateData', 'CollatableValuesTuple', 'MapFn', 'inverse_permutation',
           'OriginalTypeInfo', 'get_original_type', 'get_random_seed_per_example']


def seq_lengths_to_mask(seq_lengths: torch.LongTensor, max_seq_len: int, batch_first: bool = True):
    assert batch_first
    batch_size = seq_lengths.size(0)
    batched_ranges = torch.arange(start=1, end=max_seq_len + 1, dtype=torch.long, device=seq_lengths.device) \
        .unsqueeze(0).expand(batch_size, max_seq_len)
    mask = (batched_ranges <= seq_lengths.unsqueeze(-1).expand(batch_size, max_seq_len))
    return mask


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
    model_hps: Optional[Any] = None


CollatableValuesTuple = Union[
    Tuple[str, ...], Tuple[bool, ...], Tuple[Dict, ...], Tuple[List, ...], Tuple[Tuple, ...],
    Tuple['TensorsDataClass', ...], Tuple[torch.Tensor, ...]]


class MapFn(Protocol):
    def __call__(self, value: Optional[CollatableValuesTuple]) -> Optional[CollatableValuesTuple]: ...


def inverse_permutation(permutation: torch.LongTensor) -> torch.LongTensor:
    assert permutation.ndim == 1
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    return inverse


@dataclasses.dataclass
class OriginalTypeInfo:
    original_type: typing.Type
    unwrapped_type: typing.Optional[typing.Type] = None
    is_optional: bool = False


def get_original_type(_type: typing.Type) -> OriginalTypeInfo:
    origin_type = typing.get_origin(_type)
    if origin_type is None:
        return OriginalTypeInfo(original_type=_type)
    if origin_type == typing.Union or _type == typing.Union:
        union_types = typing.get_args(_type)
        assert len(union_types) == 2 and union_types[1] == type(None)
        original_type_info = get_original_type(union_types[0])
        original_type_info.is_optional = True
        return original_type_info
    assert typing.get_origin(origin_type) is None
    return OriginalTypeInfo(original_type=origin_type, unwrapped_type=_type)


def get_random_seed_per_example(
        batch_dependent_seed: bool, example_dependent_seed: bool,
        initial_seed_salt: str, collate_data: CollateData) -> List[int]:
    if batch_dependent_seed and example_dependent_seed:
        return [
            int(hashlib.sha256(f'{initial_seed_salt}|{"-".join(collate_data.example_hashes)}|{example_idx}'
                               .encode('ascii')).hexdigest(), 16) % (2 ** 32)
            for example_idx, _ in enumerate(collate_data.example_hashes)]
    elif not batch_dependent_seed and example_dependent_seed:
        return [
            int(hashlib.sha256(f'{initial_seed_salt}|{example_hash}'
                               .encode('ascii')).hexdigest(), 16) % (2 ** 32)
            for example_hash in collate_data.example_hashes]
    elif batch_dependent_seed and not example_dependent_seed:
        return [
            int(hashlib.sha256(f'{initial_seed_salt}|{"-".join(collate_data.example_hashes)}'
                               .encode('ascii')).hexdigest(), 16) % (2 ** 32)
            for _ in collate_data.example_hashes]
    else:
        return [
            int(hashlib.sha256(f'{initial_seed_salt}'
                               .encode('ascii')).hexdigest(), 16) % (2 ** 32)
            for _ in collate_data.example_hashes]
