import torch
from typing import List, Tuple, Union


__all__ = ['weave_tensors', 'unweave_tensor']


def weave_tensors(
        tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]], dim: int = 0):
    assert len(tensors) > 0
    assert all(tensor.shape == tensors[0].shape for tensor in tensors)
    shape = tensors[0].shape
    new_shape = shape[:dim] + (len(tensors) * shape[dim],) + shape[dim + 1:]
    woven = torch.stack(tensors, dim=dim + 1).view(new_shape)
    # Alternative way to do it:
    # woven = torch.stack(tensors, dim=dim)\
    #     .permute(tuple(range(dim)) + (dim + 1, dim) + tuple(range(dim + 2, len(shape) + 1)))\
    #     .flatten(dim, dim + 1)
    # assert woven.shape == new_shape
    return woven


def unweave_tensor(
        woven_tensor: torch.Tensor, dim: int = 0,
        nr_target_tensors: int = 2) -> Tuple[torch.Tensor]:
    woven_shape = woven_tensor.shape
    unweaving_shape = \
        woven_shape[:dim] + \
        (woven_shape[dim] // nr_target_tensors, nr_target_tensors) + \
        woven_shape[dim + 1:]
    unwoven = tuple(torch.chunk(woven_tensor.view(unweaving_shape), chunks=nr_target_tensors, dim=dim + 1))
    unwoven = tuple(tensor.squeeze(dim + 1) for tensor in unwoven)
    new_tensor_shape = woven_shape[:dim] + (woven_shape[dim] // nr_target_tensors,) + woven_shape[dim + 1:]
    assert all(tensor.shape == new_tensor_shape for tensor in unwoven)
    return unwoven
