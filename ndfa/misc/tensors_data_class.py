import torch
import dataclasses
from typing import List, Union, Optional, Tuple, final


__all__ = ['TensorsDataClass', 'TensorWithCollateMask']


@dataclasses.dataclass
class TensorsDataClass:
    _batch_size: Optional[int] = dataclasses.field(init=False, default=None)

    @property
    def is_batched(self) -> int:
        return self._batch_size is not None

    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            raise ValueError(
                f'called `batch_size()` on a not batched `{self.__class__.__name__}(TensorsDataClass)`. object id: {id(self)}')
        return self._batch_size

    def to(self, device):
        return self.map(lambda field_val, _: field_val.to(device) if hasattr(field_val, 'to') else field_val)

    def cpu(self):
        return self.map(lambda field_val, _: field_val.cpu() if hasattr(field_val, 'cpu') else field_val)

    def numpy(self):
        return self.cpu().map(lambda field_val, _: field_val.numpy() if hasattr(field_val, 'numpy') else field_val)

    def tolist(self):
        return self.cpu().map(lambda field_val, _: field_val.tolist() if hasattr(field_val, 'tolist') else field_val)

    def map(self, map_fn):
        new_obj = self.__class__(**{
            field.name: map_fn(getattr(self, field.name), field.name)
            for field in dataclasses.fields(self) if field.init})
        for field in dataclasses.fields(self):
            if not field.init:
                if field.name in {'_batch_size'}:
                    setattr(new_obj, field.name, getattr(self, field.name))
                else:
                    setattr(new_obj, field.name, map_fn(getattr(self, field.name), field.name))
        return new_obj

    @classmethod
    def collate_field(
            cls, field_name: str,
            values_as_tuple: Tuple[Union[str, bool, dict, list, tuple, 'TensorsDataClass', torch.Tensor]]):
        assert field_name not in {'_batch_size'}
        assert all(type(value) == type(values_as_tuple[0]) for value in values_as_tuple)
        if values_as_tuple[0] is None:
            return None
        if isinstance(values_as_tuple[0], TensorsDataClass):
            assert hasattr(values_as_tuple[0].__class__, 'collate')
            return values_as_tuple[0].__class__.collate(values_as_tuple)
        if isinstance(values_as_tuple[0], dict):
            all_keys = {key for dct in values_as_tuple for key in dct.keys()}
            return {key: cls.collate_field(field_name, (dct[key] for dct in values_as_tuple if key in dct))
                    for key in all_keys}
        if isinstance(values_as_tuple[0], list) or isinstance(values_as_tuple[0], tuple):
            collection_type = type(values_as_tuple[0])
            assert all(len(lst) == len(values_as_tuple[0]) for lst in values_as_tuple)
            return collection_type(cls.collate_field(field_name, vals) for vals in zip(*values_as_tuple))
        if isinstance(values_as_tuple[0], torch.Tensor):
            if any(tensor.size() != values_as_tuple[0].size() for tensor in values_as_tuple):
                raise ValueError(
                    f'Not all examples have the same tensor size for `{field_name}`. '
                    f'sizes: {[tensor.size() for tensor in values_as_tuple]}')
            return torch.cat(tuple(tensor.unsqueeze(0) for tensor in values_as_tuple), dim=0)
        return values_as_tuple

    @classmethod
    def collate(cls, inputs: List['TensorsDataClass']):
        assert all(isinstance(inp, cls) for inp in inputs)
        assert all(not inp.is_batched for inp in inputs)
        assert len(inputs) > 0
        assert any(isinstance(getattr(inputs[0], field.name), torch.Tensor)
                   for field in dataclasses.fields(cls))

        # TODO: when collating tensors, pad it to match the longest seq in the batch, and add lengths vector.
        # TODO: remove this check! it is actually ok to have different lengths.
        for field in dataclasses.fields(cls):
            if not isinstance(getattr(inputs[0], field.name), torch.Tensor):
                continue
            if any(getattr(inp, field.name).size() != getattr(inputs[0], field.name).size()
                   for inp in inputs):
                raise ValueError(
                    f'Not all examples have the same tensor size for `{field.name}`. '
                    f'sizes: {[getattr(inp, field.name).size() for inp in inputs]}')

        assert all(field.init ^ (field.name in {'_batch_size'}) for field in dataclasses.fields(cls))
        batched_obj = cls(
            **{field.name: cls.collate_field(
                field.name, tuple(getattr(inp, field.name) for inp in inputs))
                for field in dataclasses.fields(cls)
                if field.init and field.name not in {'_batch_size'}})
        batched_obj._batch_size = len(inputs)
        return batched_obj


@final
@dataclasses.dataclass
class TensorWithCollateMask(TensorsDataClass):
    tensor: torch.Tensor
    collate_mask: torch.BoolTensor = dataclasses.field(default=None)

    @classmethod
    def collate(cls, inputs: List['TensorWithCollateMask']):
        assert all(type(inp) == TensorWithCollateMask for inp in inputs)
        assert all(isinstance(inp.tensor, torch.Tensor) for inp in inputs)
        assert all(len(inp.tensor.size()) == len(inputs[0].tensor.size()) for inp in inputs)
        assert all(inp.collate_mask is None for inp in inputs)

        tensors = tuple(getattr(inp, 'tensor') for inp in inputs)
        max_dims = tuple(max(tensor.size()[dim_idx] for tensor in tensors)
                         for dim_idx in range(len(tensors[0].size())))
        collate_size = (len(tensors),) + max_dims
        collate_mask = torch.zeros(collate_size, dtype=torch.bool)
        collated_tensor = torch.zeros(collate_size, dtype=tensors[0].dtype)
        for idx, tensor in enumerate(tensors):
            if len(tensor.size()) == 0:
                collated_tensor[idx] = tensor
                collate_mask[idx] = True
            if len(tensor.size()) == 1:
                collated_tensor[idx, :tensor.size()[0]] = tensor
                collate_mask[idx, :tensor.size()[0]] = True
            elif len(tensor.size()) == 2:
                collated_tensor[idx, :tensor.size()[0], :tensor.size()[1]] = tensor
                collate_mask[idx, :tensor.size()[0], :tensor.size()[1]] = True
            elif len(tensor.size()) == 3:
                collated_tensor[idx, :tensor.size()[0], :tensor.size()[1], :tensor.size()[2]] = tensor
                collate_mask[idx, :tensor.size()[0], :tensor.size()[1], :tensor.size()[2]] = True
            elif len(tensor.size()) == 4:
                collated_tensor[idx, :tensor.size()[0], :tensor.size()[1], :tensor.size()[2], :tensor.size()[3]] = tensor
                collate_mask[idx, :tensor.size()[0], :tensor.size()[1], :tensor.size()[2], :tensor.size()[3]] = True
            else:
                raise ValueError(f'Cannot collate tensor with >= 4 dims. '
                                 f'Given input tensors with {len(tensor.size())} dims.')

        batched_obj = TensorWithCollateMask(tensor=collated_tensor, collate_mask=collate_mask)
        batched_obj._batch_size = len(inputs)
        return batched_obj


# TODO: complete impl
@dataclasses.dataclass
class ExampleBasedIndicesTensor(TensorsDataClass):
    indices: torch.LongTensor
    example_idx: torch.LongTensor = dataclasses.field(init=False, default=None)  # for accessing a `BatchFlattenedTensorsDataClass`

    @classmethod
    def collate(cls, inputs: List['ExampleBasedIndicesTensor']):
        raise NotImplementedError  # TODO: impl!

    def access(self, dst_tensor: torch.Tensor, example_offsets: torch.Tensor, mask=None):
        raise NotImplementedError  # TODO: impl!


# TODO: complete impl
@dataclasses.dataclass
class BatchFlattenedTensorsDataClass(TensorsDataClass):
    nr_items_per_example: torch.LongTensor = dataclasses.field(init=False, default=None)
    _example_offsets: torch.LongTensor = dataclasses.field(init=False, default=None)

    # for being accessed by `ExampleBasedIndexTensor`; exclusive cumsum of nr_tensors_per_example
    @property
    def example_offsets(self):
        if self._example_offsets is None:
            self._example_offsets = self.nr_items_per_example.cumsum(dim=-1) - self.nr_items_per_example
        return self._example_offsets

    @classmethod
    def collate(cls, inputs: List['BatchFlattenedTensorsDataClass']):
        raise NotImplementedError  # TODO: impl!


# TODO: complete impl
@dataclasses.dataclass
class BatchFlattenedTensor(BatchFlattenedTensorsDataClass):
    tensor: torch.Tensor


# TODO: complete impl
@dataclasses.dataclass
class BatchFlattenedSeq(BatchFlattenedTensor):
    lengths: torch.LongTensor

    @classmethod
    def collate(cls, inputs: List['BatchFlattenedSeq']):
        raise NotImplementedError  # TODO: impl!


# TODO: complete impl
@dataclasses.dataclass
class ExampleBasedIndicesBatchFlattenedTensorsDataClass(BatchFlattenedTensorsDataClass):
    example_idx: torch.LongTensor = dataclasses.field(init=False, default=None)  # for accessing a `BatchFlattenedTensorsDataClass`

    @classmethod
    def collate(cls, inputs: List['ExampleBasedIndicesBatchFlattenedTensorsDataClass']):
        raise NotImplementedError  # TODO: impl!


# TODO: complete impl
@dataclasses.dataclass
class ExampleBasedIndicesBatchFlattenedTensor(ExampleBasedIndicesBatchFlattenedTensorsDataClass):
    indices: torch.LongTensor
