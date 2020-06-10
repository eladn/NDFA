import torch
import dataclasses
from typing import List, Union, Iterable, Optional


__all__ = ['TensorsDataClass']


@dataclasses.dataclass
class TensorsDataClass:
    _is_batched: bool = dataclasses.field(init=False, default=False)
    _batch_size: Optional[int] = dataclasses.field(init=False, default=None)

    @property
    def is_batched(self) -> int:
        return self._is_batched

    @property
    def batch_size(self) -> int:
        if not self.is_batched:
            raise ValueError(
                f'called `batch_size()` on a not batched `{self.__class__.__name__}(TensorsDataClass)`. object id: {id(self)}')
        assert self._batch_size is not None
        return self._batch_size

    def to(self, device):
        return self.map(lambda field_val, _: field_val.to(device) if hasattr(field_val, 'to') else field_val)

    def cpu(self):
        return self.map(lambda field_val, _: field_val.cpu() if hasattr(field_val, 'cpu') else field_val)

    def numpy(self):
        return self.cpu().map(lambda field_val, _: field_val.numpy() if hasattr(field_val, 'numpy') else field_val)

    def map(self, map_fn):
        new_obj = self.__class__(**{
            field.name: map_fn(getattr(self, field.name), field.name)
            for field in dataclasses.fields(self) if field.init})
        for field in dataclasses.fields(self):
            if not field.init:
                if field.name in {'_is_batched', '_batch_size'}:
                    setattr(new_obj, field.name, getattr(self, field.name))
                else:
                    setattr(new_obj, field.name, map_fn(getattr(self, field.name), field.name))
        return new_obj

    @classmethod
    def collate(cls, code_task_inputs: List['TensorsDataClass']):
        assert all(not code_task_input.is_batched for code_task_input in code_task_inputs)
        assert len(code_task_inputs) > 0
        assert any(isinstance(getattr(code_task_inputs[0], field.name), torch.Tensor)
                   for field in dataclasses.fields(cls))

        # TODO: when collating tensors, pad it to match the longest seq in the batch, and add lengths vector.
        # TODO: remove this check! it is actually ok.
        for field in dataclasses.fields(cls):
            if not isinstance(getattr(code_task_inputs[0], field.name), torch.Tensor):
                continue
            if any(getattr(code_task_input, field.name).size() != getattr(code_task_inputs[0], field.name).size()
                   for code_task_input in code_task_inputs):
                raise ValueError(
                    f'Not all examples have the same tensor size for `{field.name}`. '
                    f'sizes: {[getattr(code_task_input, field.name).size() for code_task_input in code_task_inputs]}')

        def collate_field(field_name: str, values_iter: Iterable[Union[str, bool, torch.Tensor]]):
            assert field_name not in {'_is_batched', '_batch_size'}
            values_as_tuple = tuple(values_iter)
            assert all(type(value) == type(values_as_tuple[0]) for value in values_as_tuple)
            if isinstance(values_as_tuple[0], TensorsDataClass):
                assert hasattr(values_as_tuple[0].__class__, 'collate')
                return values_as_tuple[0].__class__.collate(values_as_tuple)
            if isinstance(values_as_tuple[0], torch.Tensor):
                return torch.cat(tuple(tensor.unsqueeze(0) for tensor in values_as_tuple), dim=0)
            return values_as_tuple

        assert all(field.init ^ (field.name in {'_is_batched', '_batch_size'}) for field in dataclasses.fields(cls))
        batched_obj = cls(
            **{field.name: collate_field(
                field.name, (getattr(code_task_input, field.name) for code_task_input in code_task_inputs))
                for field in dataclasses.fields(cls) if field.init and field.name not in {'_is_batched', '_batch_size'}})
        batched_obj._is_batched = True
        batched_obj._batch_size = len(code_task_inputs)
        return batched_obj
