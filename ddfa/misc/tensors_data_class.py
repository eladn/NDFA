import torch
import dataclasses
from typing import List, Union, Iterable


__all__ = ['TensorsDataClass']


@dataclasses.dataclass
class TensorsDataClass:
    @property
    def is_batched(self) -> int:
        return hasattr(self, '_TensorsDataClass_is_batched') and getattr(self, '_TensorsDataClass_is_batched')

    @property
    def batch_size(self) -> int:
        if not self.is_batched:
            raise ValueError('called `batch_size()` on a not batched `TensorsDataClass`.')
        assert hasattr(self, '_TensorsDataClass_batch_size')
        return getattr(self, '_TensorsDataClass_batch_size')

    def to(self, device):
        return self.__class__.__new__(**{
            field.name:
                getattr(self, field.name).to(device)
                if hasattr(getattr(self, field.name), 'to') else
                getattr(self, field.name)
            for field in dataclasses.fields(self)})

    def cpu(self):
        return self.__class__.__new__(**{
            field.name:
                getattr(self, field.name).cpu()
                if hasattr(getattr(self, field.name), 'cpu') else
                getattr(self, field.name)
            for field in dataclasses.fields(self)})

    def numpy(self):
        cpu = self.cpu()
        return self.__class__.__new__(**{
            field.name:
                getattr(cpu, field.name).numpy()
                if hasattr(getattr(cpu, field.name), 'numpy') else
                getattr(cpu, field.name)
            for field in dataclasses.fields(cpu)})

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
            # if field_name == 'is_batched':
            #     return True
            values_as_tuple = tuple(values_iter)
            assert all(type(value) == type(values_as_tuple[0]) for value in values_as_tuple)
            if isinstance(values_as_tuple[0], TensorsDataClass):
                assert hasattr(values_as_tuple[0].__class__, 'collate')
                return values_as_tuple[0].__class__.collate(values_as_tuple)
            if isinstance(values_as_tuple[0], torch.Tensor):
                return torch.cat(tuple(tensor.unsqueeze(0) for tensor in values_as_tuple), dim=0)
            return values_as_tuple

        batched_obj = cls.__new__(
            **{field.name: collate_field(
                field.name, (getattr(code_task_input, field.name) for code_task_input in code_task_inputs))
                for field in dataclasses.fields(cls)})
        setattr(batched_obj, '_TensorsDataClass_is_batched', True)
        setattr(batched_obj, '_TensorsDataClass_batch_size', len(code_task_inputs))
        return batched_obj
