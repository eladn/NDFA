import torch
import dataclasses
from typing import List, final

from .misc import CollateData, collate_tensors_with_variable_shapes
from .tensors_data_class import TensorsDataClass
from .mixins import TensorDataClassWithSingleDataTensorMixin


__all__ = ['TensorWithCollateMask']


@final
@dataclasses.dataclass
class TensorWithCollateMask(TensorDataClassWithSingleDataTensorMixin, TensorsDataClass):
    collate_mask: torch.BoolTensor = dataclasses.field(default=None, init=False)

    @classmethod
    def _collate_first_pass(cls, inputs: List['TensorWithCollateMask'], collate_data: CollateData):
        assert all(type(inp) is TensorWithCollateMask for inp in inputs)
        assert all(not inp.is_batched for inp in inputs)
        assert all(inp.collate_mask is None for inp in inputs)
        tensors = tuple(inp.tensor for inp in inputs)
        collated_tensor, collate_mask = collate_tensors_with_variable_shapes(
            tensors=tensors, create_collate_mask=True, create_collate_lengths=False)
        batched_obj = cls(tensor=collated_tensor)
        batched_obj.collate_mask = collate_mask
        batched_obj._batch_size = len(inputs)
        return batched_obj
