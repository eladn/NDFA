import torch
import dataclasses
from typing import List, Union, Optional, Tuple, Dict, Set, Any, final

from .misc import CollateData
from .tensors_data_class import TensorsDataClass
from .mixins import HasTargetIndexingGroupMixin, TensorDataClassWithSingleIndicesTensorMixin, HasSelfIndexingGroupMixin


__all__ = ['BatchedFlattenedIndicesTensor', 'batch_flattened_indices_tensor_field']


@final
@dataclasses.dataclass
class BatchedFlattenedIndicesTensor(
        HasTargetIndexingGroupMixin,
        TensorDataClassWithSingleIndicesTensorMixin,
        TensorsDataClass):
    within_example_indexing_start: int = dataclasses.field(default=0)

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        return super(BatchedFlattenedIndicesTensor, cls).get_management_fields() + \
               ('within_example_indexing_start', )

    @classmethod
    def get_indices_fields(cls):
        return super(BatchedFlattenedIndicesTensor, cls).get_data_fields()

    @classmethod
    def _collate_first_pass(
            cls, inputs: List['BatchedFlattenedIndicesTensor'],
            collate_data: CollateData) \
            -> 'BatchedFlattenedIndicesTensor':
        assert all(inp.within_example_indexing_start == inputs[0].within_example_indexing_start for inp in inputs)
        collated = super(BatchedFlattenedIndicesTensor, cls)._collate_first_pass(
            inputs, collate_data=collate_data)
        collated.tgt_indexing_group = inputs[0].tgt_indexing_group
        collated.within_example_indexing_start = inputs[0].within_example_indexing_start
        return collated

    def post_collate_indices_fix(
            self, parents: Tuple['TensorsDataClass', ...],
            fields_path: Tuple[str, ...],
            collate_data: CollateData,
            batched_flattened_tensors_with_self_indexing_group: Dict[str, HasSelfIndexingGroupMixin]):
        if self.tgt_indexing_group is None:
            raise ValueError(f'`{self.__class__.__name__}` must have an `tgt_indexing_group`.')
        addressed_flattened_tensor = \
            batched_flattened_tensors_with_self_indexing_group.get(self.tgt_indexing_group, None)
        # addressed_flattened_tensor = self.find_addressed_batched_flattened_tensor(parents[0])  # old expensive impl
        if addressed_flattened_tensor is None:
            raise ValueError(
                f'Not found field in tensors data class which is addressable '
                f'via index group `{self.tgt_indexing_group}`.')
        # assert isinstance(addressed_flattened_tensor, BatchFlattenedTensorsDataClassMixin)
        for field in self.get_indices_fields():
            original_indices = getattr(self, field.name)
            assert addressed_flattened_tensor.batched_index_offset_additive_fix_per_example.size(0) == \
                   addressed_flattened_tensor.nr_examples
            assert addressed_flattened_tensor.nr_examples == self.nr_examples
            assert self.nr_examples == original_indices.size(0)
            offsets_fixes = torch.where(
                original_indices < self.within_example_indexing_start,
                torch.zeros((1, ), dtype=original_indices.dtype, device=original_indices.device),
                addressed_flattened_tensor.batched_index_offset_additive_fix_per_example.unsqueeze(-1).expand(original_indices.size()))
            setattr(self, field.name, original_indices + offsets_fixes)


def batch_flattened_indices_tensor_field(
        *,
        default=dataclasses.MISSING,
        tgt_indexing_group: Optional[str] = dataclasses.MISSING) -> dataclasses.Field:
    management_fields_defaults = {'tgt_indexing_group': tgt_indexing_group}
    management_fields_defaults = {k: v for k, v in management_fields_defaults.items() if v is not dataclasses.MISSING}
    return dataclasses.field(default=default, metadata=management_fields_defaults)
