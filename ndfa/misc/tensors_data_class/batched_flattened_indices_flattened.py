import torch
import dataclasses
from typing import List, Union, Optional, Tuple, Dict, Set, Any, final

from .misc import CollateData
from .tensors_data_class import TensorsDataClass
from .mixins import HasTargetIndexingGroupMixin, TensorDataClassWithSingleIndicesTensorMixin, \
    TensorDataClassWithSingleSequenceFieldMixin
from .batch_flattened import BatchFlattenedTensorsDataClassMixin
from .batch_flattened_seq import BatchFlattenedSequencesDataClassMixin


__all__ = [
    'BatchedFlattenedIndicesFlattenedTensorsDataClassMixin',
    'BatchedFlattenedIndicesFlattenedTensorsDataClass',
    'BatchedFlattenedIndicesFlattenedTensor',
    'BatchedFlattenedIndicesFlattenedSequencesDataClassMixin',
    'BatchedFlattenedIndicesFlattenedSequencesDataClass',
    'BatchedFlattenedIndicesFlattenedSeq']


@dataclasses.dataclass
class BatchedFlattenedIndicesFlattenedTensorsDataClassMixin(BatchFlattenedTensorsDataClassMixin, HasTargetIndexingGroupMixin):
    within_example_indexing_start: int = dataclasses.field(default=0)
    # collate auxiliaries
    example_index: Optional[torch.LongTensor] = dataclasses.field(init=False, default=None)

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        return super(BatchedFlattenedIndicesFlattenedTensorsDataClassMixin, cls).get_management_fields() + \
               ('example_index', 'within_example_indexing_start')

    @classmethod
    def get_indices_fields(cls):
        return super(BatchedFlattenedIndicesFlattenedTensorsDataClassMixin, cls).get_data_fields()

    @classmethod
    def _collate_first_pass(
            cls, inputs: List['BatchedFlattenedIndicesFlattenedTensorsDataClassMixin'],
            collate_data: CollateData) \
            -> 'BatchedFlattenedIndicesFlattenedTensorsDataClassMixin':
        assert all(inp.example_index is None for inp in inputs)
        assert all(inp.within_example_indexing_start == inputs[0].within_example_indexing_start for inp in inputs)
        flattened = super(BatchedFlattenedIndicesFlattenedTensorsDataClassMixin, cls)._collate_first_pass(
            inputs, collate_data=collate_data)
        indices_fields = cls.get_indices_fields()
        flattened.tgt_indexing_group = inputs[0].tgt_indexing_group
        flattened.within_example_indexing_start = inputs[0].within_example_indexing_start
        flattened.example_index = torch.cat([
            torch.full(size=(getattr(inp, indices_fields[0].name).size(0),),
                       fill_value=example_idx, dtype=torch.long)
            for example_idx, inp in enumerate(inputs)], dim=0)
        return flattened

    def post_collate_indices_fix(self, parents: Tuple['TensorsDataClass', ...], fields_path: Tuple[str, ...],
                                 collate_data: CollateData):
        if self.tgt_indexing_group is None:
            raise ValueError(f'`{self.__class__.__name__}` must have an `tgt_indexing_group`.')
        addressed_flattened_tensor = self.find_addressed_batched_flattened_tensor(parents[0])
        if addressed_flattened_tensor is None:
            raise ValueError(
                f'Not found field in tensors data class which is addressable '
                f'via index group `{self.tgt_indexing_group}`.')
        for field in self.get_indices_fields():
            original_indices = getattr(self, field.name)
            fixes = addressed_flattened_tensor.batched_index_offset_additive_fix_per_example[self.example_index]
            assert original_indices.size()[:fixes.ndim] == fixes.size()
            if original_indices.ndim > fixes.ndim:
                fixes = fixes.view(fixes.size() + (1,) * (original_indices.ndim - fixes.ndim)).expand(original_indices.size())
            offsets_fixes = torch.where(
                original_indices < self.within_example_indexing_start,
                torch.zeros(1, dtype=original_indices.dtype, device=original_indices.device),
                fixes)
            setattr(self, field.name, original_indices + offsets_fixes)

    def post_collate_remove_unnecessary_collate_info(self):
        self.example_index = None


@dataclasses.dataclass
class BatchedFlattenedIndicesFlattenedTensorsDataClass(BatchedFlattenedIndicesFlattenedTensorsDataClassMixin, TensorsDataClass):
    pass  # the double inheritance is all the impl needed


@final
@dataclasses.dataclass
class BatchedFlattenedIndicesFlattenedTensor(BatchedFlattenedIndicesFlattenedTensorsDataClass,
                                             TensorDataClassWithSingleIndicesTensorMixin):
    pass  # the double inheritance is all the impl needed


# TODO: check implementation!
@dataclasses.dataclass
class BatchedFlattenedIndicesFlattenedSequencesDataClassMixin(
        BatchFlattenedSequencesDataClassMixin,
        BatchedFlattenedIndicesFlattenedTensorsDataClassMixin):
    @classmethod
    def _collate_first_pass(
            cls, inputs: List['BatchedFlattenedIndicesFlattenedSequencesDataClassMixin'],
            collate_data: CollateData) \
            -> 'BatchedFlattenedIndicesFlattenedSequencesDataClassMixin':
        """
        Note: The calls order to `_collate_first_pass()` might be a bit confusing because of the inheritance order
        of the mixins. if we add print("<class name>") in the first line of the method `_collate_first_pass()`
        in all inheritors we get the following output:
            > BatchedFlattenedIndicesFlattenedSequencesDataClassMixin (this class) - calls super()
            > BatchFlattenedSequencesDataClassMixin (1st inheritor) - calls super()
            > BatchedFlattenedIndicesFlattenedTensorsDataClassMixin (2nd inheritor) - calls super()
            > BatchFlattenedTensorsDataClassMixin (common 2nd-degree inheritor of both 2 direct inheritors)
        """
        flattened = super(BatchedFlattenedIndicesFlattenedSequencesDataClassMixin, cls)._collate_first_pass(
            inputs, collate_data=collate_data)
        return flattened

    def post_collate_indices_fix(self, parents: Tuple['TensorsDataClass', ...], fields_path: Tuple[str, ...],
                                 collate_data: CollateData):
        super(BatchedFlattenedIndicesFlattenedSequencesDataClassMixin, self).post_collate_indices_fix(
            parents=parents, fields_path=fields_path, collate_data=collate_data)
        sequences_fields = self.get_data_fields()
        for field in sequences_fields:
            # Fill 0s (without example offset) for sequences paddings.
            original_sequence = getattr(self, field.name)
            masked_sequence = original_sequence.masked_fill(~self.sequences_mask, 0)
            setattr(self, field.name, masked_sequence)


@dataclasses.dataclass
class BatchedFlattenedIndicesFlattenedSequencesDataClass(
        BatchedFlattenedIndicesFlattenedSequencesDataClassMixin,
        TensorsDataClass):
    pass  # the double inheritance is all the impl needed


# TODO: check implementation!
@final
@dataclasses.dataclass
class BatchedFlattenedIndicesFlattenedSeq(BatchedFlattenedIndicesFlattenedSequencesDataClass,
                                          TensorDataClassWithSingleSequenceFieldMixin):
    pass  # the double inheritance is all the impl needed
