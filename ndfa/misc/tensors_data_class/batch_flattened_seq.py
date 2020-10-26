import torch
import dataclasses
from typing import List, Union, Optional, Tuple, Dict, Set, Any, final

from .misc import seq_lengths_to_mask, collate_tensors_with_variable_shapes, CollateData
from .mixins import TensorDataClassWithSequencesMixin, TensorDataClassWithSingleSequenceFieldMixin
from .batch_flattened import BatchFlattenedTensorsDataClassMixin
from .tensors_data_class import TensorsDataClass


__all__ = ['BatchFlattenedSequencesDataClassMixin', 'BatchFlattenedSequencesDataClass', 'BatchFlattenedSeq']


@dataclasses.dataclass
class BatchFlattenedSequencesDataClassMixin(BatchFlattenedTensorsDataClassMixin, TensorDataClassWithSequencesMixin):
    @classmethod
    def _collate_first_pass(cls, inputs: List['BatchFlattenedSequencesDataClassMixin'],
                            collate_data: CollateData) -> 'BatchFlattenedSequencesDataClassMixin':
        assert all(inp.batch_first == inputs[0].batch_first for inp in inputs)
        if not inputs[0].batch_first:
            raise NotImplementedError('`batch_first` option is not implemented yet for `BatchFlattenedSeq`.')
        assert all(inp.sequences_lengths is None for inp in inputs)
        assert all(inp.sequences_mask is None for inp in inputs)

        sequences_fields = cls.get_data_fields()
        assert all(
            (all(isinstance(getattr(inp, field.name), list) for inp in inputs) and
             all(isinstance(seq, torch.Tensor) for inp in inputs for seq in getattr(inp, field.name))) or
            all(isinstance(getattr(inp, field.name), torch.Tensor) for inp in inputs)
            for field in sequences_fields)
        assert all(seq.ndim >= 1 for inp in inputs for field in sequences_fields for seq in getattr(inp, field.name))
        assert all(len(getattr(inp, field.name)) == len(getattr(inp, sequences_fields[0].name))
                   for field in sequences_fields for inp in inputs)
        assert all(seq1.size(0) == seq2.size(0)
                   for field in sequences_fields for inp in inputs
                   for seq1, seq2 in zip(getattr(inp, sequences_fields[0].name), getattr(inp, field.name)))
        seq_lengths = [seq.size(0) for inp in inputs for seq in getattr(inp, sequences_fields[0].name)]

        if any(isinstance(getattr(inputs[0], field.name), list) for field in sequences_fields):
            fixed_inputs = [
                cls(**{
                    field.name:
                        collate_tensors_with_variable_shapes(
                            tensors=tuple(getattr(inp, field.name)), create_collate_mask=False,
                            create_collate_lengths=False, last_variable_dim=0)
                        if isinstance(getattr(inputs[0], field.name), list) else
                        getattr(inp, field.name)
                    for field in sequences_fields})
                for inp in inputs]
            for inp, fixed_inp in zip(inputs, fixed_inputs):
                for fld in cls.get_management_fields():
                    setattr(fixed_inp, fld, getattr(inp, fld))
            inputs = fixed_inputs
        flattened = super(BatchFlattenedSequencesDataClassMixin, cls)._collate_first_pass(
            inputs, collate_data=collate_data)
        flattened.sequences_lengths = torch.LongTensor(seq_lengths, device=flattened.sequences.device)
        flattened.max_sequence_length = max(seq_lengths)
        flattened.sequences_mask = seq_lengths_to_mask(
            seq_lengths=flattened.sequences_lengths, max_seq_len=flattened.max_sequence_length)
        return flattened

    @classmethod
    def _flatten_tensors(cls, tensors: List[torch.Tensor]):
        max_seq_len = max(tensor.size(1) for tensor in tensors)
        padded_tensors = [
            torch.zeros((tensor.size(0), max_seq_len) + tensor.size()[2:],
                        dtype=tensor.dtype, device=tensor.device)
            for tensor in tensors]
        for tensor, padded_tensor in zip(tensors, padded_tensors):
            padded_tensor[:, :tensor.size(1)] = tensor
        return torch.cat(padded_tensors, dim=0)


@dataclasses.dataclass
class BatchFlattenedSequencesDataClass(BatchFlattenedSequencesDataClassMixin, TensorsDataClass):
    pass  # the double inheritance is all the impl needed


@final
@dataclasses.dataclass
class BatchFlattenedSeq(BatchFlattenedSequencesDataClass, TensorDataClassWithSingleSequenceFieldMixin):
    pass  # the double inheritance is all the impl needed
