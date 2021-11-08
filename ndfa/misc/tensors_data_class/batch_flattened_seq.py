import torch
import dataclasses
import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Set, Any, final, Callable

from .misc import seq_lengths_to_mask, collate_tensors_with_variable_shapes, CollateData, \
    get_random_seed_per_example
from .mixins import TensorDataClassWithSequencesMixin, TensorDataClassWithSingleSequenceFieldMixin
from .batch_flattened import BatchFlattenedTensorsDataClassMixin
from .tensors_data_class import TensorsDataClass


__all__ = ['BatchFlattenedSequencesDataClassMixin', 'BatchFlattenedSequencesDataClass', 'BatchFlattenedSeq',
           'batch_flattened_seq_field']


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

        if inputs[0].nr_sequences_to_sample_per_example is not None:
            random_seed_per_example = get_random_seed_per_example(
                batch_dependent_seed=True,
                example_dependent_seed=True,
                initial_seed_salt=inputs[0].sequences_sampling_initial_seed_salt,
                collate_data=collate_data)
            fixed_inputs_dicts = []
            for example_idx, inp in enumerate(inputs):
                fixed_input_dict = {}
                fixed_inputs_dicts.append(fixed_input_dict)
                for sequences_field in sequences_fields:
                    random_state = np.random.RandomState(random_seed_per_example[example_idx])
                    tensors = getattr(inp, sequences_field.name)
                    nr_tensors = tensors.size(0) if isinstance(tensors, torch.Tensor) else len(tensors)
                    if nr_tensors > inputs[0].nr_sequences_to_sample_per_example:
                        sampled_items_indices = random_state.choice(
                            nr_tensors, size=inputs[0].nr_sequences_to_sample_per_example, replace=False)
                        if isinstance(tensors, torch.Tensor):
                            sampled_items = torch.index_select(
                                tensors, 0, torch.LongTensor(sampled_items_indices))  # TODO: check!
                            assert sampled_items.shape[1:] == tensors.shape[1:]
                            assert sampled_items.size(0) == inputs[0].nr_sequences_to_sample_per_example
                        else:
                            assert isinstance(tensors, (list, tuple))
                            sampled_items = [tensors[index] for index in sampled_items_indices]
                            assert len(sampled_items) == inputs[0].nr_sequences_to_sample_per_example
                        fixed_input_dict[sequences_field.name] = sampled_items
            inputs = [
                dataclasses.replace(orig_inp, **fixed_inp_dict)
                for orig_inp, fixed_inp_dict in zip(inputs, fixed_inputs_dicts)]

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


def batch_flattened_seq_field(
        *,
        default=dataclasses.MISSING,
        self_indexing_group: Optional[str] = dataclasses.MISSING,
        sequences_sampling_initial_seed_salt: Optional[str] = dataclasses.MISSING,
        nr_sequences_to_sample_per_example: Optional[Union[int, Callable[[Any], int]]] = dataclasses.MISSING) \
        -> dataclasses.Field:
    management_fields_defaults = {
        'self_indexing_group': self_indexing_group,
        'sequences_sampling_initial_seed_salt': sequences_sampling_initial_seed_salt,
        'nr_sequences_to_sample_per_example': nr_sequences_to_sample_per_example}
    management_fields_defaults = {k: v for k, v in management_fields_defaults.items() if v is not dataclasses.MISSING}
    return dataclasses.field(default=default, metadata=management_fields_defaults)
