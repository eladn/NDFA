import torch
import dataclasses
from typing import List, Union, Optional, Tuple, Dict, Any, final


__all__ = [
    'TensorsDataClass', 'TensorWithCollateMask',
    'BatchFlattenedTensorsDataClass', 'BatchFlattenedTensor', 'BatchFlattenedSeq',
    'BatchedFlattenedIndicesFlattenedTensorsDataClass',
    'BatchedFlattenedIndicesFlattenedTensor', 'BatchedFlattenedIndicesFlattenedSeq']


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
class HasSelfIndexingGroup:
    self_indexing_group: str = None

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        supers = super(HasSelfIndexingGroup, cls).get_management_fields() \
            if hasattr(super(HasSelfIndexingGroup, cls), 'get_management_fields') else ()
        return supers + ('self_indexing_group',)


@dataclasses.dataclass
class HasTargetIndexingGroup:
    tgt_indexing_group: str = None

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        supers = super(HasTargetIndexingGroup, cls).get_management_fields() \
            if hasattr(super(HasTargetIndexingGroup, cls), 'get_management_fields') else ()
        return supers + ('tgt_indexing_group',)


CollatableValuesTuple = Union[
    Tuple[str, ...], Tuple[bool, ...], Tuple[Dict, ...], Tuple[List, ...], Tuple[Tuple, ...],
    Tuple['TensorsDataClass', ...], Tuple[torch.Tensor, ...]]


@dataclasses.dataclass
class TensorsDataClass:
    _batch_size: Optional[int] = dataclasses.field(init=False, default=None)

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        supers = super(TensorsDataClass, cls).get_management_fields() \
            if hasattr(super(TensorsDataClass, cls), 'get_management_fields') else ()
        return supers + ('_batch_size', )

    @classmethod
    def get_data_fields(cls) -> Tuple[dataclasses.Field, ...]:
        return tuple(field for field in dataclasses.fields(cls) if field.name not in set(cls.get_management_fields()))

    @property
    def is_batched(self) -> int:
        return self._batch_size is not None

    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            raise ValueError(
                f'called `batch_size()` on a not batched `{self.__class__.__name__}(TensorsDataClass)`. object id: {id(self)}')
        return self._batch_size

    @property
    def nr_examples(self):
        return self.batch_size  # for flattened batches it should be overridden

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
    def collate_values(cls, values_as_tuple: CollatableValuesTuple):
        assert all(type(value) == type(values_as_tuple[0]) for value in values_as_tuple)
        if values_as_tuple[0] is None:
            return None
        if isinstance(values_as_tuple[0], TensorsDataClass):
            assert hasattr(values_as_tuple[0].__class__, 'collate')
            return values_as_tuple[0].__class__.collate(values_as_tuple, is_most_outer_call=False)
        if isinstance(values_as_tuple[0], dict):
            all_keys = {key for dct in values_as_tuple for key in dct.keys()}
            return {key: cls.collate_values((dct[key] for dct in values_as_tuple if key in dct))
                    for key in all_keys}
        if isinstance(values_as_tuple[0], list) or isinstance(values_as_tuple[0], tuple):
            collection_type = type(values_as_tuple[0])
            assert all(len(lst) == len(values_as_tuple[0]) for lst in values_as_tuple)
            return collection_type(cls.collate_values(vals) for vals in zip(*values_as_tuple))
        if isinstance(values_as_tuple[0], torch.Tensor):
            return cls.collate_tensors(values_as_tuple)
        return values_as_tuple

    @classmethod
    def collate_tensors(cls, tensors: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if any(tensor.size() != tensors[0].size() for tensor in tensors):
            raise ValueError(
                f'Not all input tensors have the same tensor size. '
                f'sizes: {[tensor.size() for tensor in tensors]}')
        return torch.cat(tuple(tensor.unsqueeze(0) for tensor in tensors), dim=0)

    @classmethod
    def collate(cls, inputs: List['TensorsDataClass'], is_most_outer_call: bool = True):
        assert all(isinstance(inp, cls) for inp in inputs)
        assert all(not inp.is_batched for inp in inputs)
        assert len(inputs) > 0
        batched_obj = cls._collate_first_pass(inputs)
        if batched_obj._batch_size is None:
            batched_obj._batch_size = len(inputs)
        if is_most_outer_call:
            batched_obj.post_collate_indices_fix((), ())
            batched_obj.post_collate_remove_unnecessary_collate_info()
        return batched_obj

    @classmethod
    def _collate_first_pass(cls, inputs: List['TensorsDataClass']):
        batched_obj = cls(
            **{field.name: cls.collate_values(
                tuple(getattr(inp, field.name) for inp in inputs))
                for field in cls.get_data_fields() if field.init})
        batched_obj._batch_size = len(inputs)
        return batched_obj

    def post_collate_indices_fix(self, parents: Tuple['TensorsDataClass', ...], fields_path: Tuple[str, ...]):
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, TensorsDataClass):
                field_value.post_collate_indices_fix(
                    parents=parents + (self,), fields_path=fields_path + (field.name,))

    def post_collate_remove_unnecessary_collate_info(self):
        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, TensorsDataClass):
                field_value.post_collate_remove_unnecessary_collate_info()

    def traverse(self,
                 parents_path: Optional[Tuple['TensorsDataClass', ...]] = None,
                 fields_path: Optional[Tuple[str, ...]] = None,
                 traversal_order: str = 'previsit'):
        assert traversal_order in ('previsit', 'postvisit')
        assert not ((parents_path is None) ^ (fields_path is None))
        if traversal_order == 'previsit':
            yield self if parents_path is None else (self, parents_path, fields_path)
        for field in self.get_data_fields():
            field_value = getattr(self, field.name)
            if not isinstance(field_value, TensorsDataClass):
                continue
            new_parents_path = None if parents_path is None else parents_path + (self,)
            new_fields_path = None if fields_path is None else fields_path + (field.name,)
            for child in field_value.traverse(
                    parents_path=new_parents_path, fields_path=new_fields_path, traversal_order=traversal_order):
                yield child
        if traversal_order == 'postvisit':
            yield self if parents_path is None else (self, parents_path, fields_path)


@dataclasses.dataclass
class TensorDataClassWithSingleDataTensor:
    tensor: torch.Tensor


@dataclasses.dataclass
class TensorDataClassWithSingleIndicesTensor:
    indices: torch.LongTensor


@dataclasses.dataclass
class TensorDataClassWithSequences:
    sequences: Union[List[torch.Tensor], torch.Tensor]
    batch_first: bool = True
    sequences_lengths: Optional[torch.LongTensor] = dataclasses.field(default=None, init=False)

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        supers = super(TensorDataClassWithSequences, cls).get_management_fields() \
            if hasattr(super(TensorDataClassWithSequences, cls), 'get_management_fields') else ()
        return supers + ('batch_first', 'sequences_lengths')


@final
@dataclasses.dataclass
class TensorWithCollateMask(TensorDataClassWithSingleDataTensor, TensorsDataClass):
    collate_mask: torch.BoolTensor = dataclasses.field(default=None, init=False)

    @classmethod
    def _collate_first_pass(cls, inputs: List['TensorWithCollateMask']):
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


@dataclasses.dataclass
class BatchFlattenedTensorsDataClass(TensorsDataClass, HasSelfIndexingGroup):
    nr_items_per_example: Optional[torch.LongTensor] = dataclasses.field(init=False, default=None)  # (bsz,)
    max_nr_items: Optional[int] = dataclasses.field(init=False, default=None)
    unflattener: Optional[torch.LongTensor] = dataclasses.field(init=False, default=None)
    _nr_examples: Optional[int] = dataclasses.field(init=False, default=None)

    # collate auxiliaries
    batched_index_offset_additive_fix_per_example: Optional[torch.LongTensor] = \
        dataclasses.field(init=False, default=None)

    @property
    def nr_examples(self):
        return self._nr_examples

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        return super(BatchFlattenedTensorsDataClass, cls).get_management_fields() + (
            'nr_items_per_example', 'max_nr_items', 'unflattener',
            'batched_index_offset_additive_fix_per_example', '_nr_examples')

    @classmethod
    def _collate_first_pass(cls, inputs: List['BatchFlattenedTensorsDataClass']) -> 'BatchFlattenedTensorsDataClass':
        data_fields = cls.get_data_fields()
        assert len(data_fields) > 0
        data_tensors_grouped_by_field = {field.name: tuple(getattr(inp, field.name) for inp in inputs)
                                         for field in data_fields}
        assert all(all(isinstance(tensor, torch.Tensor) for tensor in tensors) or
                   all(tensor is None for tensor in tensors)
                   for tensors in data_tensors_grouped_by_field.values())
        non_none_data_tensors_grouped_by_field = {
            k: tensors for k, tensors in data_tensors_grouped_by_field.items() if tensors[0] is not None}
        none_data_fields = tuple(
            field_name for field_name, tensors in data_tensors_grouped_by_field.items() if tensors[0] is None)
        non_none_data_field_names = tuple(
            field_name for field_name, tensors in data_tensors_grouped_by_field.items() if tensors[0] is not None)
        assert len(non_none_data_field_names) > 0
        assert all(all(getattr(inp, field_name).size(0) == getattr(inp, non_none_data_field_names[0]).size(0)
                       for field_name in non_none_data_field_names)
                   for inp in inputs)
        flattened_data_fields = {
            field_name: cls._flatten_tensors(tensors)
            for field_name, tensors in non_none_data_tensors_grouped_by_field.items()}
        flattened = cls(
            self_indexing_group=inputs[0].self_indexing_group,
            **flattened_data_fields,
            **{field_name: None for field_name in none_data_fields})
        flattened.nr_items_per_example = torch.LongTensor(
            [getattr(inp, non_none_data_field_names[0]).size(0) for inp in inputs])
        flattened.max_nr_items = max(getattr(inp, non_none_data_field_names[0]).size(0) for inp in inputs)
        flattened.batched_index_offset_additive_fix_per_example = \
            flattened.nr_items_per_example.cumsum(dim=-1) - flattened.nr_items_per_example
        flattened.unflattener = torch.stack([
            torch.cat([
                torch.arange(getattr(inp, non_none_data_field_names[0]).size(0)) +
                flattened.batched_index_offset_additive_fix_per_example[example_idx],
                torch.zeros(flattened.max_nr_items - getattr(inp, non_none_data_field_names[0]).size(0))], dim=0)
            for example_idx, inp in enumerate(inputs)], dim=0)
        flattened._nr_examples = len(inputs)
        flattened._batch_size = getattr(flattened, non_none_data_field_names[0]).size(0)  # FIXME: does it make sense?
        return flattened

    @classmethod
    def _flatten_tensors(cls, tensors: List[torch.Tensor]):
        return torch.cat(tensors, dim=0)

    def post_collate_remove_unnecessary_collate_info(self):
        self.batched_index_offset_additive_fix_per_example = None

    def unflatten(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # TODO: implement!


@final
@dataclasses.dataclass
class BatchFlattenedTensor(BatchFlattenedTensorsDataClass, TensorDataClassWithSingleDataTensor):
    pass  # the double inheritance is all the impl needed


@final
@dataclasses.dataclass
class BatchFlattenedSeq(BatchFlattenedTensorsDataClass, TensorDataClassWithSequences):
    @classmethod
    def _collate_first_pass(cls, inputs: List['BatchFlattenedSeq']) -> 'BatchFlattenedSeq':
        assert all(inp.batch_first == inputs[0].batch_first for inp in inputs)
        if not inputs[0].batch_first:
            raise NotImplementedError('`batch_first` option is not implemented yet for `BatchFlattenedSeq`.')
        assert all(inp.sequences_lengths is None for inp in inputs)
        assert all(isinstance(inp.sequences, list) for inp in inputs) or \
               all(isinstance(inp.sequences, torch.Tensor) for inp in inputs)
        assert all(isinstance(seq, torch.Tensor) for inp in inputs for seq in inp.sequences)
        assert all(seq.ndim >= 1 for inp in inputs for seq in inp.sequences)
        seq_lengths = [seq.size(0) for inp in inputs for seq in inp.sequences]
        if isinstance(inputs[0].sequences, list):
            fixed_inputs = [
                cls(
                    sequences=collate_tensors_with_variable_shapes(
                        tensors=tuple(inp.sequences), create_collate_mask=False,
                        create_collate_lengths=False, last_variable_dim=0))
                for inp in inputs]
            for inp, fixed_inp in zip(inputs, fixed_inputs):
                for fld in cls.get_management_fields():
                    setattr(fixed_inp, fld, getattr(inp, fld))
            inputs = fixed_inputs
        flattened = super(BatchFlattenedSeq, cls)._collate_first_pass(inputs)
        flattened.sequences_lengths = torch.LongTensor(seq_lengths, device=flattened.sequences.device)
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
class BatchedFlattenedIndicesFlattenedTensorsDataClass(BatchFlattenedTensorsDataClass, HasTargetIndexingGroup):
    # collate auxiliaries
    example_index: Optional[torch.LongTensor] = dataclasses.field(init=False, default=None)

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        return super(BatchedFlattenedIndicesFlattenedTensorsDataClass, cls).get_management_fields() + \
               ('example_index',)

    @classmethod
    def get_indices_fields(cls):
        return super(BatchedFlattenedIndicesFlattenedTensorsDataClass, cls).get_data_fields()

    @classmethod
    def _collate_first_pass(cls, inputs: List['BatchedFlattenedIndicesFlattenedTensorsDataClass']) \
            -> 'BatchedFlattenedIndicesFlattenedTensorsDataClass':
        flattened = super(BatchedFlattenedIndicesFlattenedTensorsDataClass, cls)._collate_first_pass(inputs)
        indices_fields = cls.get_indices_fields()
        flattened.tgt_indexing_group = inputs[0].tgt_indexing_group
        flattened.example_index = torch.cat([
            torch.full(size=(getattr(inp, indices_fields[0].name).size(0),),
                       fill_value=example_idx, dtype=torch.long)
            for example_idx, inp in enumerate(inputs)], dim=0)
        return flattened

    def post_collate_indices_fix(self, parents: Tuple['TensorsDataClass', ...], fields_path: Tuple[str, ...]):
        if self.tgt_indexing_group is None:
            raise ValueError(f'`{self.__class__.__name__}` must have an `tgt_indexing_group`.')
        addressed_flattened_tensor = self.find_addressed_batched_flattened_tensor(parents[0])
        if addressed_flattened_tensor is None:
            raise ValueError(
                f'Not found field in tensors data class which is addressable'
                f'via index group `{self.tgt_indexing_group}`.')
        for field in self.get_indices_fields():
            setattr(self, field.name, getattr(self, field.name) +
                    addressed_flattened_tensor.batched_index_offset_additive_fix_per_example[self.example_index])

    def find_addressed_batched_flattened_tensor(self, tensors_data_class_root: TensorsDataClass) \
            -> Optional['BatchedFlattenedIndicesFlattenedTensorsDataClass']:
        return next((tensor_data_class for tensor_data_class in tensors_data_class_root.traverse()
                     if isinstance(tensor_data_class, BatchFlattenedTensorsDataClass) and
                     tensor_data_class.self_indexing_group == self.tgt_indexing_group), None)

    def post_collate_remove_unnecessary_collate_info(self):
        self.example_index = None


@final
@dataclasses.dataclass
class BatchedFlattenedIndicesFlattenedTensor(BatchedFlattenedIndicesFlattenedTensorsDataClass,
                                             TensorDataClassWithSingleIndicesTensor):
    pass  # the double inheritance is all the impl needed


# TODO: complete implementation!
@final
@dataclasses.dataclass
class BatchedFlattenedIndicesFlattenedSeq(BatchedFlattenedIndicesFlattenedTensorsDataClass,
                                          TensorDataClassWithSequences):
    # TODO: fix collate() to allow different sequences lengths across examples
    #  (each example might have another max length - we should take the max)
    #  We can use here `TensorWithCollateMask.collate()` that actually does the same for the general case.
    #  Except this issue, the super impl `BatchFlattenedTensorsDataClass.collate()` does the other things correctly.
    @classmethod
    def _collate_first_pass(cls, inputs: List['BatchedFlattenedIndicesFlattenedSeq']) \
            -> 'BatchedFlattenedIndicesFlattenedSeq':
        raise NotImplementedError  # TODO: implement!


# TODO: complete implementation!
@final
@dataclasses.dataclass
class BatchedFlattenedIndicesTensor(TensorsDataClass, HasTargetIndexingGroup, TensorDataClassWithSingleDataTensor):
    def post_collate_indices_fix(self, parents: Tuple['TensorsDataClass', ...], fields_path: Tuple[str, ...]):
        raise NotImplementedError  # TODO: implement!
