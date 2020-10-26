import torch
import dataclasses
from typing import List, Union, Optional, Tuple, Dict, Set, Any, final

from .misc import seq_lengths_to_mask, CollateData
from .tensors_data_class import TensorsDataClass
from .mixins import HasSelfIndexingGroupMixin, TensorDataClassWithSingleDataTensorMixin


__all__ = ['BatchFlattenedTensorsDataClassMixin', 'BatchFlattenedTensorsDataClass', 'BatchFlattenedTensor']


@dataclasses.dataclass
class BatchFlattenedTensorsDataClassMixin(HasSelfIndexingGroupMixin):
    nr_items_per_example: Optional[torch.LongTensor] = dataclasses.field(init=False, default=None)  # (bsz,)
    max_nr_items: Optional[int] = dataclasses.field(init=False, default=None)
    unflattener: Optional[torch.LongTensor] = dataclasses.field(init=False, default=None)
    unflattener_mask: Optional[torch.BoolTensor] = dataclasses.field(init=False, default=None)
    flattener: Optional[torch.LongTensor] = dataclasses.field(init=False, default=None)
    _nr_examples: Optional[int] = dataclasses.field(init=False, default=None)

    # collate auxiliaries
    batched_index_offset_additive_fix_per_example: Optional[torch.LongTensor] = \
        dataclasses.field(init=False, default=None)

    @property
    def nr_examples(self):
        return self._nr_examples

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        return super(BatchFlattenedTensorsDataClassMixin, cls).get_management_fields() + (
            'nr_items_per_example', 'max_nr_items', 'unflattener', 'unflattener_mask', 'flattener',
            'batched_index_offset_additive_fix_per_example', '_nr_examples')

    @classmethod
    def _collate_first_pass(cls, inputs: List['BatchFlattenedTensorsDataClassMixin'],
                            collate_data: CollateData) -> 'BatchFlattenedTensorsDataClassMixin':
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
                torch.arange(getattr(inp, non_none_data_field_names[0]).size(0), dtype=torch.long) +
                flattened.batched_index_offset_additive_fix_per_example[example_idx],
                torch.zeros(
                    size=(flattened.max_nr_items - getattr(inp, non_none_data_field_names[0]).size(0),),
                    dtype=torch.long)
            ], dim=0)
            for example_idx, inp in enumerate(inputs)], dim=0)

        flattened.unflattener_mask = seq_lengths_to_mask(
            seq_lengths=flattened.nr_items_per_example, max_seq_len=flattened.unflattener.size(1))
        assert flattened.unflattener_mask.size() == flattened.unflattener.size()

        flattened.flattener = torch.cat([
            torch.arange(nr_items, dtype=torch.long) + example_idx * flattened.max_nr_items
            for example_idx, nr_items in enumerate(flattened.nr_items_per_example)], dim=0)
        flattened._nr_examples = len(inputs)
        flattened._batch_size = getattr(flattened, non_none_data_field_names[0]).size(0)  # FIXME: does it make sense?
        return flattened

    @classmethod
    def _flatten_tensors(cls, tensors: List[torch.Tensor]):
        return torch.cat(tensors, dim=0)

    def post_collate_remove_unnecessary_collate_info(self):
        self.batched_index_offset_additive_fix_per_example = None

    def unflatten(self, tensor: torch.Tensor) -> torch.Tensor:
        expanded_mask = self.unflattener_mask
        tensor_additional_dims = tensor.size()[1:]
        new_dim = expanded_mask.size() + tensor_additional_dims
        expanded_mask = expanded_mask.view(expanded_mask.size() + (1,) * len(tensor_additional_dims)).expand(new_dim)
        return torch.where(
            expanded_mask,
            tensor[self.unflattener],
            torch.zeros(1, dtype=tensor.dtype, device=tensor.device))

    def flatten(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.flatten(0, 1)[self.flattener]


@dataclasses.dataclass
class BatchFlattenedTensorsDataClass(BatchFlattenedTensorsDataClassMixin, TensorsDataClass):
    pass  # the double inheritance is all the impl needed


@final
@dataclasses.dataclass
class BatchFlattenedTensor(BatchFlattenedTensorsDataClass, TensorDataClassWithSingleDataTensorMixin):
    pass  # the double inheritance is all the impl needed
