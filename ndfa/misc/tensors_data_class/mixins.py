import torch
import dataclasses
from typing import List, Union, Optional, Tuple, Dict, Set, Any, final


__all__ = [
    'HasSelfIndexingGroupMixin',
    'HasTargetIndexingGroupMixin',
    'TensorDataClassWithSingleDataTensorMixin',
    'TensorDataClassWithSingleIndicesTensorMixin',
    'TensorDataClassWithSingleSequenceFieldMixin',
    'TensorDataClassWithSequencesMixin']


@dataclasses.dataclass
class HasSelfIndexingGroupMixin:
    self_indexing_group: str = None

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        supers = super(HasSelfIndexingGroupMixin, cls).get_management_fields() \
            if hasattr(super(HasSelfIndexingGroupMixin, cls), 'get_management_fields') else ()
        return supers + ('self_indexing_group',)


@dataclasses.dataclass
class HasTargetIndexingGroupMixin:
    tgt_indexing_group: str = None

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        supers = super(HasTargetIndexingGroupMixin, cls).get_management_fields() \
            if hasattr(super(HasTargetIndexingGroupMixin, cls), 'get_management_fields') else ()
        return supers + ('tgt_indexing_group',)

    def find_addressed_batched_flattened_tensor(self, tensors_data_class_root: 'TensorsDataClass') \
            -> Optional['HasSelfIndexingGroupMixin']:
        # TODO: is it sufficient to check isinstance(HasSelfIndexingGroupMixin)?
        #  doesn't it have to be 'BatchFlattenedTensorsDataClass'?
        return next((tensor_data_class for tensor_data_class in tensors_data_class_root.traverse()
                     if isinstance(tensor_data_class, HasSelfIndexingGroupMixin) and
                     tensor_data_class.self_indexing_group == self.tgt_indexing_group), None)


@dataclasses.dataclass
class TensorDataClassWithSingleDataTensorMixin:
    tensor: torch.Tensor


@dataclasses.dataclass
class TensorDataClassWithSingleIndicesTensorMixin:
    indices: torch.LongTensor


@dataclasses.dataclass
class TensorDataClassWithSingleSequenceFieldMixin:
    sequences: Union[List[torch.Tensor], torch.Tensor, List[List[str]]]


@dataclasses.dataclass
class TensorDataClassWithSequencesMixin:
    batch_first: bool = True
    sequences_lengths: Optional[torch.LongTensor] = dataclasses.field(default=None, init=False)
    sequences_mask: Optional[torch.BoolTensor] = dataclasses.field(default=None, init=False)
    max_sequence_length: Optional[int] = dataclasses.field(default=None, init=False)

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        supers = super(TensorDataClassWithSequencesMixin, cls).get_management_fields() \
            if hasattr(super(TensorDataClassWithSequencesMixin, cls), 'get_management_fields') else ()
        return supers + ('batch_first', 'sequences_lengths', 'sequences_mask', 'max_sequence_length')
