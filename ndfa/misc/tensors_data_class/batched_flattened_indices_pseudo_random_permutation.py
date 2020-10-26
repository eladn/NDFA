import torch
import hashlib
import dataclasses
import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Set, Any, final

from .misc import collate_tensors_with_variable_shapes, CollateData
from .tensors_data_class import TensorsDataClass
from .mixins import HasTargetIndexingGroupMixin


__all__ = ['BatchedFlattenedIndicesPseudoRandomPermutation']


@final
@dataclasses.dataclass
class BatchedFlattenedIndicesPseudoRandomPermutation(HasTargetIndexingGroupMixin, TensorsDataClass):
    permutations: torch.LongTensor = dataclasses.field(default=None, init=False)
    inverse_permutations: torch.LongTensor = dataclasses.field(default=None, init=False)

    batch_dependent_seed: bool = dataclasses.field(default=True)
    example_dependent_seed: bool = dataclasses.field(default=True)
    initial_seed_salt: str = dataclasses.field(default='0')

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        return super(BatchedFlattenedIndicesPseudoRandomPermutation, cls).get_management_fields() + \
               ('batch_dependent_seed', 'example_dependent_seed', 'initial_seed_salt')

    @classmethod
    def get_indices_fields(cls) -> Tuple[dataclasses.Field, ...]:
        return tuple(field for field in dataclasses.fields(cls)
                     if field.name in {'permutations', 'inverse_permutations'})

    @classmethod
    def _collate_first_pass(
            cls, inputs: List['BatchedFlattenedIndicesPseudoRandomPermutation'],
            collate_data: CollateData) \
            -> 'BatchedFlattenedIndicesPseudoRandomPermutation':
        collated = super(BatchedFlattenedIndicesPseudoRandomPermutation, cls)._collate_first_pass(
            inputs, collate_data=collate_data)
        collated.tgt_indexing_group = inputs[0].tgt_indexing_group
        return collated

    def post_collate_indices_fix(self, parents: Tuple['TensorsDataClass', ...], fields_path: Tuple[str, ...],
                                 collate_data: CollateData):
        if self.tgt_indexing_group is None:
            raise ValueError(f'`{self.__class__.__name__}` must have an `tgt_indexing_group`.')
        addressed_flattened_tensor = self.find_addressed_batched_flattened_tensor(parents[0])
        if addressed_flattened_tensor is None:
            raise ValueError(
                f'Not found field in tensors data class which is addressable '
                f'via index group `{self.tgt_indexing_group}`.')
        nr_items_per_example = addressed_flattened_tensor.nr_items_per_example
        index_offsets = addressed_flattened_tensor.batched_index_offset_additive_fix_per_example

        if self.batch_dependent_seed and self.example_dependent_seed:
            random_seed_per_example = [
                int(hashlib.sha256(f'{self.initial_seed_salt}|{"-".join(collate_data.example_hashes)}|{example_idx}'
                                   .encode('ascii')).hexdigest(), 16) % (2 ** 32)
                for example_idx, _ in enumerate(collate_data.example_hashes)]
        elif not self.batch_dependent_seed and self.example_dependent_seed:
            random_seed_per_example = [
                int(hashlib.sha256(f'{self.initial_seed_salt}|{example_hash}'
                                   .encode('ascii')).hexdigest(), 16) % (2 ** 32)
                for example_hash in collate_data.example_hashes]
        elif self.batch_dependent_seed and not self.example_dependent_seed:
            random_seed_per_example = [
                int(hashlib.sha256(f'{self.initial_seed_salt}|{"-".join(collate_data.example_hashes)}'
                                   .encode('ascii')).hexdigest(), 16) % (2 ** 32)
                for _ in collate_data.example_hashes]
        else:
            random_seed_per_example = [
                int(hashlib.sha256(f'{self.initial_seed_salt}'
                                   .encode('ascii')).hexdigest(), 16) % (2 ** 32)
                for _ in collate_data.example_hashes]

        permutations_without_offsets = [
            torch.LongTensor(np.random.RandomState(random_seed_per_example[example_idx]).permutation(int(nr_items)))
            for example_idx, nr_items in enumerate(nr_items_per_example)]
        # TODO: is it always correct that perm^2 == perm^-1
        inverse_permutations_without_offsets = [perm[perm] for perm in permutations_without_offsets]
        permutations_with_offsets = [
            perm + index_offset for perm, index_offset in zip(permutations_without_offsets, index_offsets)]
        inverse_permutations_with_ranges = [
            perm + index_offset for perm, index_offset in zip(inverse_permutations_without_offsets, index_offsets)]
        self.permutations = collate_tensors_with_variable_shapes(
            tensors=tuple(permutations_with_offsets), create_collate_mask=False,
            create_collate_lengths=False, last_variable_dim=0)
        self.inverse_permutations = collate_tensors_with_variable_shapes(
            tensors=tuple(inverse_permutations_with_ranges), create_collate_mask=False,
            create_collate_lengths=False, last_variable_dim=0)

