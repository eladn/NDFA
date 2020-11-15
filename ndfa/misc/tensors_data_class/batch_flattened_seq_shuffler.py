import torch
import hashlib
import dataclasses
import numpy as np
from typing import List, Tuple, final

from .misc import collate_tensors_with_variable_shapes, CollateData
from .tensors_data_class import TensorsDataClass


__all__ = ['BatchFlattenedSeqShuffler']


def get_random_seed_per_example(
        batch_dependent_seed: bool, example_dependent_seed: bool,
        initial_seed_salt: str, collate_data: CollateData) -> List[int]:
    if batch_dependent_seed and example_dependent_seed:
        return [
            int(hashlib.sha256(f'{initial_seed_salt}|{"-".join(collate_data.example_hashes)}|{example_idx}'
                               .encode('ascii')).hexdigest(), 16) % (2 ** 32)
            for example_idx, _ in enumerate(collate_data.example_hashes)]
    elif not batch_dependent_seed and example_dependent_seed:
        return [
            int(hashlib.sha256(f'{initial_seed_salt}|{example_hash}'
                               .encode('ascii')).hexdigest(), 16) % (2 ** 32)
            for example_hash in collate_data.example_hashes]
    elif batch_dependent_seed and not example_dependent_seed:
        return [
            int(hashlib.sha256(f'{initial_seed_salt}|{"-".join(collate_data.example_hashes)}'
                               .encode('ascii')).hexdigest(), 16) % (2 ** 32)
            for _ in collate_data.example_hashes]
    else:
        return [
            int(hashlib.sha256(f'{initial_seed_salt}'
                               .encode('ascii')).hexdigest(), 16) % (2 ** 32)
            for _ in collate_data.example_hashes]


@final
@dataclasses.dataclass
class BatchFlattenedSeqShuffler(TensorsDataClass):
    permutations: torch.LongTensor = dataclasses.field(default=None, init=False)
    inverse_permutations: torch.LongTensor = dataclasses.field(default=None, init=False)

    lengths: Tuple[int, ...] = dataclasses.field(default=())
    batch_dependent_seed: bool = dataclasses.field(default=True)
    example_dependent_seed: bool = dataclasses.field(default=True)
    initial_seed_salt: str = dataclasses.field(default='0')

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        return super(BatchFlattenedSeqShuffler, cls).get_management_fields() + \
               ('lengths', 'batch_dependent_seed', 'example_dependent_seed', 'initial_seed_salt')

    @classmethod
    def get_indices_fields(cls) -> Tuple[dataclasses.Field, ...]:
        return tuple(field for field in dataclasses.fields(cls)
                     if field.name in {'permutations', 'inverse_permutations'})

    @classmethod
    def _collate_first_pass(
            cls, inputs: List['BatchFlattenedSeqShuffler'],
            collate_data: CollateData) \
            -> 'BatchFlattenedSeqShuffler':
        collated = super(BatchFlattenedSeqShuffler, cls)._collate_first_pass(
            inputs, collate_data=collate_data)
        batch_dependent_seed = inputs[0].batch_dependent_seed
        example_dependent_seed = inputs[0].example_dependent_seed
        initial_seed_salt = inputs[0].initial_seed_salt

        random_seed_per_example = get_random_seed_per_example(
            batch_dependent_seed=batch_dependent_seed,
            example_dependent_seed=example_dependent_seed,
            initial_seed_salt=initial_seed_salt, collate_data=collate_data)
        random_state_per_example = [np.random.RandomState(rs) for rs in random_seed_per_example]

        permutations = [
            torch.LongTensor(random_state_per_example[example_idx].permutation(int(nr_items)))
            for example_idx, inp in enumerate(inputs)
            for nr_items in inp.lengths]
        # TODO: is it always correct that perm^2 == perm^-1
        inverse_permutations = [perm[perm] for perm in permutations]
        collated.lengths = tuple(length for inp in inputs for length in inp.lengths)
        collated.permutations = collate_tensors_with_variable_shapes(
            tensors=tuple(permutations), create_collate_mask=False,
            create_collate_lengths=False, last_variable_dim=0)
        collated.inverse_permutations = collate_tensors_with_variable_shapes(
            tensors=tuple(inverse_permutations), create_collate_mask=False,
            create_collate_lengths=False, last_variable_dim=0)

        return collated
