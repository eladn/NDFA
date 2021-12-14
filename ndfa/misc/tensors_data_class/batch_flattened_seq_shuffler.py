__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-11-15"

import torch
import dataclasses
import numpy as np
from typing import List, Tuple, final, Optional

from .misc import collate_tensors_with_variable_shapes, CollateData, inverse_permutation, \
    seq_lengths_to_mask, get_random_seed_per_example
from .tensors_data_class import TensorsDataClass
from .mixins import TensorDataClassWithSequencesMixin


__all__ = ['BatchFlattenedSeqShuffler', 'batch_flattened_seq_shuffler_field', 'FragmentSizeDistribution']


@dataclasses.dataclass
class FragmentSizeDistribution:
    min_fragment_size: int = 2
    max_fragment_size: int = 20

    def fragment(self, seq_len: int, rng: np.random.RandomState) -> Tuple[Tuple[int, int], ...]:
        # Note: it is possible to have a fragment of size < `min_fragment_size`.
        allocated_len = 0
        fragments_lengths = []
        while allocated_len < seq_len:
            remaining_len = seq_len - allocated_len
            cur_fragment_len = \
                min(remaining_len, rng.randint(low=self.min_fragment_size, high=self.max_fragment_size + 1))
            fragments_lengths.append(cur_fragment_len)
            allocated_len += cur_fragment_len
        rng.shuffle(fragments_lengths)
        return tuple(
            (so_far - fragment_len, so_far - 1)
            for fragment_len, so_far in zip(fragments_lengths, np.cumsum(fragments_lengths)))


@final
@dataclasses.dataclass
class BatchFlattenedSeqShuffler(TensorDataClassWithSequencesMixin, TensorsDataClass):
    permutations: torch.LongTensor = dataclasses.field(default=None, init=False)
    inverse_permutations: torch.LongTensor = dataclasses.field(default=None, init=False)

    lengths: Tuple[int, ...] = dataclasses.field(default=())
    batch_dependent_seed: bool = dataclasses.field(default=True)
    example_dependent_seed: bool = dataclasses.field(default=True)
    initial_seed_salt: str = dataclasses.field(default='0')
    fragmented_shuffling_fragments_size_distribution: Optional[FragmentSizeDistribution] = \
        dataclasses.field(default=None)

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        return super(BatchFlattenedSeqShuffler, cls).get_management_fields() + \
               ('lengths', 'batch_dependent_seed', 'example_dependent_seed', 'initial_seed_salt',
                'fragmented_shuffling_fragments_size_distribution')

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

        fragmented_shuffling_fragments_size_distribution = inputs[0].fragmented_shuffling_fragments_size_distribution
        if fragmented_shuffling_fragments_size_distribution is None:
            permutations = [
                torch.LongTensor(random_state_per_example[example_idx].permutation(int(nr_items)))
                for example_idx, inp in enumerate(inputs)
                for nr_items in inp.lengths]
        else:
            permutations = [
                torch.LongTensor(np.concatenate([
                    random_state_per_example[example_idx].permutation(np.arange(fragment[0], fragment[1] + 1))
                    for fragment in fragmented_shuffling_fragments_size_distribution.fragment(
                        seq_len=nr_items, rng=random_state_per_example[example_idx])]))
                for example_idx, inp in enumerate(inputs)
                for nr_items in inp.lengths]
        inverse_permutations = [inverse_permutation(perm) for perm in permutations]
        collated.lengths = tuple(length for inp in inputs for length in inp.lengths)
        collated.sequences_lengths = torch.LongTensor(collated.lengths)
        collated.max_sequence_length = max(collated.lengths)
        collated.sequences_mask = seq_lengths_to_mask(
            seq_lengths=collated.sequences_lengths, max_seq_len=collated.max_sequence_length)
        collated.permutations = collate_tensors_with_variable_shapes(
            tensors=tuple(permutations), create_collate_mask=False,
            create_collate_lengths=False, last_variable_dim=0)
        collated.inverse_permutations = collate_tensors_with_variable_shapes(
            tensors=tuple(inverse_permutations), create_collate_mask=False,
            create_collate_lengths=False, last_variable_dim=0)

        return collated

    def shuffle(self, sequence_input: torch.Tensor) -> torch.Tensor:
        assert sequence_input.shape[:-1] == self.permutations.shape
        extended_perm = self.permutations.unsqueeze(-1).expand(sequence_input.shape)
        shuffled_seqs = torch.gather(input=sequence_input, dim=1, index=extended_perm)
        shuffled_seqs = shuffled_seqs.masked_fill(
            ~self.sequences_mask.unsqueeze(-1), 0)
        return shuffled_seqs

    def unshuffle(self, shuffled_sequence_input: torch.Tensor) -> torch.Tensor:
        assert shuffled_sequence_input.shape[:-1] == self.inverse_permutations.shape
        extended_inv_perm = self.inverse_permutations.unsqueeze(-1).expand(shuffled_sequence_input.shape)
        unshuffled_seqs = torch.gather(input=shuffled_sequence_input, dim=1, index=extended_inv_perm)
        unshuffled_seqs = unshuffled_seqs.masked_fill(
            ~self.sequences_mask.unsqueeze(-1), 0)
        return unshuffled_seqs


def batch_flattened_seq_shuffler_field(
        *,
        default=dataclasses.MISSING,
        batch_dependent_seed: bool = dataclasses.MISSING,
        example_dependent_seed: bool = dataclasses.MISSING,
        initial_seed_salt: str = dataclasses.MISSING,
        fragmented_shuffling_fragments_size_distribution: FragmentSizeDistribution = dataclasses.MISSING) \
        -> dataclasses.Field:
    management_fields_defaults = {
        'batch_dependent_seed': batch_dependent_seed,
        'example_dependent_seed': example_dependent_seed,
        'initial_seed_salt': initial_seed_salt,
        'fragmented_shuffling_fragments_size_distribution': fragmented_shuffling_fragments_size_distribution}
    management_fields_defaults = {k: v for k, v in management_fields_defaults.items() if v is not dataclasses.MISSING}
    return dataclasses.field(default=default, metadata=management_fields_defaults)
