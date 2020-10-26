import torch
import hashlib
import dataclasses
import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Set, Any, final

from .misc import CollateData
from .tensors_data_class import TensorsDataClass


__all__ = ['BatchFlattenedPseudoRandomSamplerFromRange']


@final
@dataclasses.dataclass
class BatchFlattenedPseudoRandomSamplerFromRange(TensorsDataClass):
    sample: torch.LongTensor = dataclasses.field(default=None, init=False)

    sample_size: Union[int, Tuple[int]] = dataclasses.field(default=0)
    tgt_range_start: int = dataclasses.field(default=0)
    tgt_range_end: int = dataclasses.field(default=0)
    initial_seed_salt: str = dataclasses.field(default='0')
    replace: bool = dataclasses.field(default=False)

    @classmethod
    def get_management_fields(cls) -> Tuple[str, ...]:
        return super(BatchFlattenedPseudoRandomSamplerFromRange, cls).get_management_fields() + \
               ('sample_size', 'tgt_range_start', 'tgt_range_end', 'initial_seed_salt', 'replace')

    @classmethod
    def _collate_first_pass(
            cls, inputs: List['BatchFlattenedPseudoRandomSamplerFromRange'],
            collate_data: CollateData) \
            -> 'BatchFlattenedPseudoRandomSamplerFromRange':
        sample_size = sum(inp.sample_size for inp in inputs)
        tgt_range_start = inputs[0].tgt_range_start
        tgt_range_end = inputs[0].tgt_range_end
        initial_seed_salt = inputs[0].initial_seed_salt
        replace = inputs[0].replace
        assert all(inp.tgt_range_start == tgt_range_start for inp in inputs)
        assert all(inp.tgt_range_end == tgt_range_end for inp in inputs)
        assert all(inp.initial_seed_salt == initial_seed_salt for inp in inputs)
        assert all(inp.replace == replace for inp in inputs)
        random_seed = \
            int(hashlib.sha256(f'{initial_seed_salt}|{"-".join(collate_data.example_hashes)}'
                               .encode('ascii')).hexdigest(), 16) % (2 ** 32)
        sample = torch.LongTensor(
            np.random.RandomState(random_seed).choice(
                a=np.arange(tgt_range_start, tgt_range_end), size=sample_size, replace=replace))
        collated = cls(
            sample_size=sample_size,
            tgt_range_start=tgt_range_start,
            tgt_range_end=tgt_range_end,
            initial_seed_salt=initial_seed_salt,
            replace=replace)
        collated.sample = sample
        return collated
