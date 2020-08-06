import torch
import dataclasses


__all__ = ['ScatteredEncodings']


@dataclasses.dataclass
class ScatteredEncodings:
    encodings: torch.FloatTensor
    indices: torch.LongTensor
    mask: torch.BoolTensor
