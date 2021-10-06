import torch
import dataclasses
from typing import Callable


__all__ = ['FlattenedTensor']


@dataclasses.dataclass
class FlattenedTensor:
    flattened: torch.Tensor
    unflattener_mask: torch.Tensor
    unflattener: Callable[[torch.Tensor], torch.Tensor]

    def get_unflattened(self):
        return self.unflattener(self.flattened)
