import torch
import dataclasses
from typing import Callable, Optional


__all__ = ['FlattenedTensor']


@dataclasses.dataclass
class FlattenedTensor:
    flattened: torch.Tensor
    unflattener_fn: Callable[[torch.Tensor], torch.Tensor]
    unflattener_mask: Optional[torch.Tensor] = None
    unflattener_mask_getter: Optional[Callable[[], torch.Tensor]] = None

    def get_unflattener_mask(self) -> Optional[torch.Tensor]:
        if self.unflattener_mask is not None:
            return self.unflattener_mask
        if self.unflattener_mask_getter is not None:
            return self.unflattener_mask_getter()
        return None

    def get_unflattened(self):
        return self.unflattener_fn(self.flattened)
