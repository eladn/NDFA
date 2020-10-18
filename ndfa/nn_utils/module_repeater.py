import torch
import torch.nn as nn
from typing import Union, Collection


__all__ = ['ModuleRepeater']


class ModuleRepeater(nn.Module):
    def __init__(self, module_create_fn, repeats: Union[int, Collection],
                 share: bool = False, repeat_key: str = 'repeat_key'):
        super(ModuleRepeater, self).__init__()
        if isinstance(repeats, int):
            repeats = range(repeats)
        self.repeats = tuple(repeats)
        self.share = share
        self.repeat_key = repeat_key
        if self.share:
            self.single_inner_module = module_create_fn()
        else:
            self.inner_modules_dict = nn.ModuleDict({str(k): module_create_fn() for k in self.repeats})

    def forward(self, *args, **kwargs):
        assert self.repeat_key in kwargs
        repeat_key = kwargs[self.repeat_key]
        assert repeat_key in self.repeats
        module = self.single_inner_module if self.share else self.inner_modules_dict[str(repeat_key)]
        kwargs = {k: v for k, v in kwargs.items() if k != self.repeat_key}
        return module(*args, **kwargs)
