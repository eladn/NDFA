import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm


__all__ = ['NormWrapper']


class NormWrapper(nn.Module):
    def __init__(self, nr_features: int, affine: bool = True, norm_type: str = 'layer'):
        super(NormWrapper, self).__init__()
        self.norm_type = norm_type
        if norm_type == 'layer':
            self.layer_norm = LayerNorm(nr_features, elementwise_affine=affine)
        elif norm_type == 'batch':
            self.batch_norm = nn.BatchNorm1d(nr_features, affine=affine)
        else:
            raise ValueError(f'Unsupported norm type `{norm_type}`. Use either `layer` or `batch`.')

    def forward(self, inp):
        if self.norm_type == 'layer':
            return self.layer_norm(inp)
        elif self.norm_type == 'batch':
            if inp.ndim == 3:
                inp = inp.permute(0, 2, 1)
            ret = self.batch_norm(inp)
            if inp.ndim == 3:
                ret = ret.permute(0, 2, 1)
            return ret
        else:
            assert False
