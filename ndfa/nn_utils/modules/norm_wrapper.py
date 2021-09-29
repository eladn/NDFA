import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from typing import Literal, Optional

from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams


__all__ = ['NormWrapper']


class NormWrapper(nn.Module):
    def __init__(self, nr_features: int,
                 affine: bool = True,
                 norm_type: Literal['later', 'batch'] = 'layer',
                 params: Optional[NormWrapperParams] = None):
        super(NormWrapper, self).__init__()
        self.params = NormWrapperParams(norm_type=norm_type, affine=affine) if params is None else params
        if self.params.norm_type == 'layer':
            self.layer_norm = LayerNorm(nr_features, elementwise_affine=self.params.affine)
        elif self.params.norm_type == 'batch':
            self.batch_norm = nn.BatchNorm1d(nr_features, affine=self.params.affine)
        else:
            raise ValueError(f'Unsupported norm type `{self.params.norm_type}`. Use either `layer` or `batch`.')

    def forward(self, inp):
        if self.params.norm_type == 'layer':
            return self.layer_norm(inp)
        elif self.params.norm_type == 'batch':
            if inp.ndim == 3:
                inp = inp.permute(0, 2, 1)
            ret = self.batch_norm(inp)
            if inp.ndim == 3:
                ret = ret.permute(0, 2, 1)
            return ret
        else:
            assert False
