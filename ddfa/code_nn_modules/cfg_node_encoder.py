import torch
import torch.nn as nn
from typing import Any


class CFGNodeEncoder(nn.Module):
    def __init__(self):
        super(CFGNodeEncoder, self).__init__()

    def forward(self, *x: Any):
        raise NotImplementedError()  # TODO: implement
