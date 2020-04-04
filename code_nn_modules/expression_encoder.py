import torch
import torch.nn as nn
from typing import Any


class ExpressionEncoder(nn.Module):
    def __init__(self):
        super(ExpressionEncoder, self).__init__()

    def forward(self, *x: Any):
        raise NotImplementedError()  # TODO: implement
