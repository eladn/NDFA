import torch
import torch.nn as nn
from typing import Any


class IdentifierEncoder(nn.Module):
    def __init__(self):
        super(IdentifierEncoder, self).__init__()

    def forward(self, *x: Any):
        raise NotImplementedError()  # TODO: implement
