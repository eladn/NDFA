import torch
import torch.nn as nn
from typing import Any

from ddfa.code_nn_modules.vocabulary import Vocabulary


class IdentifierEncoder(nn.Module):
    def __init__(self, sub_identifiers_vocab: Vocabulary):
        super(IdentifierEncoder, self).__init__()
        self.sub_identifiers_vocab = sub_identifiers_vocab

    def forward(self, *x: Any):
        raise NotImplementedError()  # TODO: implement
