import torch
import torch.nn as nn


class SymbolsDecoder(nn.Module):
    def __init__(self):
        super(SymbolsDecoder, self).__init__()

    def forward(self, input_state: torch.Tensor, symbols_encodings: torch.Tensor):
        raise NotImplementedError()  # TODO: implement
