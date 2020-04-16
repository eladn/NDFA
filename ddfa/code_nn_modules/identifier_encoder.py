import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm

from ddfa.code_nn_modules.vocabulary import Vocabulary


class IdentifierEncoder(nn.Module):
    def __init__(self, sub_identifiers_vocab: Vocabulary):
        super(IdentifierEncoder, self).__init__()
        self.sub_identifiers_vocab = sub_identifiers_vocab
        transformer_encoder_layer = TransformerEncoderLayer(d_model=256, nhead=1, dim_feedforward=1028)
        encoder_norm = LayerNorm(256)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=transformer_encoder_layer, num_layers=3, norm=encoder_norm)

    def forward(self, sub_identifiers: torch.Tensor):
        return self.transformer_encoder(sub_identifiers).sum(dim=0)
