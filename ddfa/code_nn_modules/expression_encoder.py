import torch
import torch.nn as nn

from ddfa.code_nn_modules.vocabulary import Vocabulary


class ExpressionEncoder(nn.Module):
    def __init__(self, tokens_vocab: Vocabulary, tokens_kinds_vocab: Vocabulary, identifier_encoder: nn.Module):
        super(ExpressionEncoder, self).__init__()
        self.tokens_vocab = tokens_vocab
        self.tokens_kinds_vocab = tokens_kinds_vocab
        self.identifier_encoder = identifier_encoder

    def forward(self, expressions: torch.Tensor, encoded_identifiers: torch.Tensor):
        raise NotImplementedError()  # TODO: implement
