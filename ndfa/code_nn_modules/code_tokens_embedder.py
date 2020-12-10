import torch
import torch.nn as nn
from functools import reduce

from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.misc.code_data_structure_api import *


__all__ = ['CodeTokensEmbedder']


class CodeTokensEmbedder(nn.Module):
    def __init__(self,
                 kos_tokens_vocab: Vocabulary,
                 tokens_kinds_vocab: Vocabulary,
                 token_encoding_dim: int,
                 kos_token_embedding_dim: int,
                 token_type_embedding_dim: int,
                 identifier_embedding_dim: int,
                 nr_out_linear_layers: int = 1,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        assert nr_out_linear_layers >= 1
        super(CodeTokensEmbedder, self).__init__()
        self.kos_tokens_vocab = kos_tokens_vocab
        self.tokens_kinds_vocab = tokens_kinds_vocab
        self.token_encoding_dim = token_encoding_dim
        self.kos_token_embedding_dim = kos_token_embedding_dim
        self.token_type_embedding_dim = token_type_embedding_dim
        self.identifier_embedding_dim = identifier_embedding_dim
        self.kos_tokens_embedding_layer = nn.Embedding(
            num_embeddings=len(kos_tokens_vocab),
            embedding_dim=self.kos_token_embedding_dim,
            padding_idx=kos_tokens_vocab.get_word_idx('<PAD>'))
        self.tokens_kinds_embedding_layer = nn.Embedding(
            num_embeddings=len(tokens_kinds_vocab),
            embedding_dim=self.token_type_embedding_dim,
            padding_idx=tokens_kinds_vocab.get_word_idx('<PAD>'))

        assert self.kos_token_embedding_dim == self.identifier_embedding_dim
        self.kos_or_identifier_token_embedding_dim = self.kos_token_embedding_dim
        self.projection_linear_layer = nn.Linear(
            in_features=self.token_type_embedding_dim + self.kos_or_identifier_token_embedding_dim,
            out_features=self.token_encoding_dim)
        self.additional_linear_layers = nn.ModuleList(
            [nn.Linear(self.token_encoding_dim, self.token_encoding_dim)
             for _ in range(nr_out_linear_layers - 1)])

        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self,
            token_type: torch.LongTensor,
            kos_token_index: torch.LongTensor,
            identifier_index: torch.LongTensor,
            encoded_identifiers: torch.Tensor):
        # Currently used for sequences of tokens. I never checked whether this code will work for 1D tokens tensor
        assert token_type.ndim == 2
        assert kos_token_index.ndim == 1 and identifier_index.ndim == 1
        assert encoded_identifiers.ndim == 2
        token_kind_embeddings = self.tokens_kinds_embedding_layer(token_type)
        kos_tokens_embeddings = self.kos_tokens_embedding_layer(kos_token_index)
        identifiers_occurrences_embeddings = encoded_identifiers[identifier_index]
        is_identifier_token = \
            token_type == self.tokens_kinds_vocab.get_word_idx(SerTokenKind.IDENTIFIER.value)

        is_identifier_token_mask = is_identifier_token.unsqueeze(-1).expand(
            is_identifier_token.size() + (self.identifier_embedding_dim,))
        token_kinds_for_kos_tokens_vocab = (
            self.tokens_kinds_vocab.get_word_idx(SerTokenKind.OPERATOR.value),
            self.tokens_kinds_vocab.get_word_idx(SerTokenKind.SEPARATOR.value),
            self.tokens_kinds_vocab.get_word_idx(SerTokenKind.KEYWORD.value))
        is_kos_token = reduce(
            torch.Tensor.logical_or,
            ((token_type == token_kind) for token_kind in token_kinds_for_kos_tokens_vocab))
        is_kos_token_mask = is_kos_token.unsqueeze(-1).expand(
            is_kos_token.size() + (self.kos_token_embedding_dim,))

        # Note: we could consider concatenate kos & identifier embeddings: <kos|None> or <None|identifier>.
        kos_or_identifier_token_encoding = torch.zeros(
            size=token_kind_embeddings.size()[:-1] + (self.kos_or_identifier_token_embedding_dim,),
            dtype=identifiers_occurrences_embeddings.dtype, device=identifiers_occurrences_embeddings.device)

        kos_or_identifier_token_encoding.masked_scatter_(
            mask=is_identifier_token_mask,
            source=identifiers_occurrences_embeddings)
        kos_or_identifier_token_encoding = kos_or_identifier_token_encoding.masked_scatter(
            is_kos_token_mask, kos_tokens_embeddings)

        final_token_embeddings = torch.cat([token_kind_embeddings, kos_or_identifier_token_encoding], dim=-1)
        assert final_token_embeddings.size()[:-1] == token_type.size()

        final_token_embeddings = self.dropout_layer(final_token_embeddings)
        final_token_encodings_projected = self.dropout_layer(self.activation_layer(
            self.projection_linear_layer(final_token_embeddings)))
        for linear_layer in self.additional_linear_layers:
            final_token_encodings_projected = self.dropout_layer(self.activation_layer(linear_layer(
                final_token_encodings_projected)))

        return final_token_encodings_projected
