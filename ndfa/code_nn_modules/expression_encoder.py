import torch
import torch.nn as nn
from functools import reduce

from ndfa.ndfa_model_hyper_parameters import CodeExpressionEncoderParams
from ndfa.nn_utils.misc import get_activation_layer
from ndfa.nn_utils.vocabulary import Vocabulary
from ndfa.misc.code_data_structure_api import *
from ndfa.nn_utils.sequence_encoder import SequenceEncoder
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors


__all__ = ['ExpressionEncoder']


class ExpressionEncoder(nn.Module):
    def __init__(self, kos_tokens_vocab: Vocabulary,
                 tokens_kinds_vocab: Vocabulary,
                 expressions_special_words_vocab: Vocabulary,
                 identifiers_special_words_vocab: Vocabulary,
                 encoder_params: CodeExpressionEncoderParams,
                 identifier_embedding_dim: int, nr_out_linear_layers: int = 1,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        assert nr_out_linear_layers >= 1
        super(ExpressionEncoder, self).__init__()
        self.encoder_params = encoder_params
        self.activation_layer = get_activation_layer(activation_fn)()
        self.kos_tokens_vocab = kos_tokens_vocab
        self.tokens_kinds_vocab = tokens_kinds_vocab
        self.expressions_special_words_vocab = expressions_special_words_vocab
        self.identifiers_special_words_vocab = identifiers_special_words_vocab
        self.identifier_embedding_dim = identifier_embedding_dim
        self.kos_tokens_embedding_layer = nn.Embedding(
            num_embeddings=len(kos_tokens_vocab),
            embedding_dim=self.encoder_params.kos_token_embedding_dim,
            padding_idx=kos_tokens_vocab.get_word_idx('<PAD>'))
        self.tokens_kinds_embedding_layer = nn.Embedding(
            num_embeddings=len(tokens_kinds_vocab),
            embedding_dim=self.encoder_params.token_type_embedding_dim,
            padding_idx=tokens_kinds_vocab.get_word_idx('<PAD>'))
        self.expressions_special_words_embedding_layer = nn.Embedding(
            num_embeddings=len(self.expressions_special_words_vocab),
            embedding_dim=self.encoder_params.token_encoding_dim)
        self.identifiers_special_words_embedding_layer = nn.Embedding(
            num_embeddings=len(self.identifiers_special_words_vocab),
            embedding_dim=self.identifier_embedding_dim)

        assert self.encoder_params.kos_token_embedding_dim == self.identifier_embedding_dim
        self.kos_or_identifier_token_embedding_dim = self.encoder_params.kos_token_embedding_dim
        self.projection_linear_layer = nn.Linear(
            in_features=self.encoder_params.token_type_embedding_dim + self.kos_or_identifier_token_embedding_dim,
            out_features=self.encoder_params.token_encoding_dim)
        self.additional_linear_layers = nn.ModuleList(
            [nn.Linear(self.encoder_params.token_encoding_dim, self.encoder_params.token_encoding_dim)
             for _ in range(nr_out_linear_layers - 1)])

        self.dropout_layer = nn.Dropout(p=dropout_rate)

        self.sequence_encoder = SequenceEncoder(
            encoder_params=self.encoder_params.sequence_encoder,
            input_dim=self.encoder_params.token_encoding_dim,
            hidden_dim=self.encoder_params.token_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(self, expressions: CodeExpressionTokensSequenceInputTensors,
                encoded_identifiers: torch.Tensor):
        token_kind_embeddings = self.tokens_kinds_embedding_layer(expressions.token_type.sequences)
        kos_tokens_embeddings = self.kos_tokens_embedding_layer(expressions.kos_token_index.tensor)
        identifiers_embeddings = encoded_identifiers[expressions.identifier_index.indices]
        is_identifier_token = \
            expressions.token_type.sequences == self.tokens_kinds_vocab.get_word_idx(SerTokenKind.IDENTIFIER.value)

        is_identifier_token_mask = is_identifier_token.unsqueeze(-1).expand(
            is_identifier_token.size() + (self.identifier_embedding_dim,))
        token_kinds_for_kos_tokens_vocab = (
            self.tokens_kinds_vocab.get_word_idx(SerTokenKind.OPERATOR.value),
            self.tokens_kinds_vocab.get_word_idx(SerTokenKind.SEPARATOR.value),
            self.tokens_kinds_vocab.get_word_idx(SerTokenKind.KEYWORD.value))
        is_kos_token = reduce(
            torch.Tensor.logical_or,
            ((expressions.token_type.sequences == token_kind) for token_kind in token_kinds_for_kos_tokens_vocab))
        is_kos_token_mask = is_kos_token.unsqueeze(-1).expand(
            is_kos_token.size() + (self.encoder_params.kos_token_embedding_dim,))

        # Note: we could consider concatenate kos & identifier embeddings: <kos|None> or <None|identifier>.
        kos_or_identifier_token_encoding = torch.zeros(
            size=token_kind_embeddings.size()[:-1] + (self.kos_or_identifier_token_embedding_dim,),
            dtype=identifiers_embeddings.dtype, device=identifiers_embeddings.device)

        kos_or_identifier_token_encoding.masked_scatter_(
            mask=is_identifier_token_mask,
            source=identifiers_embeddings)
        kos_or_identifier_token_encoding = kos_or_identifier_token_encoding.masked_scatter(
            is_kos_token_mask, kos_tokens_embeddings)

        final_token_seqs_encodings = torch.cat([token_kind_embeddings, kos_or_identifier_token_encoding], dim=-1)
        assert final_token_seqs_encodings.size()[:-1] == expressions.token_type.sequences.size()

        final_token_seqs_encodings = self.dropout_layer(final_token_seqs_encodings)
        final_token_seqs_encodings_projected = self.dropout_layer(self.activation_layer(
            self.projection_linear_layer(final_token_seqs_encodings)))
        for linear_layer in self.additional_linear_layers:
            final_token_seqs_encodings_projected = self.dropout_layer(self.activation_layer(linear_layer(
                final_token_seqs_encodings_projected)))

        expressions_encodings = self.sequence_encoder(
            sequence_input=final_token_seqs_encodings_projected,
            lengths=expressions.token_type.sequences_lengths, batch_first=True).sequence

        return expressions_encodings
