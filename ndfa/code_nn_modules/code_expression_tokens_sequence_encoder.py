import torch
import torch.nn as nn

from ndfa.ndfa_model_hyper_parameters import CodeExpressionEncoderParams
from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors
from ndfa.code_nn_modules.code_tokens_embedder import CodeTokensEmbedder


__all__ = ['CodeExpressionTokensSequenceEncoder']


class CodeExpressionTokensSequenceEncoder(nn.Module):
    def __init__(self, kos_tokens_vocab: Vocabulary,
                 tokens_kinds_vocab: Vocabulary,
                 encoder_params: CodeExpressionEncoderParams,
                 identifier_embedding_dim: int, nr_out_linear_layers: int = 1,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionTokensSequenceEncoder, self).__init__()
        self.encoder_params = encoder_params
        self.code_tokens_embedder = CodeTokensEmbedder(
            kos_tokens_vocab=kos_tokens_vocab,
            tokens_kinds_vocab=tokens_kinds_vocab,
            token_encoding_dim=self.encoder_params.token_encoding_dim,
            kos_token_embedding_dim=self.encoder_params.kos_token_embedding_dim,
            token_type_embedding_dim=self.encoder_params.token_type_embedding_dim,
            identifier_embedding_dim=identifier_embedding_dim,
            nr_out_linear_layers=nr_out_linear_layers,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.sequence_encoder = SequenceEncoder(
            encoder_params=self.encoder_params.sequence_encoder,
            input_dim=self.encoder_params.token_encoding_dim,
            hidden_dim=self.encoder_params.token_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, expressions_input: CodeExpressionTokensSequenceInputTensors,
                encoded_identifiers: torch.Tensor):
        token_seqs_embeddings = self.code_tokens_embedder(
            token_type=expressions_input.token_type.sequences,
            kos_token_index=expressions_input.kos_token_index.tensor,
            identifier_index=expressions_input.identifier_index.indices,
            encoded_identifiers=encoded_identifiers)

        if self.encoder_params.shuffle_expressions:
            token_seqs_embeddings = expressions_input.sequence_shuffler.shuffle(token_seqs_embeddings)

        expressions_encodings = self.sequence_encoder(
            sequence_input=token_seqs_embeddings,
            lengths=expressions_input.token_type.sequences_lengths, batch_first=True).sequence

        if self.encoder_params.shuffle_expressions:
            expressions_encodings = expressions_input.sequence_shuffler.unshuffle(expressions_encodings)

        return expressions_encodings
