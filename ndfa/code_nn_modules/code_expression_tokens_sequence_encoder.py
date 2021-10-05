import torch
import torch.nn as nn
from typing import Optional

from ndfa.code_nn_modules.params.code_tokens_seq_encoder_params import CodeTokensSeqEncoderParams
from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams


__all__ = ['CodeExpressionTokensSequenceEncoder']


class CodeExpressionTokensSequenceEncoder(nn.Module):
    def __init__(
            self,
            encoder_params: CodeTokensSeqEncoderParams,
            norm_params: Optional[NormWrapperParams] = None,  # TODO: use it!
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionTokensSequenceEncoder, self).__init__()
        self.encoder_params = encoder_params
        self.sequence_encoder = SequenceEncoder(
            encoder_params=self.encoder_params.sequence_encoder,
            input_dim=self.encoder_params.token_encoding_dim,
            hidden_dim=self.encoder_params.token_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.norm_params = norm_params  # TODO: use it!
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self,
                token_seqs_embeddings: torch.Tensor,
                expressions_input: CodeExpressionTokensSequenceInputTensors) \
            -> CodeExpressionEncodingsTensors:
        if self.encoder_params.shuffle_expressions:
            token_seqs_embeddings = expressions_input.sequence_shuffler.shuffle(token_seqs_embeddings)

        expressions_encodings = self.sequence_encoder(
            sequence_input=token_seqs_embeddings,
            lengths=expressions_input.token_type.sequences_lengths,
            batch_first=True).sequence

        if self.encoder_params.shuffle_expressions:
            expressions_encodings = expressions_input.sequence_shuffler.unshuffle(expressions_encodings)

        return CodeExpressionEncodingsTensors(token_seqs=expressions_encodings)
