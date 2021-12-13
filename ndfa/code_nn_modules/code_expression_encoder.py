__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-11-29"

from typing import Optional

import torch
import torch.nn as nn

from .code_expression_tokens_sequence_encoder import CodeExpressionTokensSequenceEncoder
from ndfa.code_nn_modules.ast_encoder import ASTEncoder
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors, \
    SubASTInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams


__all__ = ['CodeExpressionEncoder']


class CodeExpressionEncoder(nn.Module):
    def __init__(
            self,
            encoder_params: CodeExpressionEncoderParams,
            code_task_vocabs: CodeTaskVocabs,
            identifier_embedding_dim: int,
            is_first_encoder_layer: bool = True,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionEncoder, self).__init__()
        self.encoder_params = encoder_params
        self.is_first_encoder_layer = is_first_encoder_layer
        self.identifier_embedding_dim = identifier_embedding_dim
        if self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
            self.code_expression_linear_seq_encoder = CodeExpressionTokensSequenceEncoder(
                encoder_params=self.encoder_params.tokens_seq_encoder,
                tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
                norm_params=norm_params,
                dropout_rate=dropout_rate,
                activation_fn=activation_fn)
        elif self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.AST:
            self.ast_encoder = ASTEncoder(
                encoder_params=self.encoder_params.ast_encoder,
                code_task_vocabs=code_task_vocabs,
                identifier_embedding_dim=identifier_embedding_dim,
                is_first_encoder_layer=is_first_encoder_layer,
                norm_params=norm_params,
                dropout_rate=dropout_rate,
                activation_fn=activation_fn)
        else:
            raise ValueError(f'Unsupported expression encoder type `{self.encoder_params.encoder_type}`.')

    def forward(
            self,
            previous_code_expression_encodings: CodeExpressionEncodingsTensors,
            tokenized_expressions_input: Optional[CodeExpressionTokensSequenceInputTensors] = None,
            sub_ast_input: Optional[SubASTInputTensors] = None) -> CodeExpressionEncodingsTensors:
        if self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
            return self.code_expression_linear_seq_encoder(
                token_seqs_embeddings=previous_code_expression_encodings.token_seqs,
                expressions_input=tokenized_expressions_input)
        elif self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.AST:
            return self.ast_encoder(
                previous_code_expression_encodings=previous_code_expression_encodings,
                sub_ast_input=sub_ast_input)
        else:
            assert False
