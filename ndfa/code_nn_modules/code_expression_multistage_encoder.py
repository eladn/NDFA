import torch
import torch.nn as nn
from typing import Optional

from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.code_expression_encoder import CodeExpressionEncoder
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors, \
    SubASTInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors


__all__ = ['CodeExpressionMultistageEncoder']


class CodeExpressionMultistageEncoder(nn.Module):
    def __init__(
            self,
            encoder_params: CodeExpressionEncoderParams,
            code_task_vocabs: CodeTaskVocabs,
            identifier_embedding_dim: int,
            nr_layers: int = 1,
            reuse_inner_encodings_from_previous_input_layer: bool = False,
            reuse_inner_encodings_between_layers: bool = False,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionMultistageEncoder, self).__init__()
        self.encoder_params = encoder_params
        assert nr_layers >= 1
        self.encoding_layers = nn.ModuleList([
            CodeExpressionEncoder(
                encoder_params=encoder_params,
                code_task_vocabs=code_task_vocabs,
                identifier_embedding_dim=identifier_embedding_dim,
                is_first_encoder_layer=
                not reuse_inner_encodings_from_previous_input_layer
                if layer_idx == 0 else
                not reuse_inner_encodings_between_layers,
                norm_params=norm_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for layer_idx in range(nr_layers)])

    def forward(
            self,
            previous_code_expression_encodings: CodeExpressionEncodingsTensors,
            tokenized_expressions_input: Optional[CodeExpressionTokensSequenceInputTensors] = None,
            sub_ast_input: Optional[SubASTInputTensors] = None) -> CodeExpressionEncodingsTensors:
        encoded_code_expressions = previous_code_expression_encodings
        for encoding_layer in self.encoding_layers:
            encoded_code_expressions: CodeExpressionEncodingsTensors = encoding_layer(
                previous_code_expression_encodings=encoded_code_expressions,
                tokenized_expressions_input=tokenized_expressions_input,
                sub_ast_input=sub_ast_input
            )
        return encoded_code_expressions
