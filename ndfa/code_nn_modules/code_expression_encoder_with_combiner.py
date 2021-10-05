import torch
import torch.nn as nn
from typing import Optional

from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.code_expression_encoder import CodeExpressionEncoder
from ndfa.code_nn_modules.code_expression_combiner import CodeExpressionCombiner
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors, \
    PDGExpressionsSubASTInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors


__all__ = ['CodeExpressionEncoderWithCombiner']


class CodeExpressionEncoderWithCombiner(nn.Module):
    def __init__(
            self,
            encoder_params: CodeExpressionEncoderParams,
            code_task_vocabs: CodeTaskVocabs,
            identifier_embedding_dim: int,
            is_first_encoder_layer: bool = True,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionEncoderWithCombiner, self).__init__()
        self.encoder = CodeExpressionEncoder(
            encoder_params=encoder_params,
            code_task_vocabs=code_task_vocabs,
            identifier_embedding_dim=identifier_embedding_dim,
            is_first_encoder_layer=is_first_encoder_layer,
            norm_params=norm_params,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.combiner = CodeExpressionCombiner(
            encoder_params=encoder_params,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.combined_code_expressions_norm = None
        if norm_params is not None:
            from ndfa.nn_utils.modules.norm_wrapper import NormWrapper
            self.combined_code_expressions_norm = NormWrapper(
                nr_features=self.encoder_params.expression_encoding_dim, params=norm_params)

    def forward(
            self,
            previous_code_expression_encodings: CodeExpressionEncodingsTensors,
            cfg_nodes_has_expression_mask: torch.Tensor,
            tokenized_expressions_input: Optional[CodeExpressionTokensSequenceInputTensors] = None,
            cfg_nodes_expressions_ast: Optional[PDGExpressionsSubASTInputTensors] = None) \
            -> CodeExpressionEncodingsTensors:
        encoded_code_expressions: CodeExpressionEncodingsTensors = self.encoder(
            previous_code_expression_encodings=previous_code_expression_encodings,
            tokenized_expressions_input=tokenized_expressions_input,
            sub_ast_input=cfg_nodes_expressions_ast)
        combined_expressions = self.code_expression_combiner(
            encoded_code_expressions=encoded_code_expressions,
            tokenized_expressions_input=tokenized_expressions_input,
            cfg_nodes_expressions_ast=cfg_nodes_expressions_ast,
            cfg_nodes_has_expression_mask=cfg_nodes_has_expression_mask)
        if self.combined_code_expressions_norm is not None:
            combined_expressions = self.combined_code_expressions_norm(combined_expressions)
        encoded_code_expressions.combined_expressions = combined_expressions
        return encoded_code_expressions
