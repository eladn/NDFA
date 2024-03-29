__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-06"

import torch

from ndfa.nn_utils.model_wrapper.flattened_tensor import FlattenedTensor
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors


__all__ = ['micro_code_expression_encodings_as_unflattenable']


def micro_code_expression_encodings_as_unflattenable(
        micro_encoder_params: CodeExpressionEncoderParams,
        code_task_input: MethodCodeInputTensors,
        code_expression_encodings: CodeExpressionEncodingsTensors) -> FlattenedTensor:
    if micro_encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
        return code_task_input.pdg.cfg_nodes_tokenized_expressions.batch_flattened_tokens_seqs_as_unflattenable(
            code_expression_encodings.token_seqs)
    elif micro_encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.AST:
        if micro_encoder_params.ast_encoder.encoder_type in \
                {ASTEncoderParams.EncoderType.Tree,
                 ASTEncoderParams.EncoderType.PathsFolded,
                 ASTEncoderParams.EncoderType.GNN}:
            return code_task_input.ast.batch_flattened_ast_nodes_as_unflattenable(
                code_expression_encodings.ast_nodes)
        elif micro_encoder_params.ast_encoder.encoder_type == ASTEncoderParams.EncoderType.SetOfPaths:
            return code_task_input.pdg.cfg_nodes_expressions_ast.batch_flattened_combined_ast_paths_as_unflattenable(
                {path_type: path_encodings.combined
                 for path_type, path_encodings in code_expression_encodings.ast_paths_by_type.items()})
        else:
            assert False
    else:
        assert False
