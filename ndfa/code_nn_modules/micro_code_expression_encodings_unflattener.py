import torch

from ndfa.nn_utils.model_wrapper.flattened_tensor import FlattenedTensor
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors


__all__ = ['micro_code_expression_encodings_as_unflattenable']


def micro_code_expression_encodings_as_unflattenable(
        micro_encoder_params: CodeExpressionEncoderParams,
        code_task_input: MethodCodeInputTensors,
        code_expression_encodings: CodeExpressionEncodingsTensors) -> FlattenedTensor:
    if micro_encoder_params.encoder_type == 'FlatTokensSeq':
        return code_task_input.pdg.cfg_nodes_tokenized_expressions.batch_flattened_tokens_seqs_as_unflattenable(
            code_expression_encodings.token_seqs)
    elif micro_encoder_params.encoder_type == 'ast':
        if micro_encoder_params.ast_encoder.encoder_type in {'tree', 'paths-folded'}:
            return code_task_input.ast.batch_flattened_ast_nodes_as_unflattenable(
                code_expression_encodings.ast_nodes)
        elif micro_encoder_params.ast_encoder.encoder_type == 'set-of-paths':
            raise NotImplementedError  # TODO: impl!
        else:
            assert False
    else:
        assert False
