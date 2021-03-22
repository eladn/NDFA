from confclass import confparam
from dataclasses import dataclass

from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.params.cfg_sub_ast_expression_combiner_params import CFGSubASTExpressionCombinerParams


__all__ = ['CodeExpressionEncoderParams']


@dataclass
class CodeExpressionEncoderParams:
    encoder_type: str = confparam(
        default='ast_paths',
        choices=('tokens-seq', 'ast_paths', 'ast_treelstm', 'symbols-occurrences-seq'),
        description="Representation type of the expression "
                    "(part of the architecture of the code-encoder).")

    # relevant only if `encoder_type == 'ast_paths'`
    ast_encoder: ASTEncoderParams = confparam(
        default_factory=ASTEncoderParams,
        description="Representation type of the AST of the expression "
                    "(part of the architecture of the code-encoder).",
        arg_prefix='ast_encoder')

    token_type_embedding_dim: int = confparam(
        default=64,
        description="Embedding size for code token type (operator, identifier, etc).")

    kos_token_embedding_dim: int = confparam(
        default=256,
        description="Embedding size for code keyword/operator/separator token.")

    token_encoding_dim: int = confparam(
        default=256,
        # default_factory_with_self_access=lambda _self:
        # _self.identifier_embedding_size + _self.code_token_type_embedding_size,
        # default_description="identifier_embedding_size + code_token_type_embedding_size",
        description="Size of encoded code token vector.")

    combined_expression_encoding_dim: int = confparam(
        # default_as_other_field='code_token_encoding_size',
        default=256,
        description="Size of encoded combined code expression.")

    sequence_encoder: SequenceEncoderParams = confparam(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence-encoder')

    shuffle_expressions: bool = confparam(
        default=False)

    cfg_sub_ast_expression_combiner_params: CFGSubASTExpressionCombinerParams = confparam(
        default=CFGSubASTExpressionCombinerParams)
