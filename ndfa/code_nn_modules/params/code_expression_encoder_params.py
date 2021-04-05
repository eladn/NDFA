from confclass import confparam
from dataclasses import dataclass

from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.params.cfg_sub_ast_expression_combiner_params import CFGSubASTExpressionCombinerParams
from ndfa.code_nn_modules.params.code_tokens_seq_encoder_params import CodeTokensSeqEncoderParams
from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField


__all__ = ['CodeExpressionEncoderParams']


@dataclass
class CodeExpressionEncoderParams(HasDispatchableField):
    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'encoder_type', {
                'ast': 'ast_encoder',
                'tokens-seq': 'tokens_seq_encoder'}))

    encoder_type: str = confparam(
        default='ast',
        choices=('tokens-seq', 'ast'),
        description="Representation type of the expression "
                    "(part of the architecture of the code-encoder).")

    # relevant only if `encoder_type == 'ast_paths'`
    ast_encoder: ASTEncoderParams = confparam(
        default_factory=ASTEncoderParams,
        description="Representation type of the AST of the expression "
                    "(part of the architecture of the code-encoder).",
        arg_prefix='ast_encoder')

    tokens_seq_encoder: CodeTokensSeqEncoderParams = confparam(
        default_factory=CodeTokensSeqEncoderParams,
        arg_prefix='tokens_seq_encoder')

    combined_expression_encoding_dim: int = confparam(
        # default_as_other_field='code_token_encoding_size',
        default=256,
        description="Size of encoded combined code expression.")

    cfg_sub_ast_expression_combiner_params: CFGSubASTExpressionCombinerParams = confparam(
        default=CFGSubASTExpressionCombinerParams)
