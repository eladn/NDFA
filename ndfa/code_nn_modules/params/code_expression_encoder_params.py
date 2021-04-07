from typing import Optional
from dataclasses import dataclass

from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.params.cfg_sub_ast_expression_combiner_params import CFGSubASTExpressionCombinerParams
from ndfa.code_nn_modules.params.code_tokens_seq_encoder_params import CodeTokensSeqEncoderParams
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams
from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField, conf_field


__all__ = ['CodeExpressionEncoderParams']


@dataclass
class CodeExpressionEncoderParams(HasDispatchableField):
    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'encoder_type', {
                'ast': ['ast_encoder', 'cfg_sub_ast_expression_combiner_params'],
                'FlatTokensSeq': ['tokens_seq_encoder', 'tokenized_expression_combiner']}))

    encoder_type: str = conf_field(
        default='ast',
        choices=('FlatTokensSeq', 'ast'),
        description="Representation type of the expression "
                    "(part of the architecture of the code-encoder).")

    # relevant only if `encoder_type == 'ast_paths'`
    ast_encoder: Optional[ASTEncoderParams] = conf_field(
        default_factory=ASTEncoderParams,
        description="Representation type of the AST of the expression "
                    "(part of the architecture of the code-encoder).",
        arg_prefix='ast_encoder')

    tokens_seq_encoder: Optional[CodeTokensSeqEncoderParams] = conf_field(
        default_factory=CodeTokensSeqEncoderParams,
        arg_prefix='tokens_seq_encoder')

    combined_expression_encoding_dim: int = conf_field(
        # default_as_other_field='code_token_encoding_size',
        default=256,
        description="Size of encoded combined code expression.")

    cfg_sub_ast_expression_combiner_params: Optional[CFGSubASTExpressionCombinerParams] = conf_field(
        default_factory=CFGSubASTExpressionCombinerParams)

    tokenized_expression_combiner: Optional[SequenceCombinerParams] = conf_field(
        default_factory=lambda: SequenceCombinerParams(
            method='ends', nr_attn_heads=8, nr_dim_reduction_layers=0))
