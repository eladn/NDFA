__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-03-17"

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.params.cfg_sub_ast_expression_combiner_params import CFGSubASTExpressionCombinerParams
from ndfa.code_nn_modules.params.code_tokens_seq_encoder_params import CodeTokensSeqEncoderParams
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams
from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField, conf_field


__all__ = ['CodeExpressionEncoderParams']


@dataclass
class CodeExpressionEncoderParams(HasDispatchableField):
    class EncoderType(Enum):
        FlatTokensSeq = 'FlatTokensSeq'
        AST = 'AST'

    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'encoder_type', {
                cls.EncoderType.AST: ['ast_encoder', 'cfg_sub_ast_expression_combiner_params'],
                cls.EncoderType.FlatTokensSeq: ['tokens_seq_encoder', 'tokenized_expression_combiner']}))

    def get_descriptive_tags(self) -> Tuple[str, ...]:
        if self.encoder_type == CodeExpressionEncoderParams.EncoderType.AST:
            return self.ast_encoder.get_descriptive_tags()
        elif self.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
            return ('FlatTokensSeq',)

    encoder_type: EncoderType = conf_field(
        default=EncoderType.AST,
        description="Representation type of the expression "
                    "(part of the architecture of the code-encoder).")

    # relevant only if `encoder_type == 'AST'`
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

    @property
    def expression_encoding_dim(self) -> int:
        return self.tokens_seq_encoder.token_encoding_dim \
            if self.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq else \
            self.ast_encoder.ast_node_embedding_dim
