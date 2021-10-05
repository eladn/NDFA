from dataclasses import dataclass
from typing import Optional
from enum import Enum

from ndfa.code_nn_modules.params.method_cfg_macro_encoder_params import MethodCFGMacroEncoderParams
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField, conf_field


__all__ = ['HierarchicMicroMacroMethodCodeEncoderParams']


@dataclass
class HierarchicMicroMacroMethodCodeEncoderParams(HasDispatchableField):
    class AfterMacro(Enum):
        SimilarMicro = 'SimilarMicro'
        DifferentMicro = 'DifferentMicro'
        Pass = 'Pass'

        @property
        def requires_micro(self):
            return self in {self.SimilarMicro, self.DifferentMicro}

    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'after_macro', {
                HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.DifferentMicro:
                    ['different_local_expression_encoder_after_macro'],
                HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.SimilarMicro: [],
                HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.Pass: []}))

    local_expression_encoder: CodeExpressionEncoderParams = conf_field(
        default_factory=CodeExpressionEncoderParams)
    after_macro: AfterMacro = conf_field(default=AfterMacro.Pass)
    different_local_expression_encoder_after_macro: Optional[CodeExpressionEncoderParams] = conf_field(
        default_factory=CodeExpressionEncoderParams)
    global_context_encoder: MethodCFGMacroEncoderParams = conf_field(
        default_factory=MethodCFGMacroEncoderParams)

    @property
    def local_expression_encoder_after_macro(self) -> Optional[CodeExpressionEncoderParams]:
        if self.after_macro == HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.Pass:
            return None
        elif self.after_macro == HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.DifferentMicro:
            return self.different_local_expression_encoder_after_macro
        elif self.after_macro == HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.SimilarMicro:
            return self.local_expression_encoder
        assert False

    @property
    def expression_encoding_dim(self) -> int:
        return self.local_expression_encoder.expression_encoding_dim

    @property
    def combined_expression_encoding_dim(self) -> int:
        return self.local_expression_encoder.combined_expression_encoding_dim

    @property
    def macro_encoding_dim(self):
        return self.global_context_encoder.cfg_node_encoding_dim
