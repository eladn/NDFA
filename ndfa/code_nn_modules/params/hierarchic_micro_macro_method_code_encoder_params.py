__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-05"

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

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

    class DecoderFeedingPolicy(Enum):
        MicroItems = 'MicroItems'
        MacroItems = 'MacroItems'

    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'after_macro', {
                HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.DifferentMicro:
                    ['different_local_expression_encoder_after_macro', 'nr_micro_encoding_layers_after_macro'],
                HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.SimilarMicro:
                    ['nr_micro_encoding_layers_after_macro'],
                HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.Pass: []}))

    local_expression_encoder: CodeExpressionEncoderParams = conf_field(
        default_factory=CodeExpressionEncoderParams)
    after_macro: AfterMacro = conf_field(default=AfterMacro.Pass)
    different_local_expression_encoder_after_macro: Optional[CodeExpressionEncoderParams] = conf_field(
        default_factory=CodeExpressionEncoderParams)
    global_context_encoder: MethodCFGMacroEncoderParams = conf_field(
        default_factory=MethodCFGMacroEncoderParams)
    decoder_feeding_policy: DecoderFeedingPolicy = conf_field(
        default=DecoderFeedingPolicy.MicroItems)
    reuse_inner_encodings_between_micro_layers: bool = conf_field(
        default=True)  # TODO: remove this param; just temporary experimental..
    nr_micro_encoding_layers_before_macro: int = conf_field(
        default=1)
    nr_micro_encoding_layers_after_macro: Optional[int] = conf_field(
        default=1)
    nr_layers: int = conf_field(
        default=1)

    def get_descriptive_tags(self) -> Tuple[str, ...]:
        return ('hierarchic', f'nr_micro_after={self.nr_micro_encoding_layers_after_macro}',
                f'nr_micro_before={self.nr_micro_encoding_layers_before_macro}',
                f'nr_micro={self.nr_micro_encoding_layers_after_macro + self.nr_micro_encoding_layers_before_macro}') +\
               tuple(f'micro={tag}' for tag in self.local_expression_encoder.get_descriptive_tags()) +\
               (f'nr_hierarchic_layers={self.nr_layers}',) if self.nr_layers > 1 else () +\
               self.global_context_encoder.get_descriptive_tags()

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
    def last_local_expression_encoder(self) -> Optional[CodeExpressionEncoderParams]:
        if self.after_macro == HierarchicMicroMacroMethodCodeEncoderParams.AfterMacro.Pass:
            return self.local_expression_encoder
        else:
            return self.local_expression_encoder_after_macro

    @property
    def expression_encoding_dim(self) -> int:
        return self.local_expression_encoder.expression_encoding_dim

    @property
    def combined_expression_encoding_dim(self) -> int:
        return self.local_expression_encoder.combined_expression_encoding_dim

    @property
    def macro_encoding_dim(self):
        return self.global_context_encoder.cfg_node_encoding_dim
