__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-03-17"

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

from ndfa.code_nn_modules.params.method_cfg_encoder_params import MethodCFGEncoderParams
from ndfa.code_nn_modules.params.hierarchic_micro_macro_method_code_encoder_params import \
    HierarchicMicroMacroMethodCodeEncoderParams
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_nn_modules.params.identifier_encoder_params import IdentifierEncoderParams
from ndfa.code_nn_modules.params.symbols_encoder_params import SymbolsEncoderParams
from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField, conf_field
from ndfa.nn_utils.modules.params.sampling_params import SamplingParams, DistributionInfoParams
from ndfa.code_nn_modules.params.code_tokens_seq_encoder_params import CodeTokensSeqEncoderParams


__all__ = ['MethodCodeEncoderParams']


@dataclass
class MethodCodeEncoderParams(HasDispatchableField):
    class EncoderType(Enum):
        WholeMethod = 'WholeMethod'
        Hierarchic = 'Hierarchic'
        MethodCFG = 'MethodCFG'
        MethodCFGV2 = 'MethodCFGV2'

    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'method_encoder_type', {
                cls.EncoderType.WholeMethod: 'whole_method_expression_encoder',
                cls.EncoderType.MethodCFG: 'method_cfg_encoder',
                cls.EncoderType.MethodCFGV2: 'method_cfg_encoder',
                cls.EncoderType.Hierarchic: 'hierarchic_micro_macro_encoder'}))
    method_encoder_type: EncoderType = conf_field(
        default=EncoderType.Hierarchic,
        description="Representation type of the code "
                    "(main architecture of the method-code-encoder).")
    # relevant only if `method_encoder_type in {EncoderType.MethodCFG, EncoderType.MethodCFGV2}`
    method_cfg_encoder: Optional[MethodCFGEncoderParams] = conf_field(
        default_factory=MethodCFGEncoderParams,
        description="Representation type of the method-CFG "
                    "(specific architecture of the method-CFG-code-encoder).",
        arg_prefix='method_cfg_encoder')
    # relevant only if `method_encoder_type == EncoderType.Hierarchic`
    hierarchic_micro_macro_encoder: Optional[HierarchicMicroMacroMethodCodeEncoderParams] = conf_field(
        default_factory=HierarchicMicroMacroMethodCodeEncoderParams,
        description="Hierarchic micro-macro representation of the method.",
        arg_prefix='hierarchic_encoder')
    # relevant only if `method_encoder_type == EncoderType.WholeMethod`
    whole_method_expression_encoder: Optional[CodeExpressionEncoderParams] = conf_field(
        default_factory=lambda: CodeExpressionEncoderParams(
            encoder_type=CodeExpressionEncoderParams.EncoderType.FlatTokensSeq),
        description="Representation type of the whole method code as linear sequence "
                    "(part of the architecture of the code-encoder).",
        arg_prefix='whole_method_expression_encoder')

    def get_descriptive_tags(self) -> Tuple[str, ...]:
        if self.method_encoder_type == MethodCodeEncoderParams.EncoderType.Hierarchic:
            return ('whole-method',) + self.hierarchic_micro_macro_encoder.get_descriptive_tags()
        elif self.method_encoder_type == MethodCodeEncoderParams.EncoderType.WholeMethod:
            return ('whole-method',) + self.whole_method_expression_encoder.get_descriptive_tags()

    def get_flat_tokens_seq_code_encoder_params(self) -> Optional[CodeTokensSeqEncoderParams]:
        if self.method_encoder_type == MethodCodeEncoderParams.EncoderType.WholeMethod:
            if self.whole_method_expression_encoder.encoder_type == \
                    CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
                return self.whole_method_expression_encoder.tokens_seq_encoder
        elif self.method_encoder_type == MethodCodeEncoderParams.EncoderType.Hierarchic:
            if self.hierarchic_micro_macro_encoder.local_expression_encoder.encoder_type == \
                    CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
                return self.hierarchic_micro_macro_encoder.local_expression_encoder.tokens_seq_encoder
        else:
            assert False
        return None

    # preprocess params
    # TODO: put the preprocess params in a dedicated nested confclass.
    max_nr_identifiers: int = conf_field(
        default=110,
        description="The max number of identifiers.")
    min_nr_symbols: int = conf_field(
        default=5,
        description="The max number of .")
    max_nr_symbols: int = conf_field(
        default=55,
        description="The max number of .")
    max_nr_identifier_sub_parts: int = conf_field(
        default=5,
        description="The max number of sub-identifiers in an identifier.")
    min_nr_tokens_method_code: int = conf_field(
        default=50,
        description="The max number of .")
    max_nr_tokens_method_code: int = conf_field(
        default=700,
        description="The max number of .")
    max_nr_method_ast_leaf_to_leaf_paths: Optional[int] = conf_field(
        default=None,
        description="The max number of .")
    max_nr_method_ast_leaf_to_root_paths: Optional[int] = conf_field(
        default=None,
        description="The max number of .")
    nr_method_ast_leaf_to_leaf_paths_to_sample_during_pp: Optional[int] = conf_field(
        default=1000,
        description="The number of .")
    nr_method_ast_leaf_to_root_paths_to_sample_during_pp: Optional[int] = conf_field(
        default=150,
        description="The number of .")
    method_ast_leaf_to_leaf_paths_dataloading_sampling_params: Optional[SamplingParams] = conf_field(
        default_factory=lambda: SamplingParams(
            max_nr_items=600,
            distribution_for_rate_to_sample_by=None,
            sample_in_eval=True))
    method_ast_leaf_to_root_paths_dataloading_sampling_params: Optional[SamplingParams] = conf_field(
        default_factory=lambda: SamplingParams(
            max_nr_items=200,
            distribution_for_rate_to_sample_by=None,
            sample_in_eval=True))
    upper_pruned_ast_leaf_to_leaf_paths_dataloading_sampling_params: Optional[SamplingParams] = conf_field(
        default_factory=lambda: SamplingParams(
            max_nr_items=600,
            distribution_for_rate_to_sample_by=None,
            sample_in_eval=True))
    upper_pruned_ast_leaf_to_root_paths_dataloading_sampling_params: Optional[SamplingParams] = conf_field(
        default_factory=lambda: SamplingParams(
            max_nr_items=200,
            distribution_for_rate_to_sample_by=None,
            sample_in_eval=True))
    sub_asts_leaf_to_leaf_paths_dataloading_sampling_params: Optional[SamplingParams] = conf_field(
        default_factory=lambda: SamplingParams(
            max_nr_items=500,
            distribution_for_rate_to_sample_by=DistributionInfoParams(
                distribution_type=DistributionInfoParams.DistributionType.Normal,
                distribution_params=(0.8, 0.15)),
            min_nr_items_to_sample_by_rate=40,
            sample_in_eval=False))
    sub_asts_leaf_to_root_paths_dataloading_sampling_params: Optional[SamplingParams] = conf_field(
        default_factory=lambda: SamplingParams(
            max_nr_items=200,
            distribution_for_rate_to_sample_by=DistributionInfoParams(
                distribution_type=DistributionInfoParams.DistributionType.Normal,
                distribution_params=(0.8, 0.15)),
            min_nr_items_to_sample_by_rate=10,
            sample_in_eval=False))
    max_nr_cfg_node_sub_ast_leaf_to_leaf_paths: Optional[int] = conf_field(
        default=None,
        description="The max number of .")
    max_nr_cfg_node_sub_ast_leaf_to_root_paths: Optional[int] = conf_field(
        default=None,
        description="The max number of .")
    nr_cfg_node_sub_ast_leaf_to_leaf_paths_to_sample_during_pp: Optional[int] = conf_field(
        default=100,
        description="The number of .")
    nr_cfg_node_sub_ast_leaf_to_root_paths_to_sample_during_pp: Optional[int] = conf_field(
        default=20,
        description="The number of .")
    min_nr_pdg_nodes_with_expression: int = conf_field(
        default=4,
        description="The min number of .")
    min_nr_pdg_nodes: int = conf_field(
        default=6,
        description="The min number of .")
    max_nr_pdg_nodes: int = conf_field(
        default=80,
        description="The max number of .")
    max_nr_tokens_in_pdg_node_expression: int = conf_field(
        default=50,
        description="The max number of .")
    max_nr_pdg_edges: int = conf_field(
        default=300,
        description="The max number of .")
    max_nr_pdg_data_dependency_edges_between_two_nodes: int = conf_field(
        default=6,
        description="The max number of .")
    min_nr_control_flow_paths: int = conf_field(
        default=1,
        description="The max number of .")
    max_nr_control_flow_paths: int = conf_field(
        default=400,
        description="The max number of .")
    nr_control_flow_paths_to_sample_during_pp: Optional[int] = conf_field(
        default=150,
        description="The max number of .")
    min_nr_pdg_paths: int = conf_field(
        default=1,
        description="The max number of .")
    max_nr_pdg_paths: int = conf_field(
        default=300,
        description="The max number of .")
    min_control_flow_path_len: int = conf_field(
        default=3,
        description="The max number of .")
    max_control_flow_path_len: int = conf_field(
        default=80,
        description="The max number of .")
    min_pdg_path_len: int = conf_field(
        default=3,
        description="The max number of .")
    max_pdg_path_len: int = conf_field(
        default=80,
        description="The max number of .")

    max_sub_identifier_vocab_size: int = conf_field(
        default=5000,
        description="The max size of the sub-identifiers vocabulary.")

    max_identifier_vocab_size: int = conf_field(
        default=5000,
        description="The max size of the identifiers vocabulary.")

    identifier_encoder: IdentifierEncoderParams = conf_field(
        default_factory=IdentifierEncoderParams,
        arg_prefix='identifier_encoder')

    symbol_embedding_dim: int = conf_field(
        default=256,
        description="Embedding size for a symbol.")

    symbols_encoder_params: SymbolsEncoderParams = conf_field(
        default=SymbolsEncoderParams)
