from enum import Enum
from typing import Optional
from dataclasses import dataclass

from ndfa.code_nn_modules.params.method_cfg_encoder_params import MethodCFGEncoderParams
from ndfa.code_nn_modules.params.hierarchic_micro_macro_method_code_encoder_params import \
    HierarchicMicroMacroMethodCodeEncoderParams
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_nn_modules.params.identifier_encoder_params import IdentifierEncoderParams
from ndfa.code_nn_modules.params.symbols_encoder_params import SymbolsEncoderParams
from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField, conf_field


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
    nr_method_ast_leaf_to_leaf_paths_to_sample_during_dataloading: Optional[int] = conf_field(
        default=200,
        description="The number of .")
    nr_method_ast_leaf_to_root_paths_to_sample_during_dataloading: Optional[int] = conf_field(
        default=50,
        description="The number of .")
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
