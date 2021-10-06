from enum import Enum
from typing import Optional
from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field, DispatchField, HasDispatchableField
from ndfa.nn_utils.modules.params.state_updater_params import StateUpdaterParams
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.params.cfg_gnn_encoder_params import CFGGNNEncoderParams
from ndfa.code_nn_modules.params.cfg_paths_macro_encoder_params import CFGPathsMacroEncoderParams
from ndfa.code_nn_modules.params.cfg_single_path_macro_encoder_params import CFGSinglePathMacroEncoderParams


__all__ = ['MethodCFGMacroEncoderParams']


@dataclass
class MethodCFGMacroEncoderParams(HasDispatchableField):
    class EncoderType(Enum):
        NoMacro = 'NoMacro'
        SetOfCFGNodes = 'SetOfCFGNodes'
        FlatCFGNodesAppearanceSeq = 'FlatCFGNodesAppearanceSeq'
        CFGGNN = 'CFGGNN'
        UpperASTPaths = 'UpperASTPaths'
        CFGPaths = 'CFGPaths'

    encoder_type: EncoderType = conf_field(
        default=EncoderType.CFGPaths,
        description="Macro operator of the hierarchic method representation.")

    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'encoder_type', {
                cls.EncoderType.CFGPaths: ['paths_encoder'],
                cls.EncoderType.FlatCFGNodesAppearanceSeq: ['single_path_encoder'],
                cls.EncoderType.CFGGNN: ['gnn_encoder'],
                cls.EncoderType.SetOfCFGNodes: [],
                cls.EncoderType.NoMacro: [],
                cls.EncoderType.UpperASTPaths: ['macro_trimmed_ast_encoder']
            }))

    post_macro_cfg_nodes_encodings_state_updater: StateUpdaterParams = conf_field(
        default_factory=StateUpdaterParams)

    macro_context_to_micro_state_updater: StateUpdaterParams = conf_field(
        default_factory=StateUpdaterParams)

    cfg_node_control_kinds_embedding_dim: int = conf_field(
        default=64,
        description="Embedding size for the CFG node control kind.")

    cfg_node_encoding_dim: int = conf_field(
        default=256,
        description="Size of encoded CFG node vector.")

    gnn_encoder: Optional[CFGGNNEncoderParams] = conf_field(
        default_factory=CFGGNNEncoderParams,
        arg_prefix='gnn')

    paths_encoder: Optional[CFGPathsMacroEncoderParams] = conf_field(
        default_factory=CFGPathsMacroEncoderParams,
        arg_prefix='paths')

    single_path_encoder: Optional[CFGSinglePathMacroEncoderParams] = conf_field(
        default_factory=CFGSinglePathMacroEncoderParams,
        arg_prefix='single_path')

    macro_trimmed_ast_encoder: Optional[ASTEncoderParams] = conf_field(
        default_factory=ASTEncoderParams)
