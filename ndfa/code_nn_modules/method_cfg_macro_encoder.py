import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.params.method_cfg_macro_encoder_params import MethodCFGMacroEncoderParams
from ndfa.code_nn_modules.trimmed_ast_macro_encoder import TrimmedASTMacroEncoder
from ndfa.code_nn_modules.cfg_paths_macro_encoder import CFGPathsMacroEncoder, CFGPathsMacroEncodings
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_nn_modules.cfg_node_encoder import CFGNodeEncoder
from ndfa.code_nn_modules.cfg_gnn_encoder import CFGGNNEncoder
from ndfa.code_nn_modules.cfg_single_path_macro_encoder import CFGSinglePathMacroEncoder
from ndfa.nn_utils.model_wrapper.flattened_tensor import FlattenedTensor


__all__ = ['MethodCFGMacroEncoder', 'MethodCFGMacroEncodings']


@dataclass
class MethodCFGMacroEncodings:
    macro_encodings: FlattenedTensor
    cfg_nodes_encodings: Optional[torch.Tensor] = None


class MethodCFGMacroEncoder(nn.Module):
    def __init__(
            self,
            params: MethodCFGMacroEncoderParams,
            combined_micro_expression_dim: int,
            identifier_embedding_dim: int,
            code_task_vocabs: CodeTaskVocabs,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(MethodCFGMacroEncoder, self).__init__()
        self.params = params

        self.cfg_node_encoder = CFGNodeEncoder(
            cfg_node_dim=self.params.cfg_node_encoding_dim,
            cfg_combined_expression_dim=combined_micro_expression_dim,
            pdg_node_control_kinds_vocab=code_task_vocabs.pdg_node_control_kinds,
            pdg_node_control_kinds_embedding_dim=self.params.cfg_node_control_kinds_embedding_dim,
            norm_params=norm_params, dropout_rate=dropout_rate, activation_fn=activation_fn)

        if self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.CFGPaths:
            self.cfg_paths_encoder = CFGPathsMacroEncoder(
                params=self.params.paths_encoder,
                cfg_node_dim=self.params.cfg_node_encoding_dim,
                cfg_nodes_encodings_state_updater=self.params.post_macro_cfg_nodes_encodings_state_updater,
                control_flow_edge_types_vocab=code_task_vocabs.pdg_control_flow_edge_types,
                norm_params=norm_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.SetOfCFGNodes:
            pass  # We actually do not need to do anything in this case.
            raise NotImplementedError  # what we want to do in this case?
        elif self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.FlatCFGNodesAppearanceSeq:
            # TODO: use `self.params.post_macro_cfg_nodes_encodings_state_updater`
            self.cfg_single_path_encoder = CFGSinglePathMacroEncoder(
                cfg_node_dim=self.params.cfg_node_encoding_dim,
                params=self.params.single_path_encoder,
                norm_params=norm_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.CFGGNN:
            # TODO: use `self.params.post_macro_cfg_nodes_encodings_state_updater`
            self.cfg_gnn_encoder = CFGGNNEncoder(
                cfg_node_encoding_dim=self.params.cfg_node_encoding_dim,
                encoder_params=self.params.gnn_encoder,
                pdg_control_flow_edge_types_vocab=code_task_vocabs.pdg_control_flow_edge_types,
                norm_params=norm_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.UpperASTPaths:
            self.trimmed_ast_macro_encoder = TrimmedASTMacroEncoder(
                code_task_vocabs=code_task_vocabs,
                cfg_node_encoding_dim=self.params.cfg_node_encoding_dim,
                identifier_embedding_dim=identifier_embedding_dim,
                macro_trimmed_ast_encoder_params=self.params.macro_trimmed_ast_encoder,
                post_macro_encoder_state_updater_params=self.params.post_macro_cfg_nodes_encodings_state_updater,
                norm_params=norm_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        else:
            raise ValueError(f'Unsupported encoder type `{self.params.encoder_type}`.')

    def forward(
            self,
            code_task_input: MethodCodeInputTensors,
            encoded_identifiers: torch.Tensor,
            encoded_combined_code_expressions: torch.Tensor) -> MethodCFGMacroEncodings:
        encoded_cfg_nodes = self.cfg_node_encoder(
            combined_cfg_expressions_encodings=encoded_combined_code_expressions,
            pdg=code_task_input.pdg)

        macro_encodings = None
        if self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.CFGPaths:
            # note: norm inside `CFGPathsMacroEncoder`
            cfg_paths_macro_encodings: CFGPathsMacroEncodings = self.cfg_paths_encoder(
                cfg_nodes_encodings=encoded_cfg_nodes,
                pdg_input=code_task_input.pdg)
            # TODO: Could it be a possible case that we do fold but we want to pass the combined paths
            #  as the macro encodings?
            if cfg_paths_macro_encodings.folded_nodes_encodings is not None:
                encoded_cfg_nodes = cfg_paths_macro_encodings.folded_nodes_encodings
            else:
                # TODO: check the un-flattenning of `combined_paths`.
                #  The desired shape: (batch_size, max_nr_paths_in_example, embd)
                macro_encodings = cfg_paths_macro_encodings.combined_paths
                from warnings import warn
                warn('The un-flattening of the combined paths is not checked!')
        elif self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.SetOfCFGNodes:
            pass  # We actually do not need to do anything in this case.
            raise NotImplementedError  # what we want to do in this case?
        elif self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.FlatCFGNodesAppearanceSeq:
            # note: norm inside `CFGSinglePathMacroEncoder`
            encoded_cfg_nodes = self.cfg_single_path_encoder(
                pdg_input=code_task_input.pdg,
                cfg_nodes_encodings=encoded_cfg_nodes)
        elif self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.CFGGNN:
            # note: norm inside `CFGGNNEncoder`
            encoded_cfg_nodes = self.cfg_gnn_encoder(
                encoded_cfg_nodes=encoded_cfg_nodes,
                cfg_control_flow_graph=code_task_input.pdg.cfg_control_flow_graph)
        elif self.params.encoder_type == MethodCFGMacroEncoderParams.EncoderType.UpperASTPaths:
            # note: norm inside `TrimmedASTMacroEncoder`
            encoded_cfg_nodes = self.trimmed_ast_macro_encoder(
                code_task_input=code_task_input,
                encoded_cfg_nodes=encoded_cfg_nodes,
                method_ast_input=code_task_input.ast,
                identifiers_encodings=encoded_identifiers)
        else:
            assert False

        if macro_encodings is None:
            macro_encodings = FlattenedTensor(
                flattened=encoded_cfg_nodes,
                unflattener_mask=code_task_input.pdg.get_cfg_nodes_encodings_unflattener_mask(),
                unflattener=code_task_input.pdg.unflatten_cfg_nodes_encodings)

        return MethodCFGMacroEncodings(
            macro_encodings=macro_encodings,
            cfg_nodes_encodings=encoded_cfg_nodes)
