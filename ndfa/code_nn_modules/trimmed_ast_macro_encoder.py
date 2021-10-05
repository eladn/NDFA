import torch
import torch.nn as nn
from typing import Optional

from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors, MethodASTInputTensors
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.nn_utils.modules.state_updater import StateUpdater
from ndfa.nn_utils.modules.params.state_updater_params import StateUpdaterParams
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.ast_encoder import ASTEncoder
from ndfa.code_nn_modules.ast_nodes_embedder import ASTNodesEmbedder
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.nn_utils.modules.norm_wrapper import NormWrapper


__all__ = ['TrimmedASTMacroEncoder']


class TrimmedASTMacroEncoder(nn.Module):
    def __init__(
            self, code_task_vocabs: CodeTaskVocabs,
            cfg_node_encoding_dim: int,
            identifier_embedding_dim: int,
            macro_trimmed_ast_encoder_params: ASTEncoderParams,
            post_macro_encoder_state_updater_params: StateUpdaterParams,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(TrimmedASTMacroEncoder, self).__init__()
        self.trimmed_ast_encoder = ASTEncoder(
            encoder_params=macro_trimmed_ast_encoder_params,
            code_task_vocabs=code_task_vocabs,
            identifier_embedding_dim=identifier_embedding_dim,
            is_first_encoder_layer=True,
            norm_params=norm_params,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        ast_node_embedding_dim = macro_trimmed_ast_encoder_params.ast_node_embedding_dim
        self.ast_nodes_embedder = ASTNodesEmbedder(
            ast_node_embedding_dim=ast_node_embedding_dim,
            identifier_encoding_dim=identifier_embedding_dim,
            primitive_type_embedding_dim=ast_node_embedding_dim,  # TODO: move it to HP
            modifier_embedding_dim=ast_node_embedding_dim,  # TODO: move it to HP
            ast_node_type_vocab=code_task_vocabs.ast_node_types,
            ast_node_major_type_vocab=code_task_vocabs.ast_node_major_types,
            ast_node_minor_type_vocab=code_task_vocabs.ast_node_minor_types,
            ast_node_nr_children_vocab=code_task_vocabs.ast_node_nr_children,
            ast_node_child_pos_vocab=code_task_vocabs.ast_node_child_pos,
            primitive_types_vocab=code_task_vocabs.primitive_types,
            modifiers_vocab=code_task_vocabs.modifiers,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.post_macro_encoder_state_updater = StateUpdater(
            state_dim=cfg_node_encoding_dim,
            params=post_macro_encoder_state_updater_params,
            update_dim=cfg_node_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.norm = None if norm_params is None else NormWrapper(
            nr_features=cfg_node_encoding_dim, params=norm_params)

    def forward(
            self, code_task_input: MethodCodeInputTensors,
            encoded_cfg_nodes: torch.Tensor,
            method_ast_input: MethodASTInputTensors,
            identifiers_encodings: torch.Tensor) \
            -> torch.Tensor:
        # We take the AST nodes embeddings of the upper part of the tree.
        macro_trimmed_ast_nodes_encodings = self.ast_nodes_embedder(
            method_ast_input=method_ast_input,
            identifiers_encodings=identifiers_encodings)
        # We take the CFG nodes encodings for the roots of the relevant sub-asts!
        macro_trimmed_ast_nodes_encodings[
            code_task_input.pdg.cfg_nodes_expressions_ast.pdg_node_idx_to_sub_ast_root_idx_mapping_value.indices] = \
            encoded_cfg_nodes[
                code_task_input.pdg.cfg_nodes_expressions_ast.pdg_node_idx_to_sub_ast_root_idx_mapping_key.indices]
        encoded_ast_for_macro_trimmed_ast_encoder = CodeExpressionEncodingsTensors(
            ast_nodes=macro_trimmed_ast_nodes_encodings)
        encoded_macro_trimmed_ast: CodeExpressionEncodingsTensors = self.trimmed_ast_encoder(
            previous_code_expression_encodings=encoded_ast_for_macro_trimmed_ast_encoder,
            sub_ast_input=code_task_input.pdg.cfg_macro_trimmed_ast)
        # Replace the encodings of the CFG nodes (these that have expressions) to be the new encoding
        # of the root of their sub-ast.
        new_cfg_sub_asts_roots_encodings = encoded_macro_trimmed_ast.ast_nodes[
            code_task_input.pdg.cfg_nodes_expressions_ast.pdg_node_idx_to_sub_ast_root_idx_mapping_value.indices]
        scatter_index = code_task_input.pdg.cfg_nodes_expressions_ast. \
            pdg_node_idx_to_sub_ast_root_idx_mapping_key.indices. \
            unsqueeze(-1).expand(new_cfg_sub_asts_roots_encodings.shape)
        new_encoded_cfg_nodes = torch.scatter(
            input=encoded_cfg_nodes, dim=0, index=scatter_index,
            src=new_cfg_sub_asts_roots_encodings)
        encoded_cfg_nodes = self.post_macro_encoder_state_updater(
            previous_state=encoded_cfg_nodes, state_update=new_encoded_cfg_nodes)
        if self.norm:
            encoded_cfg_nodes = self.norm(encoded_cfg_nodes)
        return encoded_cfg_nodes
