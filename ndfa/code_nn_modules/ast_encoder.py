__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-04-05"

from typing import Optional

import torch
import torch.nn as nn

from ndfa.code_nn_modules.ast_paths_encoder import ASTPathsEncoder
from ndfa.code_nn_modules.ast_tree_lstm_encoder import ASTTreeLSTMEncoder
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.code_task_input import SubASTInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.nn_utils.modules.gnn_encoder import GNNEncoder
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams


__all__ = ['ASTEncoder']


class ASTEncoder(nn.Module):
    def __init__(
            self, encoder_params: ASTEncoderParams,
            code_task_vocabs: CodeTaskVocabs,
            identifier_embedding_dim: int,
            is_first_encoder_layer: bool = True,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(ASTEncoder, self).__init__()
        self.encoder_params = encoder_params
        self.is_first_encoder_layer = is_first_encoder_layer
        self.identifier_embedding_dim = identifier_embedding_dim

        # TODO: plug-in these params from HPs
        self.primitive_type_embedding_dim = self.encoder_params.ast_node_embedding_dim
        self.modifier_embedding_dim = self.encoder_params.ast_node_embedding_dim

        if self.encoder_params.encoder_type in \
                {ASTEncoderParams.EncoderType.PathsFolded, ASTEncoderParams.EncoderType.SetOfPaths}:
            self.ast_paths_encoder = ASTPathsEncoder(
                ast_node_embedding_dim=self.encoder_params.ast_node_embedding_dim,
                encoder_params=self.encoder_params,
                is_first_encoder_layer=self.is_first_encoder_layer,
                ast_traversal_orientation_vocab=code_task_vocabs.ast_traversal_orientation,
                norm_params=norm_params, dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.encoder_type == ASTEncoderParams.EncoderType.Tree:
            self.ast_treelstm_up = ASTTreeLSTMEncoder(
                ast_node_embedding_dim=self.encoder_params.ast_node_embedding_dim,
                direction='leaves_to_root', norm_params=norm_params, dropout_rate=dropout_rate)
            # self.ast_treelstm_down = ASTTreeLSTMEncoder(
            #     ast_node_embedding_dim=self.encoder_params.ast_node_embedding_dim,
            #     direction='root_to_leaves', dropout_rate=dropout_rate)
        elif self.encoder_params.encoder_type == ASTEncoderParams.EncoderType.GNN:
            self.gnn_encoder = GNNEncoder(
                node_encoding_dim=self.encoder_params.ast_node_embedding_dim,
                encoder_params=self.encoder_params.gnn_encoder,
                norm_params=norm_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        else:
            raise ValueError(f'Unsupported expression encoder type `{self.encoder_params.encoder_type}`.')

    def forward(
            self,
            previous_code_expression_encodings: CodeExpressionEncodingsTensors,
            sub_ast_input: Optional[SubASTInputTensors] = None) -> CodeExpressionEncodingsTensors:
        if self.encoder_params.encoder_type in \
                {ASTEncoderParams.EncoderType.PathsFolded, ASTEncoderParams.EncoderType.SetOfPaths}:
            return self.ast_paths_encoder(
                ast_nodes_encodings=previous_code_expression_encodings.ast_nodes,
                sub_ast_input=sub_ast_input,
                ast_paths_last_states=previous_code_expression_encodings.ast_paths_by_type
                if previous_code_expression_encodings.ast_paths_types ==
                   self.ast_paths_encoder.encoder_params.ast_paths_types else None)
        elif self.encoder_params.encoder_type == ASTEncoderParams.EncoderType.Tree:
            ast_nodes_encodings_up = self.ast_treelstm_up(
                ast_nodes_embeddings=previous_code_expression_encodings.ast_nodes,
                ast_batch=sub_ast_input.dgl_tree)
            # ast_nodes_encodings_down = self.ast_treelstm_down(
            #     ast_nodes_embeddings=previous_code_expression_encodings.ast_nodes,
            #     ast_batch=sub_ast_input.dgl_tree)
            return ast_nodes_encodings_up  # + ast_nodes_encodings_down
        elif self.encoder_params.encoder_type == ASTEncoderParams.EncoderType.GNN:
            return self.gnn_encoder(
                nodes_encodings=previous_code_expression_encodings.ast_nodes,
                graph=sub_ast_input.pyg_graph)
        else:
            assert False
