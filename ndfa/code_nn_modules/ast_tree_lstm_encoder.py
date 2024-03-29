import torch
import torch.nn as nn
import dgl
from typing import Optional

from ndfa.nn_utils.modules.dgl_tree_lstm import TreeLSTM
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.nn_utils.modules.norm_wrapper import NormWrapper


__all__ = ['ASTTreeLSTMEncoder']


class ASTTreeLSTMEncoder(nn.Module):
    def __init__(
            self,
            ast_node_embedding_dim: int,
            direction: str = 'root_to_leaves',
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3):
        super(ASTTreeLSTMEncoder, self).__init__()
        self.ast_node_embedding_dim = ast_node_embedding_dim
        assert direction in {'root_to_leaves', 'leaves_to_root'}
        self.direction = direction
        self.tree_lstm = TreeLSTM(
            node_embedding_size=ast_node_embedding_dim,
            cell_type='sum_children',
            dropout_rate=dropout_rate)
        self.norm = None if norm_params is None else NormWrapper(
            nr_features=ast_node_embedding_dim, params=norm_params)

    def forward(self, ast_nodes_embeddings: torch.Tensor, ast_batch: dgl.DGLGraph) -> CodeExpressionEncodingsTensors:
        assert ast_nodes_embeddings.ndim == 2
        assert ast_nodes_embeddings.size(1) == self.ast_node_embedding_dim
        h = torch.zeros_like(ast_nodes_embeddings)
        c = torch.zeros_like(ast_nodes_embeddings)
        new_node_encodings = self.tree_lstm(
            nodes_embeddings=ast_nodes_embeddings,
            tree=ast_batch, h=h, c=c,
            direction=self.direction)
        assert new_node_encodings.shape == ast_nodes_embeddings.shape
        if self.norm:
            new_node_encodings = self.norm(new_node_encodings)
        return CodeExpressionEncodingsTensors(ast_nodes=new_node_encodings)
