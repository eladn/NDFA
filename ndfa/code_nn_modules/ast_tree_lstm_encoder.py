import torch
import torch.nn as nn
import dgl

from ndfa.nn_utils.modules.dgl_tree_lstm import TreeLSTM


__all__ = ['ASTTreeLSTMEncoder']


class ASTTreeLSTMEncoder(nn.Module):
    def __init__(self, ast_node_embedding_dim: int, dropout_rate: float = 0.3):
        super(ASTTreeLSTMEncoder, self).__init__()
        self.ast_node_embedding_dim = ast_node_embedding_dim
        self.tree_lstm = TreeLSTM(
            node_embedding_size=ast_node_embedding_dim, cell_type='sum_children', dropout_rate=dropout_rate)

    def forward(self, ast_nodes_embeddings: torch.Tensor, ast_batch: dgl.DGLGraph):
        assert ast_nodes_embeddings.ndim == 2
        assert ast_nodes_embeddings.size(1) == self.ast_node_embedding_dim
        h = torch.zeros_like(ast_nodes_embeddings)
        c = torch.zeros_like(ast_nodes_embeddings)
        new_node_encodings = self.tree_lstm(
            nodes_embeddings=ast_nodes_embeddings,
            tree=ast_batch,
            h=h, c=c)
        assert new_node_encodings.shape == ast_nodes_embeddings.shape
        return new_node_encodings
