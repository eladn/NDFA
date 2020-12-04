import torch
import torch.nn as nn
from typing import Optional

from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.code_nn_modules.code_task_input import MethodASTInputTensors
from ndfa.nn_utils.misc.misc import apply_disjoint_updates_to_encodings_tensor, EncodingsUpdater


__all__ = ['ASTNodesEmbedder']


class ASTNodesEmbedder(nn.Module):
    def __init__(
            self,
            ast_node_embedding_dim: int,
            identifier_encoding_dim: int,
            primitive_type_embedding_dim: int,
            modifier_embedding_dim: int,
            ast_node_type_vocab: Optional[Vocabulary] = None,
            primitive_types_vocab: Optional[Vocabulary] = None,
            modifiers_vocab: Optional[Vocabulary] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(ASTNodesEmbedder, self).__init__()
        self.ast_node_embedding_dim = ast_node_embedding_dim
        self.ast_node_type_vocab = ast_node_type_vocab
        self.primitive_types_vocab = primitive_types_vocab
        self.modifiers_vocab = modifiers_vocab
        self.ast_node_type_embedding_layer = nn.Embedding(
            num_embeddings=len(self.ast_node_type_vocab),
            embedding_dim=self.ast_node_embedding_dim,
            padding_idx=self.ast_node_type_vocab.get_word_idx('<PAD>'))
        self.primitive_type_embedding_dim = primitive_type_embedding_dim
        self.primitive_types_embedding_layer = nn.Embedding(
            num_embeddings=len(self.primitive_types_vocab),
            embedding_dim=self.primitive_type_embedding_dim,
            padding_idx=self.primitive_types_vocab.get_word_idx('<PAD>'))
        self.modifier_embedding_dim = modifier_embedding_dim
        self.modifiers_embedding_layer = nn.Embedding(
            num_embeddings=len(self.modifiers_vocab),
            embedding_dim=self.modifier_embedding_dim,
            padding_idx=self.modifiers_vocab.get_word_idx('<PAD>'))
        self.identifier_encoding_dim = identifier_encoding_dim
        self.identifier_leaf_linear_projection_layer = nn.Linear(
            in_features=self.identifier_encoding_dim + self.ast_node_embedding_dim, out_features=self.ast_node_embedding_dim)
        self.primitive_type_leaf_linear_projection_layer = nn.Linear(
            in_features=self.primitive_type_embedding_dim + self.ast_node_embedding_dim, out_features=self.ast_node_embedding_dim)
        self.modifier_leaf_linear_projection_layer = nn.Linear(
            in_features=self.modifier_embedding_dim + self.ast_node_embedding_dim, out_features=self.ast_node_embedding_dim)

        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(
            self,
            method_ast_input: MethodASTInputTensors,
            identifiers_encodings: torch.Tensor) -> torch.Tensor:
        ast_nodes_types_embeddings = self.ast_node_type_embedding_layer(method_ast_input.ast_node_types.tensor)

        ast_nodes_with_identifier_leaf_identifier_embeddings =\
            identifiers_encodings[method_ast_input.ast_nodes_with_identifier_leaf_identifier_idx.indices]
        ast_nodes_with_primitive_type_leaf_primitive_type_embeddings = self.primitive_types_embedding_layer(
            method_ast_input.ast_nodes_with_primitive_type_leaf_primitive_type.tensor)
        ast_nodes_with_modifier_leaf_modifier_embeddings = self.modifiers_embedding_layer(
            method_ast_input.ast_nodes_with_modifier_leaf_modifier.tensor)

        ast_nodes_embeddings = apply_disjoint_updates_to_encodings_tensor(
            ast_nodes_types_embeddings,
            EncodingsUpdater(
                new_embeddings=ast_nodes_with_identifier_leaf_identifier_embeddings,
                element_indices=method_ast_input.ast_nodes_with_identifier_leaf_nodes_indices.indices,
                linear_projection_layer=self.identifier_leaf_linear_projection_layer),
            EncodingsUpdater(
                new_embeddings=ast_nodes_with_primitive_type_leaf_primitive_type_embeddings,
                element_indices=method_ast_input.ast_nodes_with_primitive_type_leaf_nodes_indices.indices,
                linear_projection_layer=self.primitive_type_leaf_linear_projection_layer),
            EncodingsUpdater(
                new_embeddings=ast_nodes_with_modifier_leaf_modifier_embeddings,
                element_indices=method_ast_input.ast_nodes_with_modifier_leaf_nodes_indices.indices,
                linear_projection_layer=self.modifier_leaf_linear_projection_layer))
        ast_nodes_embeddings = self.dropout_layer(ast_nodes_embeddings)
        return ast_nodes_embeddings
