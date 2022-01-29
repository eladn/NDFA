__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-12-04"

import torch
import torch.nn as nn
from typing import Literal

from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.code_nn_modules.code_task_input import MethodASTInputTensors
from ndfa.nn_utils.misc.misc import apply_disjoint_updates_to_encodings_tensor, EncodingsUpdater, \
    apply_disjoint_additions_to_encodings_tensor


__all__ = ['ASTNodesEmbedder']


class ASTNodesEmbedder(nn.Module):
    def __init__(
            self,
            ast_node_embedding_dim: int,
            identifier_encoding_dim: int,
            primitive_type_embedding_dim: int,
            modifier_embedding_dim: int,
            ast_node_type_vocab: Vocabulary,
            ast_node_major_type_vocab: Vocabulary,
            ast_node_minor_type_vocab: Vocabulary,
            ast_node_nr_children_vocab: Vocabulary,
            ast_node_child_pos_vocab: Vocabulary,
            primitive_types_vocab: Vocabulary,
            modifiers_vocab: Vocabulary,
            combine_parts_by: Literal['add', 'project'] = 'project',
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(ASTNodesEmbedder, self).__init__()
        self.ast_node_embedding_dim = ast_node_embedding_dim
        self.ast_node_type_vocab = ast_node_type_vocab
        self.ast_node_major_type_vocab = ast_node_major_type_vocab
        self.ast_node_minor_type_vocab = ast_node_minor_type_vocab
        self.ast_node_nr_children_vocab = ast_node_nr_children_vocab
        self.ast_node_child_pos_vocab = ast_node_child_pos_vocab
        self.primitive_types_vocab = primitive_types_vocab
        self.modifiers_vocab = modifiers_vocab
        self.combine_parts_by = combine_parts_by
        if self.combine_parts_by not in {'project', 'add'}:
            raise ValueError(
                f"`combine_parts_by` must be one of {{'project', 'add'}} (`{self.combine_parts_by}` given).")

        if self.combine_parts_by == 'project':
            self.ast_node_type_embedding_dim = self.ast_node_embedding_dim
            self.ast_node_major_type_embedding_dim = int(self.ast_node_type_embedding_dim * (2 / 3))
            self.ast_node_minor_type_embedding_dim = \
                self.ast_node_type_embedding_dim - self.ast_node_major_type_embedding_dim
            self.ast_node_nr_children_embedding_dim = 32
            self.ast_node_child_pos_embedding_dim = 32
            self.ast_nodes_embedding_dim_wo_ipm = \
                self.ast_node_type_embedding_dim + \
                self.ast_node_nr_children_embedding_dim + \
                self.ast_node_child_pos_embedding_dim
            self.primitive_type_embedding_dim = primitive_type_embedding_dim
            self.modifier_embedding_dim = modifier_embedding_dim
            self.identifier_encoding_dim = identifier_encoding_dim
        elif self.combine_parts_by == 'add':
            assert identifier_encoding_dim == self.ast_node_embedding_dim
            assert primitive_type_embedding_dim == self.ast_node_embedding_dim
            assert modifier_embedding_dim == self.ast_node_embedding_dim
            self.ast_node_type_embedding_dim = self.ast_node_embedding_dim
            self.ast_node_major_type_embedding_dim = self.ast_node_embedding_dim
            self.ast_node_minor_type_embedding_dim = self.ast_node_embedding_dim
            self.ast_node_nr_children_embedding_dim = self.ast_node_embedding_dim
            self.ast_node_child_pos_embedding_dim = self.ast_node_embedding_dim
            self.ast_nodes_embedding_dim_wo_ipm = self.ast_node_embedding_dim
            self.primitive_type_embedding_dim = self.ast_node_embedding_dim
            self.modifier_embedding_dim = self.ast_node_embedding_dim
            self.identifier_encoding_dim = self.ast_node_embedding_dim
        else:
            assert False

        # all-nodes mandatory embeddings
        self.ast_node_type_embedding_layer = nn.Embedding(
            num_embeddings=len(self.ast_node_type_vocab),
            embedding_dim=self.ast_node_type_embedding_dim,
            padding_idx=self.ast_node_type_vocab.get_word_idx('<PAD>'))
        self.ast_node_major_type_embedding_layer = nn.Embedding(
            num_embeddings=len(self.ast_node_major_type_vocab),
            embedding_dim=self.ast_node_major_type_embedding_dim,
            padding_idx=self.ast_node_major_type_vocab.get_word_idx('<PAD>'))
        self.ast_node_minor_type_embedding_layer = nn.Embedding(
            num_embeddings=len(self.ast_node_minor_type_vocab),
            embedding_dim=self.ast_node_minor_type_embedding_dim,
            padding_idx=self.ast_node_minor_type_vocab.get_word_idx('<PAD>'))
        self.ast_node_nr_children_embedding_layer = nn.Embedding(
            num_embeddings=len(self.ast_node_nr_children_vocab),
            embedding_dim=self.ast_node_nr_children_embedding_dim,
            padding_idx=self.ast_node_nr_children_vocab.get_word_idx('<PAD>'))
        self.ast_node_child_pos_embedding_layer = nn.Embedding(
            num_embeddings=len(self.ast_node_child_pos_vocab),
            embedding_dim=self.ast_node_child_pos_embedding_dim,
            padding_idx=self.ast_node_child_pos_vocab.get_word_idx('<PAD>'))

        # embeddings for certain node types
        self.primitive_types_embedding_layer = nn.Embedding(
            num_embeddings=len(self.primitive_types_vocab),
            embedding_dim=self.primitive_type_embedding_dim,
            padding_idx=self.primitive_types_vocab.get_word_idx('<PAD>'))
        self.modifiers_embedding_layer = nn.Embedding(
            num_embeddings=len(self.modifiers_vocab),
            embedding_dim=self.modifier_embedding_dim,
            padding_idx=self.modifiers_vocab.get_word_idx('<PAD>'))

        if self.combine_parts_by == 'project':
            self.identifier_leaf_linear_projection_layer = nn.Linear(
                in_features=self.identifier_encoding_dim + self.ast_nodes_embedding_dim_wo_ipm,
                out_features=self.ast_node_embedding_dim)
            self.primitive_type_leaf_linear_projection_layer = nn.Linear(
                in_features=self.primitive_type_embedding_dim + self.ast_nodes_embedding_dim_wo_ipm,
                out_features=self.ast_node_embedding_dim)
            self.modifier_leaf_linear_projection_layer = nn.Linear(
                in_features=self.modifier_embedding_dim + self.ast_nodes_embedding_dim_wo_ipm,
                out_features=self.ast_node_embedding_dim)
            self.ast_nodes_embeddings_projection_layer_wo_ipm = nn.Linear(
                in_features=self.ast_nodes_embedding_dim_wo_ipm,
                out_features=self.ast_node_embedding_dim)

        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(
            self,
            method_ast_input: MethodASTInputTensors,
            identifiers_encodings: torch.Tensor) -> torch.Tensor:
        # `ast_nodes_embeddings_wo_ipm` involves the followings:
        #   (i+ii) major/minor type embeddings,
        #   (iii) #children embeddings
        #   (iv+v) child pos (ltr&rtl) embeddings.
        ast_nodes_types_embeddings = self.ast_node_type_embedding_layer(
            method_ast_input.ast_node_types.tensor)
        ast_nodes_major_types_embeddings = self.ast_node_major_type_embedding_layer(
            method_ast_input.ast_node_major_types.tensor)
        ast_nodes_minor_types_embeddings = self.ast_node_minor_type_embedding_layer(
            method_ast_input.ast_node_minor_types.tensor)
        if self.combine_parts_by == 'project':
            ast_nodes_major_minor_types_embeddings = \
                torch.cat([ast_nodes_major_types_embeddings, ast_nodes_minor_types_embeddings], dim=-1)
        elif self.combine_parts_by == 'add':
            ast_nodes_major_minor_types_embeddings = \
                ast_nodes_major_types_embeddings + ast_nodes_minor_types_embeddings
        else:
            assert False
        ast_nodes_types_embeddings = torch.where(
            (method_ast_input.ast_node_minor_types.tensor ==
             self.ast_node_minor_type_vocab.get_word_idx('<PAD>')).unsqueeze(-1),
            ast_nodes_types_embeddings,
            ast_nodes_major_minor_types_embeddings)
        ast_nodes_nr_children_embeddings = self.ast_node_nr_children_embedding_layer(
            method_ast_input.ast_node_nr_children.tensor)
        ast_nodes_child_ltr_position_embeddings = self.ast_node_child_pos_embedding_layer(
            method_ast_input.ast_node_child_ltr_position.tensor)
        ast_nodes_child_rtl_position_embeddings = self.ast_node_child_pos_embedding_layer(
            method_ast_input.ast_node_child_rtl_position.tensor)
        ast_nodes_child_position_embeddings = \
            ast_nodes_child_ltr_position_embeddings + ast_nodes_child_rtl_position_embeddings

        ast_nodes_with_identifier_leaf_identifier_embeddings =\
            identifiers_encodings[method_ast_input.ast_nodes_with_identifier_leaf_identifier_idx.indices]
        ast_nodes_with_primitive_type_leaf_primitive_type_embeddings = self.primitive_types_embedding_layer(
                method_ast_input.ast_nodes_with_primitive_type_leaf_primitive_type.tensor)
        ast_nodes_with_modifier_leaf_modifier_embeddings = self.modifiers_embedding_layer(
            method_ast_input.ast_nodes_with_modifier_leaf_modifier.tensor)

        if self.combine_parts_by == 'project':
            ast_nodes_embeddings_wo_ipm = self.dropout_layer(torch.cat([
                ast_nodes_types_embeddings,
                ast_nodes_nr_children_embeddings,
                ast_nodes_child_position_embeddings], dim=-1))
            ast_nodes_embeddings = apply_disjoint_updates_to_encodings_tensor(
                ast_nodes_embeddings_wo_ipm,
                EncodingsUpdater(
                    new_embeddings=ast_nodes_with_identifier_leaf_identifier_embeddings,
                    element_indices=method_ast_input.ast_nodes_with_identifier_leaf_nodes_indices.indices,
                    linear_projection_layer=self.identifier_leaf_linear_projection_layer),
                EncodingsUpdater(
                    new_embeddings=self.dropout_layer(ast_nodes_with_primitive_type_leaf_primitive_type_embeddings),
                    element_indices=method_ast_input.ast_nodes_with_primitive_type_leaf_nodes_indices.indices,
                    linear_projection_layer=self.primitive_type_leaf_linear_projection_layer),
                EncodingsUpdater(
                    new_embeddings=self.dropout_layer(ast_nodes_with_modifier_leaf_modifier_embeddings),
                    element_indices=method_ast_input.ast_nodes_with_modifier_leaf_nodes_indices.indices,
                    linear_projection_layer=self.modifier_leaf_linear_projection_layer),
                otherwise_linear=self.ast_nodes_embeddings_projection_layer_wo_ipm)
            ast_nodes_embeddings = self.dropout_layer(ast_nodes_embeddings)
        elif self.combine_parts_by == 'add':
            ast_nodes_embeddings_wo_ipm = \
                self.dropout_layer(ast_nodes_types_embeddings) + \
                self.dropout_layer(ast_nodes_nr_children_embeddings) + \
                self.dropout_layer(ast_nodes_child_position_embeddings)
            ast_nodes_embeddings = apply_disjoint_additions_to_encodings_tensor(
                ast_nodes_embeddings_wo_ipm,
                EncodingsUpdater(
                    new_embeddings=ast_nodes_with_identifier_leaf_identifier_embeddings,
                    element_indices=method_ast_input.ast_nodes_with_identifier_leaf_nodes_indices.indices),
                EncodingsUpdater(
                    new_embeddings=self.dropout_layer(ast_nodes_with_primitive_type_leaf_primitive_type_embeddings),
                    element_indices=method_ast_input.ast_nodes_with_primitive_type_leaf_nodes_indices.indices),
                EncodingsUpdater(
                    new_embeddings=self.dropout_layer(ast_nodes_with_modifier_leaf_modifier_embeddings),
                    element_indices=method_ast_input.ast_nodes_with_modifier_leaf_nodes_indices.indices))
        else:
            assert False
        return ast_nodes_embeddings
