import torch
import torch.nn as nn
from typing import Optional

from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.ndfa_model_hyper_parameters import SequenceEncoderParams
from ndfa.nn_utils.modules.gate import Gate
from ndfa.nn_utils.modules.scatter_combiner import ScatterCombiner
from ndfa.nn_utils.functions.weave_tensors import weave_tensors, unweave_tensor
from ndfa.code_nn_modules.code_task_input import MethodASTInputTensors, SubASTInputTensors


__all__ = ['ASTPathsEncoder']


class ASTPathsEncoder(nn.Module):
    def __init__(
            self,
            ast_node_embedding_dim: int,
            ast_paths_sequence_encoder_params: SequenceEncoderParams,
            is_first_encoder_layer: bool = True,
            ast_node_type_vocab: Optional[Vocabulary] = None,
            ast_traversal_orientation_vocab: Optional[Vocabulary] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(ASTPathsEncoder, self).__init__()
        self.ast_node_embedding_dim = ast_node_embedding_dim
        self.is_first_encoder_layer = is_first_encoder_layer
        if self.is_first_encoder_layer:
            self.ast_node_type_vocab = ast_node_type_vocab
            self.ast_traversal_orientation_vocab = ast_traversal_orientation_vocab
            self.ast_traversal_orientation_embedding_layer = nn.Embedding(
                num_embeddings=len(self.ast_traversal_orientation_vocab),
                embedding_dim=self.ast_node_embedding_dim,
                padding_idx=self.ast_traversal_orientation_vocab.get_word_idx('<PAD>'))
            self.ast_traversal_orientation_linear_projection_layer = nn.Linear(
                in_features=2 * self.ast_node_embedding_dim, out_features=self.ast_node_embedding_dim)
            self.ast_node_type_embedding_layer = nn.Embedding(
                num_embeddings=len(self.ast_node_type_vocab),
                embedding_dim=self.ast_node_embedding_dim,
                padding_idx=self.ast_node_type_vocab.get_word_idx('<PAD>'))
        else:
            self.nodes_occurrences_update_gate = Gate(
                state_dim=self.ast_node_embedding_dim, update_dim=self.ast_node_embedding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)

        self.nodes_representation_path_folder = ScatterCombiner(
            encoding_dim=self.ast_node_embedding_dim, combining_method='sum')
        self.path_sequence_encoder = SequenceEncoder(
            encoder_params=ast_paths_sequence_encoder_params,
            input_dim=self.ast_node_embedding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(
            self,
            ast_paths_node_indices: torch.LongTensor,
            ast_paths_lengths: torch.LongTensor,
            ast_paths_mask: torch.LongTensor,
            method_ast_input: Optional[MethodASTInputTensors] = None,
            ast_nodes_representation_previous_states: Optional[torch.Tensor] = None,
            ast_paths_child_place: Optional[torch.LongTensor] = None,
            ast_paths_vertical_direction: Optional[torch.LongTensor] = None,
            ast_paths_last_states_for_nodes: Optional[torch.Tensor] = None,
            ast_paths_last_states_for_traversal_order: Optional[torch.Tensor] = None):
        if self.is_first_encoder_layer:
            ast_nodes_types_embeddings = self.ast_node_type_embedding_layer(method_ast_input.ast_nodes_types)
            ast_paths_child_place_embeddings = \
                self.ast_traversal_orientation_embedding_layer(ast_paths_child_place)
            if ast_paths_vertical_direction is None:
                ast_paths_traversal_orientation_encodings_input = ast_paths_child_place_embeddings
            else:
                ast_paths_vertical_direction_embeddings = \
                    self.ast_traversal_orientation_embedding_layer(ast_paths_vertical_direction)
                ast_paths_traversal_orientation_encodings_input = \
                    self.ast_traversal_orientation_linear_projection_layer(torch.cat([
                        ast_paths_child_place_embeddings,
                        ast_paths_vertical_direction_embeddings], dim=-1))
            ast_paths_node_occurrences_encodings_inputs = ast_nodes_types_embeddings[ast_paths_node_indices]
            nr_ast_nodes = method_ast_input.ast_nodes_types.size(0)
        else:
            # update nodes occurrences (in the path) with the last nodes representation using an update gate.
            ast_paths_node_occurrences_encodings_inputs = self.nodes_occurrences_update_gate(
                previous_state=ast_paths_last_states_for_nodes,
                state_update=ast_nodes_representation_previous_states)
            ast_paths_traversal_orientation_encodings_input = ast_paths_last_states_for_traversal_order
            nr_ast_nodes = ast_nodes_representation_previous_states.size(0)

        ast_paths_embeddings_input = weave_tensors(tensors=[
            ast_paths_node_occurrences_encodings_inputs,
            ast_paths_traversal_orientation_encodings_input], dim=1)
        ast_paths_embeddings_input = ast_paths_embeddings_input.masked_fill(
            ~ast_paths_mask.unsqueeze(-1), 0)
        ast_paths_encoded = self.path_sequence_encoder(
            sequence_input=ast_paths_embeddings_input,
            lengths=ast_paths_lengths * 2, batch_first=True)
        ast_paths_nodes_encodings, ast_paths_traversal_orientation_encodings = unweave_tensor(
            woven_tensor=ast_paths_encoded, dim=1, nr_target_tensors=2)
        new_node_representations = self.nodes_representation_path_folder(
            scattered_input=ast_paths_nodes_encodings[ast_paths_mask],
            indices=ast_paths_traversal_orientation_encodings[ast_paths_mask],
            dim_size=nr_ast_nodes, attn_key=ast_nodes_representation_previous_states)

        return new_node_representations, ast_paths_nodes_encodings, ast_paths_traversal_orientation_encodings
