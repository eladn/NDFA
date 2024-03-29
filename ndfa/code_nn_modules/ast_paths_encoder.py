__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-11-23"

import torch
import torch.nn as nn
from typing import Optional, Dict

from ndfa.nn_utils.misc.misc import seq_lengths_to_mask
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.nn_utils.modules.state_updater import StateUpdater
from ndfa.nn_utils.modules.scatter_combiner import ScatterCombiner
from ndfa.nn_utils.functions.weave_tensors import weave_tensors, unweave_tensor
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors, \
    ASTPathsEncodingsTensors
from ndfa.code_nn_modules.code_task_input import SubASTInputTensors
from ndfa.nn_utils.modules.sequence_combiner import SequenceCombiner
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.nn_utils.modules.norm_wrapper import NormWrapper


__all__ = ['ASTPathsEncoder']


class ASTPathsEncoder(nn.Module):
    def __init__(
            self,
            ast_node_embedding_dim: int,
            encoder_params: ASTEncoderParams,
            is_first_encoder_layer: bool = True,
            ast_traversal_orientation_vocab: Optional[Vocabulary] = None,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(ASTPathsEncoder, self).__init__()
        self.encoder_params = encoder_params
        assert self.encoder_params.encoder_type in \
               {ASTEncoderParams.EncoderType.PathsFolded, ASTEncoderParams.EncoderType.SetOfPaths}
        self.ast_node_embedding_dim = ast_node_embedding_dim
        assert all(
            ast_paths_type in {'leaf_to_leaf', 'leaf_to_root', 'siblings_sequences',
                               'siblings_w_parent_sequences', 'leaves_sequence'}
            for ast_paths_type in self.encoder_params.ast_paths_types)
        self.is_first_encoder_layer = is_first_encoder_layer

        # TODO: Add and use HP param `reuse_paths_encodings_from_previous_input_layer`.
        if self.is_first_encoder_layer:
            if self.encoder_params.paths_add_traversal_edges:
                self.ast_traversal_orientation_vocab = ast_traversal_orientation_vocab
                self.ast_traversal_orientation_embedding_layer = nn.ModuleDict({
                    ast_paths_type: nn.Embedding(
                        num_embeddings=len(self.ast_traversal_orientation_vocab),
                        embedding_dim=self.ast_node_embedding_dim,
                        padding_idx=self.ast_traversal_orientation_vocab.get_word_idx('<PAD>'))
                    for ast_paths_type in self.encoder_params.ast_paths_types
                    if SubASTInputTensors.path_type_has_child_place(ast_paths_type) or
                       SubASTInputTensors.path_type_has_vertical_direction(ast_paths_type)})
                self.ast_traversal_orientation_linear_projection_layer = nn.ModuleDict({
                    ast_paths_type: nn.Linear(
                        in_features=2 * self.ast_node_embedding_dim,
                        out_features=self.ast_node_embedding_dim)
                    for ast_paths_type in self.encoder_params.ast_paths_types
                    if SubASTInputTensors.path_type_has_child_place(ast_paths_type) and
                       SubASTInputTensors.path_type_has_vertical_direction(ast_paths_type)})
        else:
            self.nodes_occurrences_update_gate = nn.ModuleDict({
                ast_paths_type: StateUpdater(
                    state_dim=self.ast_node_embedding_dim,
                    params=self.encoder_params.state_updater_for_nodes_occurrences_from_previous_layer,
                    update_dim=self.ast_node_embedding_dim,
                    dropout_rate=dropout_rate, activation_fn=activation_fn)
                for ast_paths_type in self.encoder_params.ast_paths_types})

        self.path_sequence_encoder = nn.ModuleDict({
            ast_paths_type: SequenceEncoder(
                encoder_params=self.encoder_params.paths_sequence_encoder_params,
                input_dim=self.ast_node_embedding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for ast_paths_type in self.encoder_params.ast_paths_types})

        if self.encoder_params.encoder_type == ASTEncoderParams.EncoderType.PathsFolded:
            self.nodes_representation_path_folder = ScatterCombiner(
                encoding_dim=self.ast_node_embedding_dim,
                combiner_params=self.encoder_params.nodes_folding_params)
        self.path_combiner = nn.ModuleDict({
            ast_paths_type: SequenceCombiner(
                encoding_dim=self.ast_node_embedding_dim,
                combined_dim=self.ast_node_embedding_dim,  # TODO: define a dedicated HP. it should be bigger.
                combiner_params=self.encoder_params.paths_combiner_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for ast_paths_type in self.encoder_params.ast_paths_types})
        self.norm = None if norm_params is None else NormWrapper(
            nr_features=self.ast_node_embedding_dim, params=norm_params)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward_single_path_type(
            self,
            ast_nodes_encodings: torch.Tensor,
            sub_ast_input: SubASTInputTensors,
            ast_paths_type: str,
            ast_paths_last_states: Optional[ASTPathsEncodingsTensors] = None) -> ASTPathsEncodingsTensors:
        ast_paths_node_indices = sub_ast_input.get_ast_paths_node_indices(ast_paths_type)
        paths_shuffler = \
            sub_ast_input.get_ast_paths_shuffler(ast_paths_type) if self.encoder_params.shuffle_ast_paths else None
        assert not self.encoder_params.shuffle_ast_paths or not self.encoder_params.paths_add_traversal_edges

        ast_paths_traversal_orientation_encodings_input = None
        if ast_paths_last_states is not None and \
                self.encoder_params.encoder_type == ASTEncoderParams.EncoderType.SetOfPaths:
            # If no collapse stage, don't re-expand the AST nodes encodings. Actually, the AST nodes encodings are
            #   not being updated when there is no collapse, so they include the initial AST nodes embeddings.
            # Instead, just use the paths encoding from previous layer.
            ast_paths_node_occurrences_encodings_inputs = ast_paths_last_states.nodes_occurrences
            ast_paths_traversal_orientation_encodings_input = ast_paths_last_states.traversal_orientation
        elif self.is_first_encoder_layer:  # TODO: Use HP param `reuse_paths_encodings_from_previous_input_layer`.
            assert ast_paths_node_indices is not None
            ast_paths_node_occurrences_encodings_inputs = ast_nodes_encodings[ast_paths_node_indices.sequences]
            if self.encoder_params.paths_add_traversal_edges:
                ast_paths_traversal_orientation_encodings = []
                if sub_ast_input.path_type_has_child_place(ast_paths_type):
                    ast_paths_traversal_orientation_encodings.append(
                        self.ast_traversal_orientation_embedding_layer[ast_paths_type]
                            (sub_ast_input.get_ast_paths_child_place(ast_paths_type).sequences))
                if sub_ast_input.path_type_has_vertical_direction(ast_paths_type):
                    ast_paths_traversal_orientation_encodings.append(
                        self.ast_traversal_orientation_embedding_layer[ast_paths_type]
                            (sub_ast_input.get_ast_paths_vertical_direction(ast_paths_type).sequences))
                if len(ast_paths_traversal_orientation_encodings) == 1:
                    ast_paths_traversal_orientation_encodings_input = ast_paths_traversal_orientation_encodings[0]
                elif len(ast_paths_traversal_orientation_encodings) > 1:
                    assert len(ast_paths_traversal_orientation_encodings) == 2
                    ast_paths_traversal_orientation_encodings_input = \
                        self.ast_traversal_orientation_linear_projection_layer[ast_paths_type](torch.cat(
                            ast_paths_traversal_orientation_encodings, dim=-1))
        else:
            # update nodes occurrences (in the path) with the last nodes representation using an update gate.
            ast_paths_node_occurrences_encodings_inputs = self.nodes_occurrences_update_gate[ast_paths_type](
                previous_state=ast_paths_last_states.nodes_occurrences,
                state_update=ast_nodes_encodings[ast_paths_node_indices.sequences])
            ast_paths_traversal_orientation_encodings_input = ast_paths_last_states.traversal_orientation

        # TODO: fix following `assert` for case when no path-types with traversal are used (like `leaves_sequence`)!
        assert paths_shuffler is None or ast_paths_traversal_orientation_encodings_input is None

        if ast_paths_traversal_orientation_encodings_input is None:
            ast_paths_embeddings_input = ast_paths_node_occurrences_encodings_inputs
            ast_paths_lengths = ast_paths_node_indices.sequences_lengths
        else:
            ast_paths_embeddings_input = weave_tensors(tensors=[
                ast_paths_node_occurrences_encodings_inputs,
                ast_paths_traversal_orientation_encodings_input], dim=1)
            ast_paths_with_edges_mask = seq_lengths_to_mask(
                seq_lengths=ast_paths_node_indices.sequences_lengths * 2,
                max_seq_len=2 * ast_paths_node_indices.sequences.size(1))
            ast_paths_embeddings_input = ast_paths_embeddings_input.masked_fill(
                ~ast_paths_with_edges_mask.unsqueeze(-1), 0)
            ast_paths_lengths = ast_paths_node_indices.sequences_lengths * 2

        if paths_shuffler is not None:
            ast_paths_embeddings_input = paths_shuffler.shuffle(ast_paths_embeddings_input)
        ast_paths_encoded = self.path_sequence_encoder[ast_paths_type](
            sequence_input=ast_paths_embeddings_input,
            lengths=ast_paths_lengths, batch_first=True).sequence
        if paths_shuffler is not None:
            ast_paths_encoded = paths_shuffler.unshuffle(ast_paths_encoded)

        if self.norm:
            ast_paths_encoded = self.norm(ast_paths_encoded)

        if ast_paths_traversal_orientation_encodings_input is None:
            ast_paths_nodes_encodings = ast_paths_encoded
            ast_paths_traversal_orientation_encodings = None
        else:
            ast_paths_nodes_encodings, ast_paths_traversal_orientation_encodings = unweave_tensor(
                woven_tensor=ast_paths_encoded, dim=1, nr_target_tensors=2)

        # TODO: do it conditionally only when needed
        ast_paths_combined = self.path_combiner[ast_paths_type](
            sequence_encodings=ast_paths_nodes_encodings,
            sequence_mask=ast_paths_node_indices.sequences_mask,
            sequence_lengths=ast_paths_node_indices.sequences_lengths,
            batch_first=True)
        if self.norm:
            ast_paths_combined = self.norm(ast_paths_combined)

        return ASTPathsEncodingsTensors(
            nodes_occurrences=ast_paths_nodes_encodings,
            traversal_orientation=ast_paths_traversal_orientation_encodings,
            combined=ast_paths_combined)

    def forward(
            self,
            ast_nodes_encodings: torch.Tensor,
            sub_ast_input: SubASTInputTensors,
            ast_paths_last_states: Optional[Dict[str, ASTPathsEncodingsTensors]] = None) \
            -> CodeExpressionEncodingsTensors:
        nr_ast_nodes = ast_nodes_encodings.size(0)
        assert ast_paths_last_states is None or \
               set(ast_paths_last_states.keys()) == set(self.encoder_params.ast_paths_types)

        encoded_paths_by_path_type = {
            ast_paths_type: self.forward_single_path_type(
                ast_nodes_encodings=ast_nodes_encodings,
                sub_ast_input=sub_ast_input,
                ast_paths_type=ast_paths_type,
                ast_paths_last_states=None if ast_paths_last_states is None else ast_paths_last_states[ast_paths_type])
            for ast_paths_type in self.encoder_params.ast_paths_types}

        if self.encoder_params.encoder_type == ASTEncoderParams.EncoderType.PathsFolded:
            ast_paths_masks = {
                ast_paths_type: sub_ast_input.get_ast_paths_node_indices(ast_paths_type).sequences_mask
                for ast_paths_type in encoded_paths_by_path_type.keys()}
            all_ast_paths_nodes_encodings = torch.cat([
                encoded_paths.nodes_occurrences[ast_paths_masks[ast_paths_type]]
                for ast_paths_type, encoded_paths in encoded_paths_by_path_type.items()], dim=0)
            all_ast_paths_node_indices = torch.cat([
                sub_ast_input.get_ast_paths_node_indices(ast_paths_type).sequences[ast_paths_masks[ast_paths_type]]
                for ast_paths_type in encoded_paths_by_path_type.keys()], dim=0)
            new_ast_nodes_encodings = self.nodes_representation_path_folder(
                scattered_input=all_ast_paths_nodes_encodings,
                indices=all_ast_paths_node_indices,
                dim_size=nr_ast_nodes, attn_queries=ast_nodes_encodings)
            if self.norm:
                new_ast_nodes_encodings = self.norm(new_ast_nodes_encodings)
        else:
            new_ast_nodes_encodings = ast_nodes_encodings

        return CodeExpressionEncodingsTensors(
            ast_nodes=new_ast_nodes_encodings,
            ast_paths_by_type=encoded_paths_by_path_type,
            ast_paths_types=self.encoder_params.ast_paths_types)
