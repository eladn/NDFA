import torch
import torch.nn as nn
from warnings import warn
from typing import Tuple, NamedTuple, Collection

from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors, MethodASTInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.nn_utils.functions.last_item_in_sequence import get_last_item_in_sequence


__all__ = ['SymbolOccurrencesExtractorFromEncodedMethod']


class ScatteredOccurrences(NamedTuple):
    encodings: torch.Tensor
    indices: torch.LongTensor

    def verify_matching_shapes(self):
        assert self.encodings.shape[:-1] == self.indices.shape

    @classmethod
    def cat(cls, items: Collection['ScatteredOccurrences']) -> 'ScatteredOccurrences':
        scattered_occ = cls(
            encodings=torch.cat([scattered_occ.encodings for scattered_occ in items], dim=0),
            indices=torch.cat([scattered_occ.indices for scattered_occ in items], dim=0))
        scattered_occ.verify_matching_shapes()
        return scattered_occ


def get_symbols_occurrences_from_ast_nodes_occurrences(
        method_ast_input: MethodASTInputTensors,
        ast_nodes_occurrences: ScatteredOccurrences) \
        -> ScatteredOccurrences:
    ast_nodes_occurrences.verify_matching_shapes()
    # TODO: delete cast to bool after we fix the pp input
    ast_nodes_has_symbol_mask = \
        method_ast_input.ast_nodes_has_symbol_mask.tensor.type(torch.bool)
    symbol_leaves_mask = ast_nodes_has_symbol_mask[ast_nodes_occurrences.indices]
    encodings_of_symbols_occurrences = ast_nodes_occurrences.encodings[symbol_leaves_mask]
    ast_node_indices_of_symbols_occurrences = ast_nodes_occurrences.indices[symbol_leaves_mask]
    symbols_indices_of_symbols_occurrences = \
        method_ast_input.ast_nodes_symbol_idx.indices[ast_node_indices_of_symbols_occurrences]
    symbols_occurrences = ScatteredOccurrences(
        encodings=encodings_of_symbols_occurrences, indices=symbols_indices_of_symbols_occurrences)
    symbols_occurrences.verify_matching_shapes()
    return symbols_occurrences


class SymbolOccurrencesExtractorFromEncodedMethod(nn.Module):
    def __init__(self, code_expression_encoder_params: CodeExpressionEncoderParams):
        super(SymbolOccurrencesExtractorFromEncodedMethod, self).__init__()
        self.code_expression_encoder_params = code_expression_encoder_params

    def forward(
            self,
            code_expression_encodings: CodeExpressionEncodingsTensors,
            tokenized_expressions_input: CodeExpressionTokensSequenceInputTensors,
            method_ast_input: MethodASTInputTensors) -> Tuple[torch.Tensor, torch.LongTensor]:
        if self.code_expression_encoder_params.encoder_type == 'ast':
            if self.code_expression_encoder_params.ast_encoder.encoder_type == 'set-of-paths':
                assert code_expression_encodings.ast_paths_by_type is not None
                all_symbols_occurrences = []
                for paths_type, ast_paths in code_expression_encodings.ast_paths_by_type.items():
                    if paths_type == 'leaf_to_leaf':
                        assert len(ast_paths.nodes_occurrences.shape) == 3
                        assert method_ast_input.ast_leaf_to_leaf_paths_node_indices.sequences.shape == \
                               ast_paths.nodes_occurrences.shape[:2]
                        first_item_encoding = ast_paths.nodes_occurrences[:, 0, :]
                        first_item_node_index = method_ast_input.ast_leaf_to_leaf_paths_node_indices.sequences[:, 0]
                        last_item_encoding = get_last_item_in_sequence(
                            sequence_encodings=ast_paths.nodes_occurrences,
                            sequence_lengths=method_ast_input.ast_leaf_to_leaf_paths_node_indices.sequences_lengths)
                        last_item_node_index = get_last_item_in_sequence(
                            sequence_encodings=method_ast_input.ast_leaf_to_leaf_paths_node_indices.sequences.unsqueeze(-1),
                            sequence_lengths=method_ast_input.ast_leaf_to_leaf_paths_node_indices.sequences_lengths).squeeze(-1)
                        assert first_item_encoding.shape == last_item_encoding.shape
                        assert first_item_node_index.shape == last_item_node_index.shape
                        assert first_item_encoding.shape[:-1] == first_item_node_index.shape
                        ast_nodes_occurrences = ScatteredOccurrences(
                            encodings=torch.cat([first_item_encoding, last_item_encoding], dim=0),
                            indices=torch.cat([first_item_node_index, last_item_node_index], dim=0))
                        ast_nodes_occurrences.verify_matching_shapes()
                        all_symbols_occurrences.append(get_symbols_occurrences_from_ast_nodes_occurrences(
                            method_ast_input=method_ast_input, ast_nodes_occurrences=ast_nodes_occurrences))
                    elif paths_type == 'leaf_to_root':
                        leaves_encodings = ast_paths.nodes_occurrences[:, 0, :]
                        leaves_indices = method_ast_input.ast_leaf_to_root_paths_node_indices.sequences[:, 0]
                        ast_nodes_occurrences = ScatteredOccurrences(
                            encodings=leaves_encodings, indices=leaves_indices)
                        ast_nodes_occurrences.verify_matching_shapes()
                        all_symbols_occurrences.append(get_symbols_occurrences_from_ast_nodes_occurrences(
                            method_ast_input=method_ast_input, ast_nodes_occurrences=ast_nodes_occurrences))
                    elif paths_type in {'siblings_sequences', 'siblings_w_parent_sequences'}:
                        node_indices = method_ast_input.siblings_sequences_node_indices \
                            if paths_type == 'siblings_sequences' \
                            else method_ast_input.siblings_w_parent_sequences_node_indices
                        ast_nodes_occurrences = ScatteredOccurrences(
                            encodings=ast_paths.nodes_occurrences[node_indices.sequences_mask],
                            indices=node_indices.sequences[node_indices.sequences_mask])
                        ast_nodes_occurrences.verify_matching_shapes()
                        all_symbols_occurrences.append(get_symbols_occurrences_from_ast_nodes_occurrences(
                            method_ast_input=method_ast_input, ast_nodes_occurrences=ast_nodes_occurrences))
                    else:
                        warn(f'Extracting symbol occurrences from ast paths of type '
                             f'`{paths_type}` is currently not supported.')
                assert len(all_symbols_occurrences) >= 1
                all_symbols_occurrences = all_symbols_occurrences[0] if len(all_symbols_occurrences) == 1 else \
                    ScatteredOccurrences.cat(all_symbols_occurrences)
                all_symbols_occurrences.verify_matching_shapes()
                encodings_of_symbols_occurrences, symbols_indices_of_symbols_occurrences = all_symbols_occurrences
            else:
                assert self.code_expression_encoder_params.ast_encoder.encoder_type in {'tree', 'paths-folded'}
                assert code_expression_encodings.ast_nodes is not None
                encoded_ast_nodes = code_expression_encodings.ast_nodes
                ast_nodes_with_symbol_leaf_nodes_indices = method_ast_input.ast_nodes_with_symbol_leaf_nodes_indices.indices
                ast_nodes_with_symbol_leaf_symbol_idx = method_ast_input.ast_nodes_with_symbol_leaf_symbol_idx.indices
                assert ast_nodes_with_symbol_leaf_nodes_indices.shape == ast_nodes_with_symbol_leaf_symbol_idx.shape
                encodings_of_symbols_occurrences = encoded_ast_nodes[ast_nodes_with_symbol_leaf_nodes_indices, :]
                symbols_indices_of_symbols_occurrences = ast_nodes_with_symbol_leaf_symbol_idx
        elif self.code_expression_encoder_params.encoder_type == 'FlatTokensSeq':
            assert code_expression_encodings.token_seqs is not None
            assert tokenized_expressions_input.is_symbol_mask.sequences.shape == \
                   code_expression_encodings.token_seqs.shape[:-1]
            encodings_of_symbols_occurrences = \
                code_expression_encodings.token_seqs[tokenized_expressions_input.is_symbol_mask.sequences]
            assert encodings_of_symbols_occurrences.shape[:-1] == tokenized_expressions_input.symbol_index.indices.shape
            symbols_indices_of_symbols_occurrences = tokenized_expressions_input.symbol_index.indices  # symbols.symbols_appearances_symbol_idx.indices
        else:
            assert False

        return encodings_of_symbols_occurrences, symbols_indices_of_symbols_occurrences
