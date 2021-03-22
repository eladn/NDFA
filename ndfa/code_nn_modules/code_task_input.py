import torch
import dataclasses
from typing import Optional
from torch_geometric.data import Data as TGData
import dgl

from ndfa.misc.tensors_data_class import TensorsDataClass, BatchFlattenedTensor, BatchFlattenedSeq, \
    TensorWithCollateMask, BatchedFlattenedIndicesFlattenedTensor, BatchedFlattenedIndicesFlattenedSeq, \
    BatchedFlattenedIndicesPseudoRandomPermutation, BatchFlattenedPseudoRandomSamplerFromRange, \
    BatchFlattenedSeqShuffler, TensorsDataDict


__all__ = ['MethodCodeInputPaddedTensors',
           'MethodCodeInputTensors', 'CodeExpressionTokensSequenceInputTensors',
           'SymbolsInputTensors', 'CFGPathsInputTensors', 'CFGPathsNGramsInputTensors',
           'PDGInputTensors', 'MethodASTInputTensors', 'SubASTInputTensors', 'IdentifiersInputTensors',
           'PDGExpressionsSubASTInputTensors']


# TODO: this is an old impl - REMOVE!
@dataclasses.dataclass
class MethodCodeInputPaddedTensors(TensorsDataClass):
    method_hash: str
    identifiers: torch.LongTensor
    sub_identifiers_mask: torch.BoolTensor
    cfg_nodes_mask: torch.BoolTensor
    cfg_nodes_control_kind: torch.LongTensor
    cfg_nodes_expressions: torch.LongTensor
    cfg_nodes_expressions_mask: torch.BoolTensor
    cfg_edges: torch.LongTensor
    cfg_edges_mask: torch.BoolTensor
    cfg_edges_attrs: torch.LongTensor
    identifiers_idxs_of_all_symbols: torch.LongTensor
    identifiers_idxs_of_all_symbols_mask: torch.BoolTensor
    indices_of_symbols_occurrences_in_cfg_nodes_expressions: TensorWithCollateMask
    symbols_idxs_of_symbols_occurrences_in_cfg_nodes_expressions: TensorWithCollateMask


@dataclasses.dataclass
class CodeExpressionTokensSequenceInputTensors(TensorsDataClass):
    token_type: BatchFlattenedSeq  # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    kos_token_index: BatchFlattenedTensor  # (nr_kos_tokens_in_all_expressions_in_batch,)
    identifier_index: BatchedFlattenedIndicesFlattenedTensor  # (nr_identifier_tokens_in_all_expressions_in_batch,)
    symbol_index: BatchedFlattenedIndicesFlattenedTensor  # (nr_symbol_occurrences_in_all_expressions_in_batch,)
    is_symbol_mask: BatchFlattenedSeq  # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    sequence_shuffler: BatchFlattenedSeqShuffler  # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    token_idx_to_ast_leaf_idx_mapping_key: BatchedFlattenedIndicesFlattenedTensor
    token_idx_to_ast_leaf_idx_mapping_value: BatchedFlattenedIndicesFlattenedTensor


@dataclasses.dataclass
class SymbolsInputTensors(TensorsDataClass):
    symbols_identifier_indices: BatchedFlattenedIndicesFlattenedTensor  # (nr_symbols_in_batch,);  value meaning: identifier batched index
    symbols_appearances_symbol_idx: BatchedFlattenedIndicesFlattenedTensor  # (nr_symbols_appearances,);
    symbols_appearances_expression_token_idx: BatchFlattenedTensor = None  # (nr_symbols_appearances,);
    symbols_appearances_cfg_expression_idx: BatchedFlattenedIndicesFlattenedTensor = None  # (nr_symbols_appearances,);


@dataclasses.dataclass
class CFGPathsInputTensors(TensorsDataClass):
    nodes_indices: BatchedFlattenedIndicesFlattenedSeq
    edges_types: BatchFlattenedSeq


@dataclasses.dataclass
class CFGPathsNGramsInputTensors(TensorsDataClass):
    nodes_indices: BatchedFlattenedIndicesFlattenedSeq
    edges_types: BatchFlattenedSeq


@dataclasses.dataclass
class PDGInputTensors(TensorsDataClass):
    cfg_nodes_control_kind: Optional[BatchFlattenedTensor] = None  # (nr_cfg_nodes_in_batch, )
    cfg_nodes_has_expression_mask: Optional[BatchFlattenedTensor] = None  # (nr_cfg_nodes_in_batch, )
    cfg_nodes_tokenized_expressions: Optional[CodeExpressionTokensSequenceInputTensors] = None
    # cfg_nodes_expressions_ref_to_method_tokenized_expressions: Optional[BatchFlattenedTensor] = None
    cfg_nodes_expressions_ast: Optional['PDGExpressionsSubASTInputTensors'] = None

    # cfg_edges: Optional[torch.LongTensor] = None
    # cfg_edges_lengths: Optional[torch.BoolTensor] = None
    # cfg_edges_attrs: Optional[torch.LongTensor] = None

    cfg_nodes_random_permutation: Optional[BatchedFlattenedIndicesPseudoRandomPermutation] = None
    cfg_control_flow_paths: Optional[CFGPathsInputTensors] = None
    cfg_control_flow_paths_exact_ngrams: Optional[TensorsDataDict[int, CFGPathsNGramsInputTensors]] = None
    cfg_control_flow_paths_partial_ngrams: Optional[TensorsDataDict[int, CFGPathsNGramsInputTensors]] = None

    cfg_control_flow_graph: Optional[TGData] = None

    @property
    def nr_cfg_nodes(self) -> int:
        return self.cfg_nodes_control_kind.tensor.size(0)


@dataclasses.dataclass
class IdentifiersInputTensors(TensorsDataClass):
    sub_parts_batch: BatchFlattenedTensor  # (nr_sub_parts_in_batch, )
    sub_parts_vocab_word_index: BatchFlattenedTensor  # (nr_sub_parts_in_batch, )

    identifier_sub_parts_index: BatchedFlattenedIndicesFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier)
    identifier_sub_parts_vocab_word_index: BatchFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier)
    identifiers_vocab_word_index: BatchFlattenedTensor  # (nr_identifiers_in_batch, )
    identifier_sub_parts_hashings: BatchFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier, nr_hashing_features)
    sub_parts_obfuscation: BatchFlattenedPseudoRandomSamplerFromRange  # (nr_sub_parts_obfuscation_embeddings)


@dataclasses.dataclass
class SubASTInputTensors(TensorsDataClass):
    ast_leaf_to_leaf_paths_node_indices: BatchedFlattenedIndicesFlattenedSeq
    ast_leaf_to_leaf_paths_child_place: BatchFlattenedSeq
    ast_leaf_to_leaf_paths_vertical_direction: BatchFlattenedSeq
    ast_leaf_to_root_paths_node_indices: BatchedFlattenedIndicesFlattenedSeq
    ast_leaf_to_root_paths_child_place: BatchFlattenedSeq
    ast_leaves_sequence_node_indices: BatchedFlattenedIndicesFlattenedSeq
    siblings_sequences_node_indices: BatchedFlattenedIndicesFlattenedSeq
    siblings_w_parent_sequences_node_indices: BatchedFlattenedIndicesFlattenedSeq
    dgl_tree: dgl.DGLGraph

    def get_ast_paths_node_indices(self, path_type: str) -> BatchedFlattenedIndicesFlattenedSeq:
        if path_type == 'leaf_to_leaf':
            return self.ast_leaf_to_leaf_paths_node_indices
        elif path_type == 'leaf_to_root':
            return self.ast_leaf_to_root_paths_node_indices
        elif path_type == 'leaves_sequence':
            return self.ast_leaves_sequence_node_indices
        elif path_type == 'siblings_sequences':
            return self.siblings_sequences_node_indices
        elif path_type == 'siblings_w_parent_sequences':
            return self.siblings_w_parent_sequences_node_indices
        else:
            raise ValueError(f'Unsupported path type `{path_type}`.')

    def get_ast_paths_child_place(self, path_type: str) -> Optional[BatchFlattenedSeq]:
        if path_type == 'leaf_to_leaf':
            return self.ast_leaf_to_leaf_paths_child_place
        elif path_type == 'leaf_to_root':
            return self.ast_leaf_to_root_paths_child_place
        elif path_type in {'leaves_sequence', 'siblings_sequences', 'siblings_w_parent_sequences'}:
            return None
        else:
            raise ValueError(f'Unsupported path type `{path_type}`.')

    def get_ast_paths_vertical_direction(self, path_type: str) -> Optional[BatchFlattenedSeq]:
        if path_type == 'leaf_to_leaf':
            return self.ast_leaf_to_leaf_paths_vertical_direction
        elif path_type in {'leaf_to_root', 'leaves_sequence', 'siblings_sequences', 'siblings_w_parent_sequences'}:
            return None
        else:
            raise ValueError(f'Unsupported path type `{path_type}`.')


@dataclasses.dataclass
class MethodASTInputTensors(SubASTInputTensors):
    ast_node_types: BatchFlattenedTensor
    ast_node_major_types: BatchFlattenedTensor
    ast_node_minor_types: BatchFlattenedTensor
    ast_node_child_ltr_position: BatchFlattenedTensor
    ast_node_child_rtl_position: BatchFlattenedTensor
    ast_node_nr_children: BatchFlattenedTensor

    ast_nodes_with_identifier_leaf_nodes_indices: BatchedFlattenedIndicesFlattenedTensor
    ast_nodes_with_identifier_leaf_identifier_idx: BatchedFlattenedIndicesFlattenedTensor

    ast_nodes_with_symbol_leaf_nodes_indices: BatchedFlattenedIndicesFlattenedTensor
    ast_nodes_with_symbol_leaf_symbol_idx: BatchedFlattenedIndicesFlattenedTensor

    ast_nodes_with_primitive_type_leaf_nodes_indices: BatchedFlattenedIndicesFlattenedTensor
    ast_nodes_with_primitive_type_leaf_primitive_type: BatchFlattenedTensor

    ast_nodes_with_modifier_leaf_nodes_indices: BatchedFlattenedIndicesFlattenedTensor
    ast_nodes_with_modifier_leaf_modifier: BatchFlattenedTensor


@dataclasses.dataclass
class PDGExpressionsSubASTInputTensors(SubASTInputTensors):
    ast_leaf_to_leaf_paths_pdg_node_indices: BatchedFlattenedIndicesFlattenedTensor
    ast_leaf_to_root_paths_pdg_node_indices: BatchedFlattenedIndicesFlattenedTensor
    siblings_sequences_pdg_node_indices: BatchedFlattenedIndicesFlattenedTensor

    pdg_node_idx_to_sub_ast_root_idx_mapping_key: BatchedFlattenedIndicesFlattenedTensor
    pdg_node_idx_to_sub_ast_root_idx_mapping_value: BatchedFlattenedIndicesFlattenedTensor
    ast_node_idx_to_pdg_node_idx_mapping_key: BatchedFlattenedIndicesFlattenedTensor
    ast_node_idx_to_pdg_node_idx_mapping_value: BatchedFlattenedIndicesFlattenedTensor

    def get_ast_paths_pdg_node_indices(self, path_type: str) -> Optional[BatchedFlattenedIndicesFlattenedTensor]:
        if path_type == 'leaf_to_leaf':
            return self.ast_leaf_to_leaf_paths_pdg_node_indices
        elif path_type == 'leaf_to_root':
            return self.ast_leaf_to_root_paths_pdg_node_indices
        elif path_type == 'leaves_sequence':
            return None
        elif path_type in {'siblings_sequences', 'siblings_w_parent_sequences'}:
            return self.siblings_sequences_pdg_node_indices
        else:
            raise ValueError(f'Unsupported path type `{path_type}`.')


@dataclasses.dataclass
class MethodCodeInputTensors(TensorsDataClass):
    method_hash: str

    identifiers: IdentifiersInputTensors
    symbols: SymbolsInputTensors

    method_tokenized_code: Optional[CodeExpressionTokensSequenceInputTensors] = None
    pdg: Optional[PDGInputTensors] = None
    ast: Optional[MethodASTInputTensors] = None
