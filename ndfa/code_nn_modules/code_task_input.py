import torch
import dataclasses
from typing import Optional

from ndfa.misc.tensors_data_class import TensorsDataClass, BatchFlattenedTensor, BatchFlattenedSeq, \
    TensorWithCollateMask, BatchedFlattenedIndicesFlattenedTensor, BatchedFlattenedIndicesFlattenedSeq, \
    BatchedFlattenedIndicesPseudoRandomPermutation, BatchFlattenedPseudoRandomSamplerFromRange, \
    BatchFlattenedSeqShuffler, TensorsDataDict


__all__ = ['MethodCodeInputPaddedTensors',
           'MethodCodeInputTensors', 'CodeExpressionTokensSequenceInputTensors',
           'SymbolsInputTensors', 'CFGPathsInputTensors', 'CFGPathsNGramsInputTensors',
           'PDGInputTensors', 'IdentifiersInputTensors']


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
    sequence_permuter: BatchFlattenedSeqShuffler  # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)


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

    # cfg_edges: Optional[torch.LongTensor] = None
    # cfg_edges_lengths: Optional[torch.BoolTensor] = None
    # cfg_edges_attrs: Optional[torch.LongTensor] = None

    cfg_nodes_random_permutation: Optional[BatchedFlattenedIndicesPseudoRandomPermutation] = None
    cfg_control_flow_paths: Optional[CFGPathsInputTensors] = None
    cfg_control_flow_paths_ngrams: Optional[TensorsDataDict[int, CFGPathsNGramsInputTensors]] = None


@dataclasses.dataclass
class IdentifiersInputTensors(TensorsDataClass):
    sub_parts_batch: BatchFlattenedTensor  # (nr_sub_parts_in_batch, )
    identifier_sub_parts_index: BatchedFlattenedIndicesFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier)
    identifier_sub_parts_vocab_word_index: BatchFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier)
    identifier_sub_parts_hashings: BatchFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier, nr_hashing_features)
    sub_parts_obfuscation: BatchFlattenedPseudoRandomSamplerFromRange  # (nr_sub_parts_obfuscation_embeddings)


@dataclasses.dataclass
class MethodCodeInputTensors(TensorsDataClass):
    method_hash: str

    identifiers: IdentifiersInputTensors
    symbols: SymbolsInputTensors

    method_tokenized_code: Optional[CodeExpressionTokensSequenceInputTensors] = None
    pdg: Optional[PDGInputTensors] = None
