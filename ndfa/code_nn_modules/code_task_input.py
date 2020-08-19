import torch
import dataclasses
from typing import Optional, Dict

from ndfa.misc.tensors_data_class import TensorsDataClass, BatchFlattenedTensor, BatchFlattenedSeq, \
    TensorWithCollateMask, BatchedFlattenedIndicesFlattenedTensor


__all__ = ['MethodCodeInputToEncoder']


# TODO: this is an old impl - REMOVE!
@dataclasses.dataclass
class MethodCodeInputToEncoder(TensorsDataClass):
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


# TODO: complete impl and use
@dataclasses.dataclass
class CodeExpressionTokensSequenceInputTensors(TensorsDataClass):
    token_type: BatchFlattenedSeq  # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    kos_token_index: BatchFlattenedTensor  # (nr_kos_tokens_in_all_expressions_in_batch,)
    identifier_index: BatchedFlattenedIndicesFlattenedTensor  # (nr_identifier_tokens_in_all_expressions_in_batch,)


# TODO: complete impl and use
@dataclasses.dataclass
class SymbolsInputTensors(TensorsDataClass):  # TODO: inherit from some other version of `TensorsDataClass` that supports indexing
    symbols_identifier_indices: BatchedFlattenedIndicesFlattenedTensor  # (nr_symbols_in_batch,);  value meaning: identifier batched index
    symbols_appearances_symbol_idx: BatchedFlattenedIndicesFlattenedTensor  # (nr_symbols_appearances,);
    symbols_appearances_expression_token_idx: BatchedFlattenedIndicesFlattenedTensor = None  # (nr_symbols_appearances,);
    symbols_appearances_cfg_expression_idx: BatchedFlattenedIndicesFlattenedTensor = None  # (nr_symbols_appearances,);


class PDGInputTensors(TensorsDataClass):
    cfg_nodes_control_kind: Optional[BatchFlattenedTensor] = None  # (nr_cfg_nodes_in_batch, )
    cfg_nodes_has_expression_mask: Optional[BatchFlattenedTensor] = None  # (nr_cfg_nodes_in_batch, )
    cfg_nodes_tokenized_expressions: Optional[CodeExpressionTokensSequenceInputTensors] = None
    # cfg_nodes_expressions_ref_to_method_tokenized_expressions: Optional[BatchFlattenedTensor] = None

    # cfg_edges: Optional[torch.LongTensor] = None
    # cfg_edges_lengths: Optional[torch.BoolTensor] = None
    # cfg_edges_attrs: Optional[torch.LongTensor] = None

    # cfg_control_flow_paths: Optional[Dict[str, ExampleBasedIndicesBatchFlattenedTensor]] = None


# TODO: complete impl and use
@dataclasses.dataclass
class MethodCodeInputTensors(TensorsDataClass):
    method_hash: str

    identifiers_sub_parts: BatchFlattenedSeq  # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier)
    symbols: SymbolsInputTensors

    method_tokenized_code: Optional[CodeExpressionTokensSequenceInputTensors] = None
    pdg: Optional[PDGInputTensors] = None
