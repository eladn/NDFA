import torch
import dataclasses
from typing import Optional, Dict

from ddfa.misc.tensors_data_class import TensorsDataClass, BatchFlattenedTensor, BatchFlattenedSeq, \
    ExampleBasedIndicesBatchFlattenedTensor


__all__ = ['MethodCodeInputToEncoder']


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


# TODO: complete impl and use
@dataclasses.dataclass
class TokenizedCodeExpressionsInputTensors(TensorsDataClass):
    token_type: torch.LongTensor  # (nr_expressions_in_batch, max_nr_tokens_in_expr)
    kos_token_index: torch.LongTensor  # (nr_expressions_in_batch, max_nr_tokens_in_expr)
    identifier_token_index: torch.LongTensor  # (nr_expressions_in_batch, max_nr_tokens_in_expr)
    lengths: torch.LongTensor  # (nr_expressions_in_batch, )
    example_idx: torch.LongTensor = dataclasses.field(init=False, default=None)  # (nr_expressions_in_batch, )


# TODO: complete impl and use
@dataclasses.dataclass
class SymbolsInputTensors(TensorsDataClass):  # TODO: inherit from some other version of `TensorsDataClass` that supports indexing
    symbols_identifier_indices: torch.LongTensor
    symbols_appearances_in_cfg_expressions: Optional[torch.LongTensor] = None
    symbols_appearances_in_method_code: Optional[torch.LongTensor] = None


# TODO: complete impl and use
@dataclasses.dataclass
class MethodCodeInputTensors(TensorsDataClass):
    method_hash: str

    identifiers_sub_parts: BatchFlattenedSeq  # (nr_identifiers_in_batch, max_nr_sub_parts_in_identifier)
    symbols: SymbolsInputTensors

    method_tokenized_code: Optional[TokenizedCodeExpressionsInputTensors] = None

    cfg_nodes_control_kind: Optional[BatchFlattenedTensor] = None  # (nr_cfg_nodes_in_batch, )
    cfg_nodes_tokenized_expressions: Optional[TokenizedCodeExpressionsInputTensors] = None
    cfg_nodes_expressions_ref_to_method_tokenized_expressions: Optional[BatchFlattenedTensor] = None

    cfg_edges: Optional[torch.LongTensor] = None
    cfg_edges_lengths: Optional[torch.BoolTensor] = None
    cfg_edges_attrs: Optional[torch.LongTensor] = None

    cfg_control_flow_paths: Optional[Dict[str, ExampleBasedIndicesBatchFlattenedTensor]] = None
