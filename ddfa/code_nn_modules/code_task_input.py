import torch
import dataclasses

from ddfa.misc.tensors_data_class import TensorsDataClass


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


# new design:
# identifiers: torch.LongTensor
# sub_identifiers_lengths: torch.LongTensor
# method_code_tokenized: torch.LongTensor
# method_code_tokenized_lengths: torch.LongTensor
# cfg_nodes_lengths: torch.LongTensor
# cfg_nodes_control_kind: torch.LongTensor
# cfg_nodes_expressions_tokenized: torch.LongTensor
# cfg_nodes_expressions_tokenized_lengths: torch.LongTensor
# cfg_edges: torch.LongTensor
# cfg_edges_lengths: torch.LongTensor
# cfg_edges_attrs: torch.LongTensor
# identifiers_idxs_of_all_symbols: torch.LongTensor
# identifiers_idxs_of_all_symbols_lengths: torch.LongTensor
