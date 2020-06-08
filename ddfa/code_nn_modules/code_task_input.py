import torch
from typing import NamedTuple, List


__all__ = ['CodeTaskInput']


class CodeTaskInput(NamedTuple):
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
    logging_call_cfg_node_idx: torch.LongTensor
    is_batched: bool = False

    @property
    def batch_size(self) -> int:
        assert self.is_batched and len(self.identifiers.size()) == 3
        return self.identifiers.size()[0]

    def to(self, device):
        return CodeTaskInput(**{
            field_name:
                getattr(self, field_name).to(device)
                if hasattr(getattr(self, field_name), 'to') else
                getattr(self, field_name)
            for field_name in self._fields})

    @classmethod
    def collate(cls, code_task_inputs: List['CodeTaskInput']):
        assert all(not code_task_input.is_batched for code_task_input in code_task_inputs)
        for field_name in cls._fields:
            if field_name == 'is_batched':
                continue
            if any(getattr(code_task_input, field_name).size() != getattr(code_task_inputs[0], field_name).size()
                   for code_task_input in code_task_inputs):
                raise ValueError(
                    f'Not all examples have the same tensor size for `{field_name}`. '
                    f'sizes: {[getattr(code_task_input, field_name).size() for code_task_input in code_task_inputs]}')
        return cls(
            **{field_name: torch.cat(
                tuple(getattr(code_task_input, field_name).unsqueeze(0) for code_task_input in code_task_inputs), dim=0)
               for field_name in cls._fields if field_name != 'is_batched'},
            is_batched=True)


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
# logging_call_cfg_node_idx: torch.LongTensor
# is_batched: bool = False
