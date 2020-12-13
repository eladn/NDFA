import torch
import dataclasses
from typing import Optional


__all__ = ['CodeExpressionEncodingsTensors']


@dataclasses.dataclass
class CodeExpressionEncodingsTensors:
    token_seqs: Optional[torch.Tensor] = None
    ast_nodes: Optional[torch.Tensor] = None
    ast_paths_nodes_occurrences: Optional[torch.Tensor] = None
    ast_paths_traversal_orientation: Optional[torch.Tensor] = None
    combined_expressions: Optional[torch.Tensor] = None
