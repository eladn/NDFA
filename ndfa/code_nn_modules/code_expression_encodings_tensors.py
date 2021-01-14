import torch
import dataclasses
from typing import Optional, Tuple, Dict


__all__ = ['CodeExpressionEncodingsTensors', 'ASTPathsEncodingsTensors']


@dataclasses.dataclass
class ASTPathsEncodingsTensors:
    nodes_occurrences: Optional[torch.Tensor] = None
    traversal_orientation: Optional[torch.Tensor] = None
    combined: Optional[torch.Tensor] = None


@dataclasses.dataclass
class CodeExpressionEncodingsTensors:
    token_seqs: Optional[torch.Tensor] = None
    ast_nodes: Optional[torch.Tensor] = None
    ast_paths_by_type: Optional[Dict[str, ASTPathsEncodingsTensors]] = None
    combined_expressions: Optional[torch.Tensor] = None
    ast_paths_types: Optional[Tuple[str, ...]] = None
