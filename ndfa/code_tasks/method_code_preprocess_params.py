import dataclasses
from typing import Optional


__all__ = [
    'ASTPathsPreprocessParams', 'ASTPreprocessParams', 'NGramsPreprocessParams',
    'ControlFlowPathsPreprocessParams', 'HierarchicMethodEncoderPreprocessParams',
    'MethodCodePreprocessParams', 'NDFAModelPreprocessParams'
]


@dataclasses.dataclass
class ASTPathsPreprocessParams:
    traversal: bool = False
    leaf_to_leaf: bool = False
    leaf_to_root: bool = False
    leaves_sequence: bool = False
    siblings_sequences: bool = False
    siblings_w_parent_sequences: bool = False


@dataclasses.dataclass
class ASTPreprocessParams:
    paths: Optional[ASTPathsPreprocessParams] = None
    tree: bool = False


@dataclasses.dataclass
class NGramsPreprocessParams:
    min_n: int = 1
    max_n: int = 6


@dataclasses.dataclass
class ControlFlowPathsPreprocessParams:
    traversal_edges: bool = False
    full_paths: bool = False
    ngrams: Optional[NGramsPreprocessParams] = None
    cfg_nodes_random_permutation: bool = False


@dataclasses.dataclass
class HierarchicMethodEncoderPreprocessParams:
    micro_ast: Optional[ASTPreprocessParams] = None
    micro_tokens_seq: bool = False
    macro_ast: Optional[ASTPreprocessParams] = None
    control_flow_paths: Optional[ControlFlowPathsPreprocessParams] = None
    control_flow_graph = False


@dataclasses.dataclass
class MethodCodePreprocessParams:
    whole_method_ast: Optional[ASTPreprocessParams] = None
    whole_method_tokens_seq: bool = False
    hierarchic: Optional[HierarchicMethodEncoderPreprocessParams] = None

    @property
    def general_ast(self):
        return self.whole_method_ast or self.hierarchic and (self.hierarchic.macro_ast or self.hierarchic.micro_ast)


@dataclasses.dataclass
class NDFAModelPreprocessParams:
    method_code: MethodCodePreprocessParams
