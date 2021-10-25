import base64
import hashlib
import dataclasses
from typing import Optional

from ndfa.misc.configurations_utils import DeterministicallyHashable
from ndfa.nn_utils.model_wrapper.dataset_properties import DatasetProperties


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

    @classmethod
    def full(cls):
        return ASTPathsPreprocessParams(
            traversal=True,
            leaf_to_leaf=True,
            leaf_to_root=True,
            leaves_sequence=True,
            siblings_sequences=True,
            siblings_w_parent_sequences=True)


@dataclasses.dataclass
class ASTPreprocessParams:
    paths: Optional[ASTPathsPreprocessParams] = None
    tree: bool = False

    @classmethod
    def full(cls):
        return ASTPreprocessParams(
            paths=ASTPathsPreprocessParams.full(),
            tree=True)



@dataclasses.dataclass
class NGramsPreprocessParams:
    min_n: int = 1
    max_n: int = 6

    @classmethod
    def full(cls):
        return NGramsPreprocessParams()


@dataclasses.dataclass
class ControlFlowPathsPreprocessParams:
    traversal_edges: bool = False
    full_paths: bool = False
    ngrams: Optional[NGramsPreprocessParams] = None
    cfg_nodes_random_permutation: bool = False

    @classmethod
    def full(cls):
        return ControlFlowPathsPreprocessParams(
            traversal_edges=True,
            full_paths=True,
            ngrams=NGramsPreprocessParams.full(),
            cfg_nodes_random_permutation=True)


@dataclasses.dataclass
class HierarchicMethodEncoderPreprocessParams:
    micro_ast: Optional[ASTPreprocessParams] = None
    micro_tokens_seq: bool = False
    macro_ast: Optional[ASTPreprocessParams] = None
    control_flow_paths: Optional[ControlFlowPathsPreprocessParams] = None
    control_flow_graph: bool = False

    @classmethod
    def full(cls):
        return HierarchicMethodEncoderPreprocessParams(
            micro_ast=ASTPreprocessParams.full(),
            micro_tokens_seq=True,
            macro_ast=ASTPreprocessParams.full(),
            control_flow_paths=ControlFlowPathsPreprocessParams.full(),
            control_flow_graph=True)


@dataclasses.dataclass
class MethodCodePreprocessParams:
    whole_method_ast: Optional[ASTPreprocessParams] = None
    whole_method_tokens_seq: bool = False
    hierarchic: Optional[HierarchicMethodEncoderPreprocessParams] = None

    @property
    def general_ast(self):
        return self.whole_method_ast or self.hierarchic and (self.hierarchic.macro_ast or self.hierarchic.micro_ast)

    @classmethod
    def full(cls):
        return MethodCodePreprocessParams(
            whole_method_ast=ASTPreprocessParams.full(),
            whole_method_tokens_seq=True,
            hierarchic=HierarchicMethodEncoderPreprocessParams.full())


@dataclasses.dataclass
class NDFAModelPreprocessParams(DeterministicallyHashable):
    method_code: MethodCodePreprocessParams
    dataset_props: DatasetProperties

    @classmethod
    def full(cls):
        """Get an instance with all options present."""
        return NDFAModelPreprocessParams(
            method_code=MethodCodePreprocessParams.full(),
            dataset_props=DatasetProperties())
