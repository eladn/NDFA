__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-14"

import dataclasses
from typing import Optional

from omegaconf import OmegaConf

from ndfa.misc.configurations_utils import DeterministicallyHashable
from ndfa.nn_utils.model_wrapper.dataset_properties import DatasetProperties
from ndfa.misc.configurations_utils import reinstantiate_omegaconf_container


__all__ = [
    'ASTPathsPreprocessParams', 'ASTPreprocessParams', 'NGramsPreprocessParams',
    'ControlFlowPathsPreprocessParams', 'ControlFlowFlatSeqPreprocessParams',
    'HierarchicMethodEncoderPreprocessParams', 'MethodCodePreprocessParams',
    'NDFAModelPreprocessParams', 'NDFAModelPreprocessedDataParams'
]


@dataclasses.dataclass
class ASTPathsPreprocessParams:
    traversal: bool = False
    leaf_to_leaf: bool = False
    leaf_to_leaf_shuffler: bool = False
    leaf_to_root: bool = False
    leaf_to_root_shuffler: bool = False
    leaves_sequence: bool = False
    leaves_sequence_shuffler: bool = False
    siblings_sequences: bool = False
    siblings_w_parent_sequences: bool = False

    @classmethod
    def full(cls):
        return ASTPathsPreprocessParams(
            traversal=True,
            leaf_to_leaf=True,
            leaf_to_leaf_shuffler=True,
            leaf_to_root=True,
            leaf_to_root_shuffler=True,
            leaves_sequence=True,
            leaves_sequence_shuffler=True,
            siblings_sequences=True,
            siblings_w_parent_sequences=True)

    def is_containing(self, other: 'ASTPathsPreprocessParams') -> bool:
        return (self.traversal or not other.traversal) and \
               (self.leaf_to_leaf or not other.leaf_to_leaf) and \
               (self.leaf_to_leaf_shuffler or not other.leaf_to_leaf_shuffler) and \
               (self.leaf_to_root or not other.leaf_to_root) and \
               (self.leaf_to_root_shuffler or not other.leaf_to_root_shuffler) and \
               (self.leaves_sequence or not other.leaves_sequence) and \
               (self.leaves_sequence_shuffler or not other.leaves_sequence_shuffler) and \
               (self.siblings_sequences or not other.siblings_sequences) and \
               (self.siblings_w_parent_sequences or not other.siblings_w_parent_sequences)


@dataclasses.dataclass
class ASTPreprocessParams:
    paths: Optional[ASTPathsPreprocessParams] = None
    tree: bool = False

    @classmethod
    def full(cls):
        return ASTPreprocessParams(
            paths=ASTPathsPreprocessParams.full(),
            tree=True)

    def is_containing(self, other: 'ASTPreprocessParams') -> bool:
        return ((self.paths and other.paths and self.paths.is_containing(other.paths)) or not other.paths) and \
               (self.tree or not other.tree)


@dataclasses.dataclass
class NGramsPreprocessParams:
    min_n: int = 1
    max_n: int = 6

    @classmethod
    def full(cls):
        return NGramsPreprocessParams()

    def is_containing(self, other: 'NGramsPreprocessParams') -> bool:
        return self.min_n <= other.min_n and self.max_n >= other.max_n


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

    def is_containing(self, other: 'ControlFlowPathsPreprocessParams') -> bool:
        return (self.traversal_edges or not other.traversal_edges) and \
               (self.full_paths or not other.full_paths) and \
               ((self.ngrams and other.ngrams and self.ngrams.is_containing(other.ngrams)) or not other.ngrams) and \
               (self.cfg_nodes_random_permutation or not other.cfg_nodes_random_permutation)


@dataclasses.dataclass
class ControlFlowFlatSeqPreprocessParams:
    cfg_nodes_random_permutation: bool = False

    @classmethod
    def full(cls):
        return ControlFlowFlatSeqPreprocessParams(
            cfg_nodes_random_permutation=True)

    def is_containing(self, other: 'ControlFlowFlatSeqPreprocessParams') -> bool:
        return self.cfg_nodes_random_permutation or not other.cfg_nodes_random_permutation


@dataclasses.dataclass
class HierarchicMethodEncoderPreprocessParams:
    micro_ast: Optional[ASTPreprocessParams] = None
    micro_tokens_seq: bool = False
    macro_ast: Optional[ASTPreprocessParams] = None
    control_flow_paths: Optional[ControlFlowPathsPreprocessParams] = None
    control_flow_single_flat_seq: Optional[ControlFlowFlatSeqPreprocessParams] = None
    control_flow_graph: bool = False

    @classmethod
    def full(cls):
        return HierarchicMethodEncoderPreprocessParams(
            micro_ast=ASTPreprocessParams.full(),
            micro_tokens_seq=True,
            macro_ast=ASTPreprocessParams.full(),
            control_flow_paths=ControlFlowPathsPreprocessParams.full(),
            control_flow_graph=True)

    def is_containing(self, other: 'HierarchicMethodEncoderPreprocessParams') -> bool:
        return ((self.micro_ast and other.micro_ast and self.micro_ast.is_containing(other.micro_ast)) or
                not other.micro_ast) and \
               (self.micro_tokens_seq or not other.micro_tokens_seq) and \
               ((self.macro_ast and other.macro_ast and self.macro_ast.is_containing(other.macro_ast)) or
                not other.macro_ast) and \
               ((self.control_flow_paths and other.control_flow_paths and
                 self.control_flow_paths.is_containing(other.control_flow_paths)) or
                not other.control_flow_paths) and \
               (self.control_flow_graph or not other.control_flow_graph)


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

    def is_containing(self, other: 'MethodCodePreprocessParams') -> bool:
        return ((self.whole_method_ast and other.whole_method_ast and
                 self.whole_method_ast.is_containing(other.whole_method_ast)) or
                not other.whole_method_ast) and \
               (self.whole_method_tokens_seq or not other.whole_method_tokens_seq) and \
               ((self.hierarchic and other.hierarchic and self.hierarchic.is_containing(other.hierarchic)) or
                not other.hierarchic)


@dataclasses.dataclass
class NDFAModelPreprocessParams(DeterministicallyHashable):
    method_code: MethodCodePreprocessParams

    @classmethod
    def full(cls):
        """Get an instance with all options present."""
        return NDFAModelPreprocessParams(
            method_code=MethodCodePreprocessParams.full())

    def is_containing(self, other: 'NDFAModelPreprocessParams') -> bool:
        return self.method_code.is_containing(other.method_code)


@dataclasses.dataclass
class NDFAModelPreprocessedDataParams(DeterministicallyHashable):
    preprocess_params: NDFAModelPreprocessParams
    dataset_props: DatasetProperties

    @classmethod
    def load_from_yaml(cls, yaml_filepath) -> 'NDFAModelPreprocessedDataParams':
        with open(yaml_filepath, 'r') as yaml_file:
            instance = OmegaConf.structured(NDFAModelPreprocessedDataParams)
            instance = OmegaConf.merge(instance, OmegaConf.load(yaml_file))
        return reinstantiate_omegaconf_container(instance, NDFAModelPreprocessedDataParams)

    def to_yaml(self, output_yaml_file):
        output_yaml_file.write(OmegaConf.to_yaml(OmegaConf.structured(self)))

    def is_containing(self, other: 'NDFAModelPreprocessedDataParams') -> bool:
        return self.dataset_props == other.dataset_props and \
               self.preprocess_params.is_containing(other.preprocess_params)
