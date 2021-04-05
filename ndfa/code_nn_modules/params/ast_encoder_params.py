from confclass import confparam
from dataclasses import dataclass
from typing import Tuple

from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams
from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams
from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField


__all__ = ['ASTEncoderParams']


@dataclass
class ASTEncoderParams(HasDispatchableField):
    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'encoder_type', {
                'paths-folded': ['paths_sequence_encoder_params', 'nodes_folding_params', 'ast_paths_types'],
                'set-of-paths': ['paths_sequence_encoder_params', 'paths_combiner_params', 'ast_paths_types'],
                'tree': []}))
    encoder_type: str = confparam(
        default='paths-folded',
        choices=('set-of-paths', 'tree', 'paths-folded'),
        description="Representation type of the AST (specific architecture of the AST code encoder).")

    ast_node_embedding_dim: int = confparam(
        default=256)

    paths_sequence_encoder_params: SequenceEncoderParams = confparam(
        default_factory=SequenceEncoderParams,
        arg_prefix='paths_sequence_encoder')

    paths_combiner_params: SequenceCombinerParams = confparam(
        default_factory=SequenceCombinerParams)

    nodes_folding_params: ScatterCombinerParams = confparam(
        default_factory=ScatterCombinerParams)

    ast_paths_types: Tuple[str, ...] = confparam(
        default=('leaf_to_leaf', 'leaf_to_root'))  # 'leaf_to_leaf', 'leaf_to_root', 'siblings_sequences', 'siblings_w_parent_sequences'
