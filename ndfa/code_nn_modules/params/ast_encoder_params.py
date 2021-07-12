from dataclasses import dataclass
from typing import Tuple, Optional

from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams
from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams
from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField, conf_field


__all__ = ['ASTEncoderParams']


@dataclass
class ASTEncoderParams(HasDispatchableField):
    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'encoder_type', {
                'paths-folded': ['paths_sequence_encoder_params', 'paths_combiner_params', 'nodes_folding_params', 'ast_paths_types', 'paths_add_traversal_edges'],  # TODO: remove 'paths_combiner_params'?
                'set-of-paths': ['paths_sequence_encoder_params', 'paths_combiner_params', 'ast_paths_types', 'paths_add_traversal_edges'],
                'tree': []}))
    encoder_type: str = conf_field(
        default='paths-folded',
        choices=('set-of-paths', 'tree', 'paths-folded'),
        description="Representation type of the AST (specific architecture of the AST code encoder).")

    ast_node_embedding_dim: int = conf_field(
        default=256)

    paths_sequence_encoder_params: SequenceEncoderParams = conf_field(
        default_factory=SequenceEncoderParams,
        arg_prefix='paths_sequence_encoder')

    paths_add_traversal_edges: Optional[bool] = conf_field(
        default=True)

    paths_combiner_params: Optional[SequenceCombinerParams] = conf_field(
        default_factory=SequenceCombinerParams)

    nodes_folding_params: Optional[ScatterCombinerParams] = conf_field(
        default_factory=ScatterCombinerParams)

    ast_paths_types: Tuple[str, ...] = conf_field(
        default=('leaf_to_leaf', 'leaf_to_root'),
        elements_choices=['leaf_to_leaf', 'leaf_to_root', 'siblings_sequences',
                          'siblings_w_parent_sequences', 'leaves_sequence'])
