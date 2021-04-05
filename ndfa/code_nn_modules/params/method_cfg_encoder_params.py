from typing import Optional

from confclass import confparam
from dataclasses import dataclass
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_nn_modules.params.cfg_gnn_encoder_params import CFGGNNEncoderParams
from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams
from ndfa.misc.configurations_utils import HasDispatchableField, DispatchField


__all__ = ['MethodCFGEncoderParams']


@dataclass
class MethodCFGEncoderParams(HasDispatchableField):
    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'encoder_type', {
                'pdg-paths-folded-to-nodes': ['cfg_paths_sequence_encoder', 'cfg_nodes_folding_params'],
                'set-of-control-flow-paths': ['cfg_paths_sequence_encoder'],
                'control-flow-paths-folded-to-nodes': ['cfg_paths_sequence_encoder', 'cfg_nodes_folding_params'],
                'gnn': ['cfg_gnn_encoder'],
                'set-of-control-flow-paths-ngrams': ['cfg_paths_sequence_encoder', 'create_sub_grams_from_long_gram', 'cfg_paths_ngrams_min_n', 'cfg_paths_ngrams_max_n'],
                'control-flow-paths-ngrams-folded-to-nodes': ['cfg_paths_sequence_encoder', 'create_sub_grams_from_long_gram', 'cfg_paths_ngrams_min_n', 'cfg_paths_ngrams_max_n', 'cfg_nodes_folding_params'],
                'set-of-nodes': [],
                'all-nodes-single-unstructured-linear-seq': ['cfg_paths_sequence_encoder'],
                'all-nodes-single-unstructured-linear-seq-ngrams': ['cfg_paths_sequence_encoder', 'create_sub_grams_from_long_gram', 'cfg_paths_ngrams_min_n', 'cfg_paths_ngrams_max_n'],
                'all-nodes-single-random-permutation-seq': ['cfg_paths_sequence_encoder']
            }))
    encoder_type: str = confparam(
        default='control-flow-paths-folded-to-nodes',
        choices=('pdg-paths-folded-to-nodes',
                 'set-of-control-flow-paths', 'control-flow-paths-folded-to-nodes', 'gnn',
                 'set-of-control-flow-paths-ngrams', 'control-flow-paths-ngrams-folded-to-nodes',
                 'set-of-nodes', 'all-nodes-single-unstructured-linear-seq',
                 'all-nodes-single-unstructured-linear-seq-ngrams',  # TODO: support it!
                 'all-nodes-single-random-permutation-seq'),
        description="Representation type of the method-CFG (specific architecture of the method-CFG-code-encoder).")

    cfg_node_expression_encoder: CodeExpressionEncoderParams = confparam(
        default_factory=CodeExpressionEncoderParams,
        description="Representation type of the expression of a CFG node "
                    "(part of the architecture of the code-encoder).",
        arg_prefix='cfg_node_expression_encoder')

    # TODO: dispatch it somehow; not relevant for non tokens-seq encoder
    cfg_node_tokenized_expression_combiner: SequenceCombinerParams = confparam(
        default_factory=lambda: SequenceCombinerParams(
            method='ends', nr_attn_heads=8, nr_dim_reduction_layers=0),
        arg_prefix='cfg_node_expression_combiner')

    cfg_node_control_kinds_embedding_dim: int = confparam(
        default=64,
        description="Embedding size for the CFG node control kind.")

    cfg_node_encoding_dim: int = confparam(
        default=256,
        # default_factory_with_self_access=lambda _self:
        # _self.cfg_node_type_embedding_size + _self.code_expression_encoding_size,
        # default_description="cfg_node_type_embedding_size + code_expression_encoding_size",
        description="Size of encoded CFG node vector.")

    cfg_paths_sequence_encoder: SequenceEncoderParams = confparam(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence-encoder')

    cfg_gnn_encoder: Optional[CFGGNNEncoderParams] = confparam(
        default_factory=CFGGNNEncoderParams,
        arg_prefix='gnn-encoder')

    create_sub_grams_from_long_gram: Optional[bool] = confparam(
        default=False)

    cfg_nodes_folding_params: Optional[ScatterCombinerParams] = confparam(
        default_factory=ScatterCombinerParams)

    cfg_paths_ngrams_min_n: Optional[int] = confparam(
        default=None)
    cfg_paths_ngrams_max_n: Optional[int] = confparam(
        default=3)
