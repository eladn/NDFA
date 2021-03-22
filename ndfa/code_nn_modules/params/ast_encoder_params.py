from confclass import confparam
from dataclasses import dataclass

from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams
from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams


__all__ = ['ASTEncoderParams']


@dataclass
class ASTEncoderParams:
    encoder_type: str = confparam(
        default='paths-folded',
        choices=('set-of-paths', 'tree', 'paths-folded'),
        description="Representation type of the AST (specific architecture of the AST code encoder).")

    paths_sequence_encoder_params: SequenceEncoderParams = confparam(
        default_factory=SequenceEncoderParams,
        arg_prefix='paths_sequence_encoder')

    paths_combiner_params: SequenceCombinerParams = confparam(
        default_factory=SequenceCombinerParams)

    nodes_folding_params: ScatterCombinerParams = confparam(
        default_factory=ScatterCombinerParams)
