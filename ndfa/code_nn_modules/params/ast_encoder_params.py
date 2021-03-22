from confclass import confparam
from dataclasses import dataclass
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams


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
