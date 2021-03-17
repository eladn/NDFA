from confclass import confclass, confparam
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams


__all__ = ['IdentifierEncoderParams']


@confclass
class IdentifierEncoderParams:
    identifier_embedding_dim: int = confparam(
        default=256,
        description="Embedding size for an identifier.")
    nr_sub_identifier_hashing_features: int = confparam(
        default=256)
    sequence_encoder: SequenceEncoderParams = confparam(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence_encoder')
    sequence_combiner: SequenceCombinerParams = confparam(
        default_factory=SequenceCombinerParams,
        arg_prefix='sequence_combiner')
