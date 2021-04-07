from dataclasses import dataclass

from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.nn_utils.modules.params.embedding_with_unknowns_params import EmbeddingWithUnknownsParams
from ndfa.misc.configurations_utils import conf_field


__all__ = ['IdentifierEncoderParams']


@dataclass
class IdentifierEncoderParams:
    identifier_embedding_dim: int = conf_field(
        default=256,
        description="Embedding size for an identifier.")
    nr_sub_identifier_hashing_features: int = conf_field(
        default=256)
    nr_identifier_hashing_features: int = conf_field(
        default=256)
    sequence_encoder: SequenceEncoderParams = conf_field(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence_encoder')
    sequence_combiner: SequenceCombinerParams = conf_field(
        default_factory=SequenceCombinerParams,
        arg_prefix='sequence_combiner')
    use_sub_identifiers: bool = conf_field(
        default=True)
    embedding_params: EmbeddingWithUnknownsParams = conf_field(
        default_factory=EmbeddingWithUnknownsParams)
