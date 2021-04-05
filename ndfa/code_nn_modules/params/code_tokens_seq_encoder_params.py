from confclass import confparam
from dataclasses import dataclass

from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams


__all__ = ['CodeTokensSeqEncoderParams']


@dataclass
class CodeTokensSeqEncoderParams:
    token_type_embedding_dim: int = confparam(
        default=64,
        description="Embedding size for code token type (operator, identifier, etc).")

    kos_token_embedding_dim: int = confparam(
        default=256,
        description="Embedding size for code keyword/operator/separator token.")

    token_encoding_dim: int = confparam(
        default=256,
        # default_factory_with_self_access=lambda _self:
        # _self.identifier_embedding_size + _self.code_token_type_embedding_size,
        # default_description="identifier_embedding_size + code_token_type_embedding_size",
        description="Size of encoded code token vector.")

    sequence_encoder: SequenceEncoderParams = confparam(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence-encoder')

    shuffle_expressions: bool = confparam(
        default=False)
