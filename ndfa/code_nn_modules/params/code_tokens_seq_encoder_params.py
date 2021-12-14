__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-04-05"

from dataclasses import dataclass
from typing import Optional, List

from ndfa.misc.tensors_data_class import FragmentSizeDistribution
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.misc.configurations_utils import conf_field, HasDispatchableField, DispatchField


__all__ = ['CodeTokensSeqEncoderParams']


@dataclass
class CodeTokensSeqEncoderParams(HasDispatchableField):
    @dataclass
    class ShufflingOptions(HasDispatchableField):
        fragmented_shuffling: bool = conf_field(
            default=False)
        fragmented_shuffling_distribution_params: Optional[FragmentSizeDistribution] = conf_field(
            default_factory=FragmentSizeDistribution)

        @classmethod
        def set_dispatch_fields(cls):
            cls.register_dispatch_field(DispatchField(
                'fragmented_shuffling', {True: ['fragmented_shuffling_distribution_params'], False: []}))

    token_type_embedding_dim: int = conf_field(
        default=64,
        description="Embedding size for code token type (operator, identifier, etc).")

    kos_token_embedding_dim: int = conf_field(
        default=256,
        description="Embedding size for code keyword/operator/separator token.")

    token_encoding_dim: int = conf_field(
        default=256,
        # default_factory_with_self_access=lambda _self:
        # _self.identifier_embedding_size + _self.code_token_type_embedding_size,
        # default_description="identifier_embedding_size + code_token_type_embedding_size",
        description="Size of encoded code token vector.")

    sequence_encoder: SequenceEncoderParams = conf_field(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence-encoder')

    shuffle_expressions: bool = conf_field(
        default=False)

    shuffling_options: Optional[ShufflingOptions] = conf_field(
        default_factory=ShufflingOptions)

    @classmethod
    def set_dispatch_fields(cls):
        cls.register_dispatch_field(DispatchField(
            'shuffle_expressions', {True: ['shuffling_options'], False: []}))

    ignore_token_kinds: Optional[List[str]] = conf_field(
        default=None)
