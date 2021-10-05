from typing import Optional
from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams


__all__ = ['CFGSinglePathMacroEncoderParams']


@dataclass
class CFGSinglePathMacroEncoderParams:
    path_sequence_encoder: Optional[SequenceEncoderParams] = conf_field(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence-encoder')
    is_random_order: bool = conf_field(
        default=False)
