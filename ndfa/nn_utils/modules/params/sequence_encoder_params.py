from dataclasses import dataclass
from typing import Optional
from .sequence_combiner_params import SequenceCombinerParams

from ndfa.misc.configurations_utils import conf_field


__all__ = ['SequenceEncoderParams']


@dataclass
class SequenceEncoderParams:
    encoder_type: str = conf_field(
        default='rnn',
        choices=('rnn', 'transformer'),
        description="...")
    rnn_type: str = conf_field(
        default='lstm', choices=('lstm', 'gru'))
    nr_rnn_layers: int = conf_field(
        default=1)
    bidirectional_rnn: bool = conf_field(
        default=True)
    sequence_combiner: Optional[SequenceCombinerParams] = conf_field(
        default=None)
