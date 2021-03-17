from confclass import confclass, confparam
from typing import Optional
from .sequence_combiner_params import SequenceCombinerParams


__all__ = ['SequenceEncoderParams']


@confclass
class SequenceEncoderParams:
    encoder_type: str = confparam(
        default='rnn',
        choices=('rnn', 'transformer'),
        description="...")
    rnn_type: str = confparam(
        default='lstm', choices=('lstm', 'gru'))
    nr_rnn_layers: int = confparam(
        default=1)
    bidirectional_rnn: bool = confparam(
        default=True)
    sequence_combiner: Optional[SequenceCombinerParams] = confparam(
        default=None)
