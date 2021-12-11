__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-05"

from enum import Enum
from typing import Optional
from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams


__all__ = ['SingleFlatCFGNodesSeqMacroEncoderParams']


@dataclass
class SingleFlatCFGNodesSeqMacroEncoderParams:
    class CFGNodesOrder(Enum):
        CodeTextualAppearance = 'CodeTextualAppearance'
        Random = 'Random'

    path_sequence_encoder: Optional[SequenceEncoderParams] = conf_field(
        default_factory=SequenceEncoderParams,
        arg_prefix='sequence-encoder')
    cfg_nodes_order: CFGNodesOrder = conf_field(
        default=CFGNodesOrder.CodeTextualAppearance)
