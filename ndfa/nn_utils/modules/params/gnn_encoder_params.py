__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-12-12"

from enum import Enum
from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['GNNEncoderParams']


@dataclass
class GNNEncoderParams:
    class GNNType(Enum):
        GGNN = 'GGNN'
        GCN = 'GCN'
        TransformerConv = 'TransformerConv'
        GAT = 'GAT'
        GATv2 = 'GATv2'

    gnn_type: GNNType = conf_field(
        default=GNNType.GGNN)
    nr_layers: int = conf_field(
        default=4)
