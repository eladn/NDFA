from enum import Enum
from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['CFGGNNEncoderParams']


@dataclass
class CFGGNNEncoderParams:
    class GNNType(Enum):
        GGNN = 'GGNN'
        GCN = 'GCN'
        TransformerConv = 'TransformerConv'
        GAT = 'GAT'
        GATv2 = 'GATv2'

    gnn_type: GNNType = conf_field(
        default=GNNType.GGNN)
    nr_layers: int = conf_field(
        default=2)
