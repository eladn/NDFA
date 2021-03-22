from confclass import confparam
from dataclasses import dataclass


__all__ = ['CFGGNNEncoderParams']


@dataclass
class CFGGNNEncoderParams:
    gnn_type: str = confparam(
        default='ggnn',
        choices=('ggnn', 'res_ggnn', 'gcn', 'transformer_conv', ))
    nr_layers: int = confparam(
        default=2)
