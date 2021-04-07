from dataclasses import dataclass

from ndfa.misc.configurations_utils import conf_field


__all__ = ['CFGGNNEncoderParams']


@dataclass
class CFGGNNEncoderParams:
    gnn_type: str = conf_field(
        default='ggnn',
        choices=('ggnn', 'res_ggnn', 'gcn', 'transformer_conv', ))
    nr_layers: int = conf_field(
        default=2)
