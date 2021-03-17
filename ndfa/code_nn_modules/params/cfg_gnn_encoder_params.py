from confclass import confclass, confparam


__all__ = ['CFGGNNEncoderParams']


@confclass
class CFGGNNEncoderParams:
    gnn_type: str = confparam(
        default='ggnn',
        choices=('ggnn', 'res_ggnn', 'gcn', 'transformer_conv', ))
    nr_layers: int = confparam(
        default=2)
