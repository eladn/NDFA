from confclass import confclass, confparam


__all__ = ['SequenceCombinerParams']


@confclass
class SequenceCombinerParams:
    method: str = confparam(
        default='ends',
        choices=('attn', 'sum', 'mean', 'last', 'ends'),
        description="...")
    nr_attn_heads: int = confparam(
        default=8)
    nr_dim_reduction_layers: int = confparam(
        default=1)
