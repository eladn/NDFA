from dataclasses import dataclass
from confclass import confparam

from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams


__all__ = ['SymbolsEncoderParams']


@dataclass
class SymbolsEncoderParams:
    combining_params: ScatterCombinerParams = confparam(
        default_factory=lambda: ScatterCombinerParams(method='sum'))

    use_symbols_occurrences: bool = confparam(
        default=True)
