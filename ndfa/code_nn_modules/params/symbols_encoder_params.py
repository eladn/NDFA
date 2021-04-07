from dataclasses import dataclass

from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams
from ndfa.misc.configurations_utils import conf_field


__all__ = ['SymbolsEncoderParams']


@dataclass
class SymbolsEncoderParams:
    combining_params: ScatterCombinerParams = conf_field(
        default_factory=lambda: ScatterCombinerParams(method='sum'))
    use_symbols_occurrences: bool = conf_field(
        default=True)
    use_identifier_encoding: bool = conf_field(
        default=True)
