from dataclasses import dataclass
from confclass import confparam

from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams

__all__ = ['CFGSubASTExpressionCombinerParams']


@dataclass
class CFGSubASTExpressionCombinerParams(ScatterCombinerParams):
    combining_subject: str = confparam(
        default='ast_nodes',
        choices=('ast_nodes', 'ast_paths'))
