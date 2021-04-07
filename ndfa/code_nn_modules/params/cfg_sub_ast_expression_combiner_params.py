from dataclasses import dataclass

from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams
from ndfa.misc.configurations_utils import conf_field

__all__ = ['CFGSubASTExpressionCombinerParams']


@dataclass
class CFGSubASTExpressionCombinerParams(ScatterCombinerParams):
    combining_subject: str = conf_field(
        default='ast_nodes',
        choices=('ast_nodes', 'ast_paths'))
