import torch
import torch.nn as nn

from ndfa.nn_utils.modules.scatter_combiner import ScatterCombiner
from ndfa.code_nn_modules.code_task_input import PDGExpressionsSubASTInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.code_nn_modules.params.cfg_sub_ast_expression_combiner_params import CFGSubASTExpressionCombinerParams


__all__ = ['CFGSubASTExpressionCombiner']


class CFGSubASTExpressionCombiner(nn.Module):
    def __init__(self, ast_node_encoding_dim: int, combined_dim: int,
                 combining_params: CFGSubASTExpressionCombinerParams,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGSubASTExpressionCombiner, self).__init__()
        self.combined_dim = combined_dim
        self.combining_params = combining_params
        self.ast_node_encoding_dim = ast_node_encoding_dim
        self.scatter_combiner_layer = ScatterCombiner(
            encoding_dim=ast_node_encoding_dim,
            combiner_params=self.combining_params,
            applied_attn_output_dim=combined_dim)

    def forward(self,
                encoded_code_expressions: CodeExpressionEncodingsTensors,
                cfg_nodes_expressions_ast_input: PDGExpressionsSubASTInputTensors,
                nr_cfg_nodes: int):
        if self.combining_params.combining_subject == 'ast_nodes':
            ast_node_idx_to_pdg_node_idx_mapping_key = \
                cfg_nodes_expressions_ast_input.ast_node_idx_to_pdg_node_idx_mapping_key.indices
            ast_node_idx_to_pdg_node_idx_mapping_value = \
                cfg_nodes_expressions_ast_input.ast_node_idx_to_pdg_node_idx_mapping_value.indices
            pdg_node_idx_to_sub_ast_root_idx_mapping_key = \
                cfg_nodes_expressions_ast_input.pdg_node_idx_to_sub_ast_root_idx_mapping_key.indices
            pdg_node_idx_to_sub_ast_root_idx_mapping_value = \
                cfg_nodes_expressions_ast_input.pdg_node_idx_to_sub_ast_root_idx_mapping_value.indices
            ast_nodes_encodings = encoded_code_expressions.ast_nodes

            attn_queries = ast_nodes_encodings[pdg_node_idx_to_sub_ast_root_idx_mapping_value]
            attn_queries = ast_nodes_encodings.new_zeros(size=(nr_cfg_nodes, self.ast_node_encoding_dim)).scatter_(
                dim=0,
                index=pdg_node_idx_to_sub_ast_root_idx_mapping_key.unsqueeze(-1).expand(attn_queries.shape),
                src=attn_queries)
            combined_sub_asts = self.scatter_combiner_layer(
                scattered_input=ast_nodes_encodings[ast_node_idx_to_pdg_node_idx_mapping_key],
                indices=ast_node_idx_to_pdg_node_idx_mapping_value,
                dim_size=nr_cfg_nodes,
                attn_queries=attn_queries)
            assert combined_sub_asts.size() == (nr_cfg_nodes, self.combined_dim)
            return combined_sub_asts
        elif self.combining_params.combining_subject == 'ast_paths':
            all_ast_paths_combined = torch.cat(
                [ast_paths.combined for ast_paths in encoded_code_expressions.ast_paths_by_type.values()], dim=0)
            all_ast_paths_pdg_node_indices = torch.cat(
                [cfg_nodes_expressions_ast_input.get_ast_paths_pdg_node_indices(ast_paths_type).indices
                 for ast_paths_type in encoded_code_expressions.ast_paths_by_type.keys()], dim=0)
            # TODO: FIXME: what attn query should we use here?
            #  we currently use zeros so only the bias vector affects as a global attention.
            attn_queries = all_ast_paths_combined.new_zeros(size=(nr_cfg_nodes, self.ast_node_encoding_dim))
            combined_sub_asts = self.scatter_combiner_layer(
                scattered_input=all_ast_paths_combined,
                indices=all_ast_paths_pdg_node_indices,
                dim_size=nr_cfg_nodes,
                attn_queries=attn_queries)
            assert combined_sub_asts.size() == (nr_cfg_nodes, self.combined_dim)
            return combined_sub_asts
        else:
            assert False
