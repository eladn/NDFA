import torch
import torch.nn as nn

from ndfa.nn_utils.modules.scatter_combiner import ScatterCombiner


__all__ = ['CFGSubASTExpressionCombiner']


class CFGSubASTExpressionCombiner(nn.Module):
    def __init__(self, ast_node_encoding_dim: int, combined_dim: int,
                 combining_subject: str = 'ast_nodes', combining_method: str = 'attn',
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGSubASTExpressionCombiner, self).__init__()
        self.combined_dim = combined_dim
        assert combining_subject in {'ast_nodes', 'ast_paths'}
        self.combining_subject = combining_subject
        self.combining_method = combining_method
        self.ast_node_encoding_dim = ast_node_encoding_dim
        self.scatter_combiner_layer = ScatterCombiner(
            encoding_dim=ast_node_encoding_dim, combining_method=combining_method,
            nr_attn_heads=8, applied_attn_output_dim=combined_dim)  # TODO: plug `nr_attn_heads` in HPs

    def forward(self, ast_nodes_encodings: torch.Tensor,
                ast_node_idx_to_pdg_node_idx_mapping_key: torch.LongTensor,
                ast_node_idx_to_pdg_node_idx_mapping_value: torch.LongTensor,
                pdg_node_idx_to_sub_ast_root_idx_mapping_key: torch.LongTensor,
                pdg_node_idx_to_sub_ast_root_idx_mapping_value: torch.LongTensor,
                nr_cfg_nodes: int):
        if self.combining_subject == 'ast_nodes':
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
        elif self.combining_subject == 'ast_paths':
            raise NotImplementedError  # TODO: impl
            # ast_paths_combined
        else:
            assert False
