import torch
import torch.nn as nn

from ndfa.nn_utils.modules.params.state_updater_params import StateUpdaterParams
from ndfa.nn_utils.modules.state_updater import StateUpdater
from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams
from ndfa.nn_utils.modules.scatter_combiner import ScatterCombiner


__all__ = ['ScatterEncodedPathsToNodeEncodings']


class ScatterEncodedPathsToNodeEncodings(nn.Module):
    def __init__(
            self,
            node_encoding_dim: int,
            folding_params: ScatterCombinerParams,
            state_updater_params: StateUpdaterParams,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(ScatterEncodedPathsToNodeEncodings, self).__init__()
        self.folding_params = folding_params
        self.scatter_combiner_layer = ScatterCombiner(
            encoding_dim=node_encoding_dim, combiner_params=folding_params)
        self.gate = StateUpdater(
            state_dim=node_encoding_dim, params=state_updater_params,
            update_dim=node_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(
            self,
            encoded_paths: torch.Tensor,
            paths_mask: torch.BoolTensor,
            paths_node_indices: torch.LongTensor,
            previous_nodes_encodings: torch.Tensor,
            nr_nodes: int):
        # `encoded_paths` is in form of sequences. We flatten it by applying a mask selector.
        # The mask also helps to ignore paddings.
        if self.folding_params.method == 'attn':
            assert previous_nodes_encodings is not None
            assert previous_nodes_encodings.size(0) == nr_nodes
        updated_nodes_encodings = self.scatter_combiner_layer(
            scattered_input=encoded_paths[paths_mask],
            indices=paths_node_indices[paths_mask],
            dim_size=nr_nodes,
            attn_queries=previous_nodes_encodings)
        assert updated_nodes_encodings.size() == (nr_nodes, encoded_paths.size(2))

        new_nodes_encodings = self.gate(
            previous_state=previous_nodes_encodings,
            state_update=updated_nodes_encodings)
        return new_nodes_encodings
