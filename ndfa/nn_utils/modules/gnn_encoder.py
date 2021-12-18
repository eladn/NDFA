__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-12-12"

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric import nn as tgnn
from torch_geometric.data import Data as TGData

from ndfa.nn_utils.modules.params.gnn_encoder_params import GNNEncoderParams
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.modules.norm_wrapper import NormWrapper
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams


__all__ = ['GNNEncoder']


class GNNEncoder(nn.Module):
    def __init__(
            self,
            node_encoding_dim: int,
            encoder_params: GNNEncoderParams,
            edge_types_vocab: Optional[Vocabulary] = None,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(GNNEncoder, self).__init__()
        self.node_encoding_dim = node_encoding_dim
        self.encoder_params = encoder_params
        if self.encoder_params.gnn_type == GNNEncoderParams.GNNType.GCN:
            gnn_layer_ctor = lambda: tgnn.GCNConv(
                in_channels=self.node_encoding_dim,
                out_channels=self.node_encoding_dim,
                improved=True, cached=False,
                normalize=True, add_self_loops=True)
        elif encoder_params.gnn_type == GNNEncoderParams.GNNType.GGNN:
            gnn_layer_ctor = lambda: tgnn.GatedGraphConv(
                out_channels=self.node_encoding_dim,
                num_layers=self.encoder_params.nr_layers)
        elif encoder_params.gnn_type == GNNEncoderParams.GNNType.TransformerConv:
            gnn_layer_ctor = lambda: tgnn.TransformerConv(
                in_channels=self.node_encoding_dim,
                out_channels=self.node_encoding_dim,)
        elif encoder_params.gnn_type == GNNEncoderParams.GNNType.GAT:
            gnn_layer_ctor = lambda: tgnn.GATConv(
                in_channels=self.node_encoding_dim,
                out_channels=self.node_encoding_dim, )
        elif encoder_params.gnn_type == GNNEncoderParams.GNNType.GATv2:
            gnn_layer_ctor = lambda: tgnn.GATConv(
                in_channels=self.node_encoding_dim,
                out_channels=self.node_encoding_dim, )
        else:
            raise ValueError(f'Unsupported GNN type {self.encoder_params.gnn_type}')
        self_unrolling_gnn_types = {GNNEncoderParams.GNNType.GGNN}
        self.nr_layers_to_apply = 1 \
            if encoder_params.gnn_type in self_unrolling_gnn_types \
            else self.encoder_params.nr_layers
        self.gnn_layers = nn.ModuleList([
            gnn_layer_ctor() for _ in range(self.nr_layers_to_apply)])
        self.edge_types_embeddings = nn.Embedding(
            num_embeddings=len(edge_types_vocab),
            embedding_dim=self.node_encoding_dim,
            padding_idx=edge_types_vocab.get_word_idx('<PAD>')) if edge_types_vocab else None
        self.nodes_encodings_norms = None
        if norm_params is not None:
            if self.encoder_params.apply_norm_after_each_layer:
                self.nodes_encodings_norms = nn.ModuleList([
                    NormWrapper(nr_features=node_encoding_dim, params=norm_params)
                    for _ in range(self.nr_layers_to_apply)])
            else:
                self.nodes_encodings_norms = NormWrapper(nr_features=node_encoding_dim, params=norm_params)

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, nodes_encodings: torch.Tensor, graph: TGData):
        # edge_weight = self.edge_types_embeddings(graph.edge_attr)
        for gnn_layer_idx in range(self.nr_layers_to_apply):
            nodes_encodings = self.dropout_layer(self.activation_layer(self.gnn_layers[gnn_layer_idx](
                x=nodes_encodings,
                edge_index=graph.edge_index,
                # edge_weight=edge_weight
            )))
            if self.nodes_encodings_norms and self.encoder_params.apply_norm_after_each_layer:
                nodes_encodings = self.nodes_encodings_norms[gnn_layer_idx](nodes_encodings)
        if self.nodes_encodings_norms and not self.encoder_params.apply_norm_after_each_layer:
            nodes_encodings = self.nodes_encodings_norms(nodes_encodings)
        return nodes_encodings
