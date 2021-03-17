import torch
import torch.nn as nn
from torch_geometric import nn as tgnn
from typing import Optional, Callable
from torch_geometric.data import Data as TGData

from ndfa.code_nn_modules.params.cfg_gnn_encoder_params import CFGGNNEncoderParams
from ndfa.nn_utils.modules.module_repeater import ModuleRepeater
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.modules.norm_wrapper import NormWrapper


__all__ = ['CFGGNNEncoder']


class CFGGNNEncoder(nn.Module):
    def __init__(
            self,
            cfg_node_encoding_dim: int,
            encoder_params: CFGGNNEncoderParams,
            pdg_control_flow_edge_types_vocab: Vocabulary,
            norm_layer_ctor: Optional[Callable[[], NormWrapper]],  # TODO: get `NormWrapperParams` instead
            share_norm_between_usage_points: bool = False,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGGNNEncoder, self).__init__()
        self.cfg_node_encoding_dim = cfg_node_encoding_dim
        self.encoder_params = encoder_params
        if self.encoder_params.gnn_type == 'gcn':
            gnn_layer_ctor = lambda: tgnn.GCNConv(
                in_channels=self.cfg_node_encoding_dim,
                out_channels=self.cfg_node_encoding_dim,
                cached=False, normalize=True, add_self_loops=True)
        elif encoder_params.gnn_type == 'ggnn':
            gnn_layer_ctor = lambda: tgnn.GatedGraphConv(
                out_channels=self.cfg_node_encoding_dim,
                num_layers=self.encoder_params.nr_layers)
        elif encoder_params.gnn_type == 'transformer_conv':
            gnn_layer_ctor = lambda: tgnn.TransformerConv(
                in_channels=self.cfg_node_encoding_dim,
                out_channels=self.cfg_node_encoding_dim,)
        else:
            raise ValueError(f'Unsupported GNN type {self.encoder_params.gnn_type}')
        self_unrolling_gnn_types = {'ggnn'}
        self.nr_layers_to_apply = 1 \
            if encoder_params.gnn_type in self_unrolling_gnn_types \
            else self.encoder_params.nr_layers
        self.cfg_gnn = ModuleRepeater(
            gnn_layer_ctor,
            repeats=self.nr_layers_to_apply,
            share=False, repeat_key='gnn_layer_idx')
        self.control_flow_edge_types_embeddings = nn.Embedding(
            num_embeddings=len(pdg_control_flow_edge_types_vocab),
            embedding_dim=self.cfg_node_encoding_dim,
            padding_idx=pdg_control_flow_edge_types_vocab.get_word_idx('<PAD>'))
        self.cfg_nodes_norm = None
        if norm_layer_ctor is not None:
            self.cfg_nodes_norm = ModuleRepeater(
                module_create_fn=norm_layer_ctor,
                repeats=self.nr_layers_to_apply,
                share=share_norm_between_usage_points, repeat_key='usage_point')

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, encoded_cfg_nodes: torch.Tensor, cfg_control_flow_graph: TGData):
        # edge_weight = self.control_flow_edge_types_embeddings(
        #     code_task_input.pdg.cfg_control_flow_graph.edge_attr)
        for gnn_layer_idx in range(self.nr_layers_to_apply):
            encoded_cfg_nodes = self.dropout_layer(self.activation_layer(self.cfg_gnn(
                x=encoded_cfg_nodes,
                edge_index=cfg_control_flow_graph.edge_index,  # edge_weight=edge_weight,
                gnn_layer_idx=gnn_layer_idx)))
            if self.cfg_nodes_norm:
                encoded_cfg_nodes = self.cfg_nodes_norm(encoded_cfg_nodes, usage_point=gnn_layer_idx)
        return encoded_cfg_nodes
