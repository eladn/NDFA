import torch
import torch.nn as nn
import dataclasses

from ndfa.nn_utils.misc.misc import seq_lengths_to_mask
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.functions.weave_tensors import weave_tensors, unweave_tensor


__all__ = ['CFGPathEncoder', 'EncodedCFGPaths']


@dataclasses.dataclass
class EncodedCFGPaths:
    nodes_occurrences: torch.Tensor
    edges_occurrences: torch.Tensor


class CFGPathEncoder(nn.Module):
    def __init__(self, cfg_node_dim: int,
                 cfg_paths_sequence_encoder_params: SequenceEncoderParams,
                 control_flow_edge_types_vocab: Vocabulary,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGPathEncoder, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.control_flow_edge_types_vocab = control_flow_edge_types_vocab
        self.control_flow_edge_types_embeddings = nn.Embedding(
            num_embeddings=len(self.control_flow_edge_types_vocab),
            embedding_dim=self.cfg_node_dim,
            padding_idx=self.control_flow_edge_types_vocab.get_word_idx('<PAD>'))
        self.sequence_encoder_layer = SequenceEncoder(
            encoder_params=cfg_paths_sequence_encoder_params,
            input_dim=self.cfg_node_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, cfg_nodes_encodings: torch.Tensor,
                cfg_paths_nodes_indices: torch.LongTensor,
                cfg_paths_edge_types: torch.LongTensor,
                cfg_paths_lengths: torch.LongTensor) -> EncodedCFGPaths:
        assert cfg_nodes_encodings.ndim == 2
        assert cfg_nodes_encodings.size(1) == self.cfg_node_dim
        assert cfg_paths_nodes_indices.ndim == 2
        assert cfg_paths_edge_types.shape == cfg_paths_edge_types.shape

        cfg_paths_nodes_embeddings = cfg_nodes_encodings[cfg_paths_nodes_indices]
        cfg_paths_edge_types_embeddings = self.control_flow_edge_types_embeddings(cfg_paths_edge_types)
        cfg_paths_edge_types_embeddings = self.dropout_layer(cfg_paths_edge_types_embeddings)
        assert cfg_paths_nodes_embeddings.shape == cfg_paths_edge_types_embeddings.shape

        # weave nodes & edge-types in each path
        cfg_paths_interwoven_nodes_and_edge_types_embeddings = weave_tensors(
            tensors=[cfg_paths_nodes_embeddings, cfg_paths_edge_types_embeddings], dim=1)
        assert cfg_paths_interwoven_nodes_and_edge_types_embeddings.shape == \
            (cfg_paths_nodes_embeddings.size(0), 2 * cfg_paths_nodes_embeddings.size(1), self.cfg_node_dim)

        # mask-out the paths (remove paddings)
        paths_with_edges_lengths = cfg_paths_lengths * 2
        paths_with_edges_mask = seq_lengths_to_mask(
            seq_lengths=paths_with_edges_lengths, max_seq_len=2 * cfg_paths_nodes_embeddings.size(1))
        paths_with_edges_mask_expanded = \
            paths_with_edges_mask.unsqueeze(-1).expand(cfg_paths_interwoven_nodes_and_edge_types_embeddings.size())
        cfg_paths_interwoven_nodes_and_edge_types_embeddings = \
            cfg_paths_interwoven_nodes_and_edge_types_embeddings.masked_fill(
                ~paths_with_edges_mask_expanded, 0)

        paths_encodings = self.sequence_encoder_layer(
            sequence_input=cfg_paths_interwoven_nodes_and_edge_types_embeddings,
            lengths=cfg_paths_lengths * 2, batch_first=True).sequence

        # separate nodes encodings and edge types embeddings from paths
        nodes_occurrences_encodings, edges_occurrences_encodings = unweave_tensor(
            woven_tensor=paths_encodings, dim=1, nr_target_tensors=2)
        assert nodes_occurrences_encodings.shape == edges_occurrences_encodings.shape == \
               cfg_paths_nodes_embeddings.shape

        return EncodedCFGPaths(
            nodes_occurrences=nodes_occurrences_encodings,
            edges_occurrences=edges_occurrences_encodings)
