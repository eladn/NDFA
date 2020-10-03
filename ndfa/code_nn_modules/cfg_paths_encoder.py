import torch
import torch.nn as nn

from ndfa.nn_utils.sequence_encoder import SequenceEncoder
from ndfa.ndfa_model_hyper_parameters import SequenceEncoderParams
from ndfa.code_nn_modules.vocabulary import Vocabulary


__all__ = ['CFGPathEncoder']


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

    def forward(self, cfg_nodes_encodings: torch.Tensor,
                cfg_paths_nodes_indices: torch.LongTensor,
                cfg_paths_edge_types: torch.LongTensor,
                cfg_paths_lengths: torch.LongTensor):
        assert cfg_nodes_encodings.ndim == 2
        assert cfg_nodes_encodings.size(1) == self.cfg_node_dim
        assert cfg_paths_nodes_indices.ndim == 2
        assert cfg_paths_edge_types.shape == cfg_paths_edge_types.shape

        cfg_paths_nodes_embeddings = cfg_nodes_encodings[cfg_paths_nodes_indices]
        cfg_paths_edge_types_embeddings = self.control_flow_edge_types_embeddings(cfg_paths_edge_types)
        assert cfg_paths_nodes_embeddings.shape == cfg_paths_edge_types_embeddings.shape

        # weave nodes & edge-types in each path
        cfg_paths_interwoven_nodes_and_edge_types_embeddings = \
            torch.stack([cfg_paths_nodes_embeddings, cfg_paths_edge_types_embeddings], dim=1)\
                .permute((0, 2, 1, 3)).flatten(1, 2)
        assert cfg_paths_interwoven_nodes_and_edge_types_embeddings.shape == \
            (cfg_paths_nodes_embeddings.size(0), 2 * cfg_paths_nodes_embeddings.size(1), self.cfg_node_dim)

        # TODO: should we zero the unnecessary encodings (using `cfg_paths_lengths`)?
        #  because these places are filled with index 0 and will end-up having the encoding of the first node.

        paths_encodings = self.sequence_encoder_layer(
            sequence_input=cfg_paths_interwoven_nodes_and_edge_types_embeddings,
            lengths=cfg_paths_lengths * 2, batch_first=True).sequence

        # remove edge types embeddings
        paths_encodings_wo_edges = \
            paths_encodings.view(paths_encodings.size(0), paths_encodings.size(1) // 2, 2, paths_encodings.size(2))[:, :, 0, :]
        assert paths_encodings_wo_edges.shape == cfg_paths_nodes_embeddings.shape

        return paths_encodings_wo_edges
