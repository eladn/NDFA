import torch
import torch.nn as nn

from ndfa.nn_utils.rnn_encoder import RNNEncoder


__all__ = ['CFGPathEncoder']


class CFGPathEncoder(nn.Module):
    def __init__(self, cfg_node_dim: int):
        super(CFGPathEncoder, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.rnn_encoder_layer = RNNEncoder(input_dim=self.cfg_node_dim)

    def forward(self, cfg_nodes_encodings: torch.Tensor,
                cfg_paths_nodes_indices: torch.LongTensor,
                cfg_paths_lengths: torch.LongTensor):
        cfg_paths_nodes_embeddings = cfg_nodes_encodings[cfg_paths_nodes_indices]
        # TODO: should we zero the unnecessary encodings (using `cfg_paths_lengths`)?
        #  because these places are filled with index 0 and will end-up having the encoding of the first node.
        _, rnn_outputs = self.rnn_encoder_layer(
            sequence_input=cfg_paths_nodes_embeddings, lengths=cfg_paths_lengths, batch_first=True)
        return rnn_outputs
