import torch
import torch.nn as nn

from ndfa.nn_utils.sequence_encoder import SequenceEncoder
from ndfa.ndfa_model_hyper_parameters import SequenceEncoderParams


__all__ = ['CFGPathEncoder']


class CFGPathEncoder(nn.Module):
    def __init__(self, cfg_node_dim: int,
                 cfg_paths_sequence_encoder_params: SequenceEncoderParams,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGPathEncoder, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.sequence_encoder_layer = SequenceEncoder(
            encoder_params=cfg_paths_sequence_encoder_params,
            input_dim=self.cfg_node_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(self, cfg_nodes_encodings: torch.Tensor,
                cfg_paths_nodes_indices: torch.LongTensor,
                cfg_paths_lengths: torch.LongTensor):
        cfg_paths_nodes_embeddings = cfg_nodes_encodings[cfg_paths_nodes_indices]
        # TODO: should we zero the unnecessary encodings (using `cfg_paths_lengths`)?
        #  because these places are filled with index 0 and will end-up having the encoding of the first node.
        paths_encodings = self.sequence_encoder_layer(
            sequence_input=cfg_paths_nodes_embeddings,
            lengths=cfg_paths_lengths, batch_first=True).sequence
        return paths_encodings
