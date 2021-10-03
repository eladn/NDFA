import torch
import torch.nn as nn
import dataclasses
from typing import Optional

from ndfa.nn_utils.modules.params.paths_encoder_params import EdgeTypeInsertionMode
from ndfa.nn_utils.misc.misc import seq_lengths_to_mask
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.functions.weave_tensors import weave_tensors, unweave_tensor


__all__ = ['PathsEncoder', 'EncodedPaths', 'EdgeTypeInsertionMode']


@dataclasses.dataclass
class EncodedPaths:
    nodes_occurrences: torch.Tensor
    edges_occurrences: Optional[torch.Tensor] = None


class PathsEncoder(nn.Module):
    def __init__(
            self,
            node_dim: int,
            paths_sequence_encoder_params: SequenceEncoderParams,
            edge_types_vocab: Vocabulary,
            edge_type_insertion_mode: EdgeTypeInsertionMode = EdgeTypeInsertionMode.AsStandAlongToken,
            edge_type_dim: Optional[int] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(PathsEncoder, self).__init__()
        self.node_dim = node_dim
        self.edge_type_insertion_mode = edge_type_insertion_mode
        if self.edge_type_insertion_mode in \
                {EdgeTypeInsertionMode.AsStandAlongToken, EdgeTypeInsertionMode.MixWithNodeEmbedding}:
            self.edge_type_dim = self.node_dim if edge_type_dim is None else edge_type_dim
            if self.edge_type_insertion_mode == EdgeTypeInsertionMode.AsStandAlongToken and \
                    self.edge_type_dim != self.node_dim:
                raise ValueError(f'For `EdgeTypeInsertionMode.AsStandAlongToken` mode the edge type dim should be '
                                 f'not set or set as equal to the node dim.')
            self.edge_types_vocab = edge_types_vocab
            self.edge_types_embeddings = nn.Embedding(
                num_embeddings=len(self.edge_types_vocab),
                embedding_dim=self.edge_type_dim,
                padding_idx=self.edge_types_vocab.get_word_idx('<PAD>'))
            if self.edge_type_insertion_mode == EdgeTypeInsertionMode.MixWithNodeEmbedding:
                self.node_with_edge_projection = nn.Linear(
                    in_features=self.node_dim + self.edge_type_dim,
                    out_features=self.node_dim,
                    bias=False)
        else:
            assert self.edge_type_insertion_mode == EdgeTypeInsertionMode.Without
            self.edge_type_dim = None
        self.sequence_encoder_layer = SequenceEncoder(
            encoder_params=paths_sequence_encoder_params,
            input_dim=self.node_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(
            self,
            all_nodes_encodings: torch.Tensor,
            paths_nodes_indices: torch.LongTensor,
            paths_edge_types: torch.LongTensor,
            paths_lengths: torch.LongTensor) -> EncodedPaths:
        assert all_nodes_encodings.ndim == 2
        assert all_nodes_encodings.size(1) == self.node_dim
        assert paths_nodes_indices.ndim == 2
        assert paths_edge_types.shape == paths_edge_types.shape

        paths_nodes_embeddings = all_nodes_encodings[paths_nodes_indices]

        if self.edge_type_insertion_mode == EdgeTypeInsertionMode.AsStandAlongToken:
            paths_edge_types_embeddings = self.edge_types_embeddings(paths_edge_types)
            paths_edge_types_embeddings = self.dropout_layer(paths_edge_types_embeddings)
            assert paths_nodes_embeddings.shape == paths_edge_types_embeddings.shape

            paths_effective_lengths = paths_lengths * 2
            max_path_effective_len = paths_nodes_embeddings.size(1) * 2

            # weave nodes & edge-types in each path
            paths_effective_embeddings = weave_tensors(
                tensors=[paths_nodes_embeddings, paths_edge_types_embeddings], dim=1)
            assert paths_effective_embeddings.shape == \
                (paths_nodes_embeddings.size(0), max_path_effective_len, self.node_dim)
        elif self.edge_type_insertion_mode == EdgeTypeInsertionMode.MixWithNodeEmbedding:
            paths_edge_types_embeddings = self.edge_types_embeddings(paths_edge_types)
            paths_edge_types_embeddings = self.dropout_layer(paths_edge_types_embeddings)
            assert paths_nodes_embeddings.shape == paths_edge_types_embeddings.shape
            paths_effective_embeddings = self.node_with_edge_projection(torch.cat(
                [paths_nodes_embeddings, paths_edge_types_embeddings], dim=2))
            assert paths_effective_embeddings.shape == paths_nodes_embeddings.shape
            paths_effective_lengths = paths_lengths
            max_path_effective_len = paths_nodes_embeddings.size(1)
        else:
            paths_effective_embeddings = paths_nodes_embeddings
            paths_effective_lengths = paths_lengths
            max_path_effective_len = paths_nodes_embeddings.size(1)

        # masking: replace incorrect tails with paddings
        paths_mask = seq_lengths_to_mask(seq_lengths=paths_effective_lengths, max_seq_len=max_path_effective_len)
        paths_mask_expanded = paths_mask.unsqueeze(-1).expand(paths_effective_embeddings.size())
        paths_effective_embeddings = paths_effective_embeddings.masked_fill(~paths_mask_expanded, 0)

        paths_encodings = self.sequence_encoder_layer(
            sequence_input=paths_effective_embeddings, lengths=paths_effective_lengths, batch_first=True).sequence

        if self.edge_type_insertion_mode == EdgeTypeInsertionMode.AsStandAlongToken:
            # separate nodes encodings and edge types embeddings from paths
            nodes_occurrences_encodings, edges_occurrences_encodings = unweave_tensor(
                woven_tensor=paths_encodings, dim=1, nr_target_tensors=2)
            assert nodes_occurrences_encodings.shape == edges_occurrences_encodings.shape == \
                   paths_nodes_embeddings.shape
        elif self.edge_type_insertion_mode in \
                {EdgeTypeInsertionMode.Without, EdgeTypeInsertionMode.MixWithNodeEmbedding}:
            nodes_occurrences_encodings = paths_encodings
            edges_occurrences_encodings = None
        else:
            assert False

        return EncodedPaths(
            nodes_occurrences=nodes_occurrences_encodings,
            edges_occurrences=edges_occurrences_encodings)
