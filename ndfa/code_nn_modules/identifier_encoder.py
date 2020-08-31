import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
from typing import Optional

from ndfa.code_nn_modules.vocabulary import Vocabulary
from ndfa.nn_utils.attn_rnn_encoder import AttnRNNEncoder
from ndfa.misc.tensors_data_class import BatchFlattenedSeq


class IdentifierEncoder(nn.Module):
    def __init__(self, sub_identifiers_vocab: Vocabulary, method: str = 'bi-lstm', embedding_dim: int = 256,
                 nr_rnn_layers: int = 2, dropout_rate: float = 0.3):
        assert method in {'bi-lstm', 'transformer_encoder'}
        self.method = method
        super(IdentifierEncoder, self).__init__()
        self.sub_identifiers_vocab = sub_identifiers_vocab
        self.embedding_dim = embedding_dim
        self.sub_identifiers_embedding_layer = nn.Embedding(
            num_embeddings=len(sub_identifiers_vocab), embedding_dim=self.embedding_dim,
            padding_idx=sub_identifiers_vocab.get_word_idx('<PAD>'))

        if method == 'transformer_encoder':
            transformer_encoder_layer = TransformerEncoderLayer(
                d_model=self.embedding_dim, nhead=1, dim_feedforward=1028)
            encoder_norm = LayerNorm(self.embedding_dim)
            self.transformer_encoder = TransformerEncoder(
                encoder_layer=transformer_encoder_layer, num_layers=3, norm=encoder_norm)
        elif method == 'bi-lstm':
            self.attn_rnn_encoder = AttnRNNEncoder(
                input_dim=self.embedding_dim, hidden_dim=self.embedding_dim, rnn_type='lstm',
                nr_rnn_layers=nr_rnn_layers, rnn_bi_direction=True)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        nr_hashing_features = 256  # TODO: plug-in HP
        self.identifier_sub_parts_hashing_linear = nn.Linear(nr_hashing_features, nr_hashing_features, bias=False)
        self.vocab_and_hashing_combiner = nn.Linear(embedding_dim + nr_hashing_features, embedding_dim)
        self.final_linear_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, identifiers_sub_parts: BatchFlattenedSeq,
                identifiers_sub_parts_hashings: Optional[BatchFlattenedSeq] = None):
        assert isinstance(identifiers_sub_parts, BatchFlattenedSeq)
        assert isinstance(identifiers_sub_parts.sequences, torch.Tensor)
        assert identifiers_sub_parts_hashings is None or \
               isinstance(identifiers_sub_parts_hashings, BatchFlattenedSeq)
        assert identifiers_sub_parts_hashings is None or \
               isinstance(identifiers_sub_parts_hashings.sequences, torch.Tensor)
        assert identifiers_sub_parts.sequences.dtype == torch.long
        assert identifiers_sub_parts.sequences.ndim == 2
        nr_identifiers_in_batch, max_nr_sub_identifiers_in_identifier = identifiers_sub_parts.sequences.size()

        identifiers_sub_parts_vocab_embeddings = self.sub_identifiers_embedding_layer(identifiers_sub_parts.sequences)
        assert identifiers_sub_parts_vocab_embeddings.size() == \
               (nr_identifiers_in_batch, max_nr_sub_identifiers_in_identifier, self.embedding_dim)
        identifiers_sub_parts_vocab_embeddings = self.dropout_layer(identifiers_sub_parts_vocab_embeddings)

        if identifiers_sub_parts_hashings is not None:
            identifiers_sub_parts_hashings_projected = self.identifier_sub_parts_hashing_linear(
                identifiers_sub_parts_hashings.sequences)
            identifiers_sub_parts_hashings_projected = self.dropout_layer(identifiers_sub_parts_hashings_projected)
            identifiers_sub_parts_vocab_embeddings_and_hashings_combined = self.vocab_and_hashing_combiner(
                torch.cat([identifiers_sub_parts_vocab_embeddings, identifiers_sub_parts_hashings_projected], dim=-1))
            identifiers_sub_parts_embeddings = self.dropout_layer(F.relu(
                identifiers_sub_parts_vocab_embeddings_and_hashings_combined))
            identifiers_sub_parts_embeddings = self.dropout_layer(F.relu(self.final_linear_layer(
                identifiers_sub_parts_embeddings)))
        else:
            identifiers_sub_parts_embeddings = identifiers_sub_parts_vocab_embeddings

        if self.method == 'transformer_encoder':
            raise NotImplementedError
            # sub_identifiers_embeddings_SNE = sub_identifiers_embeddings.permute(1, 0, 2)  # (nr_sub_identifiers, bsz * nr_identifiers_in_example, embedding_dim)
            # if sub_identifiers_mask is not None:
            #     sub_identifiers_mask = ~sub_identifiers_mask.flatten(0, 1)  # (bsz * nr_identifiers_in_example, nr_sub_identifiers)
            # sub_identifiers_encoded = self.transformer_encoder(
            #     sub_identifiers_embeddings_SNE, src_key_padding_mask=sub_identifiers_mask).sum(dim=0)  # (bsz * nr_identifiers_in_example, embedding_dim)
            # assert sub_identifiers_encoded.size() == (batch_size * nr_identifiers_in_example, self.embedding_dim)
            # return sub_identifiers_encoded.view(batch_size, nr_identifiers_in_example, self.embedding_dim)

        assert self.method == 'bi-lstm'
        encoded_identifiers, _ = self.attn_rnn_encoder(
            sequence_input=identifiers_sub_parts_embeddings,
            lengths=identifiers_sub_parts.sequences_lengths,
            batch_first=True)
        assert encoded_identifiers.size() == (nr_identifiers_in_batch, self.embedding_dim)
        return encoded_identifiers
