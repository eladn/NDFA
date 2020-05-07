import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Union, Optional

from ddfa.code_nn_modules.vocabulary import Vocabulary


class IdentifierEncoder(nn.Module):
    def __init__(self, sub_identifiers_vocab: Vocabulary, method: str = 'bi-lstm', embedding_dim: int = 256):
        assert method in {'bi-lstm', 'transformer_encoder'}
        self.method = method
        super(IdentifierEncoder, self).__init__()
        self.sub_identifiers_vocab = sub_identifiers_vocab
        self.embedding_dim = embedding_dim
        self.sub_identifiers_embedding_layer = nn.Embedding(
            num_embeddings=len(sub_identifiers_vocab), embedding_dim=self.embedding_dim,
            padding_idx=sub_identifiers_vocab.get_word_idx_or_unk('<PAD>'))

        if method == 'transformer_encoder':
            transformer_encoder_layer = TransformerEncoderLayer(
                d_model=self.embedding_dim, nhead=1, dim_feedforward=1028)
            encoder_norm = LayerNorm(self.embedding_dim)
            self.transformer_encoder = TransformerEncoder(
                encoder_layer=transformer_encoder_layer, num_layers=3, norm=encoder_norm)
        elif method == 'bi-lstm':
            self.lstm_layer = nn.LSTM(self.embedding_dim, self.embedding_dim, bidirectional=True, num_layers=2)

    def forward(self, sub_identifiers_indices: Union[torch.Tensor, nn.utils.rnn.PackedSequence],
                sub_identifiers_mask: Optional[torch.BoolTensor] = None):
        assert sub_identifiers_indices.dtype == torch.long
        assert sub_identifiers_mask is None or (sub_identifiers_mask.dtype == torch.bool and
                                                sub_identifiers_mask.size() == sub_identifiers_indices.size())

        assert isinstance(sub_identifiers_indices, torch.Tensor)
        assert len(sub_identifiers_indices.size()) == 3
        batch_size, nr_identifiers_in_example, nr_sub_identifiers_in_identifier = sub_identifiers_indices.size()
        sub_identifiers_indices = sub_identifiers_indices.flatten(0, 1)\
            .permute(1, 0)  # (nr_sub_identifiers, bs*nr_identifiers)
        assert sub_identifiers_indices.size() == \
               (nr_sub_identifiers_in_identifier, batch_size * nr_identifiers_in_example)
        sub_identifiers_embeddings = self.sub_identifiers_embedding_layer(
            sub_identifiers_indices)  # (nr_sub_identifiers, bs*nr_identifiers, embedding_dim)
        assert sub_identifiers_embeddings.size() == \
               (nr_sub_identifiers_in_identifier, batch_size * nr_identifiers_in_example, self.embedding_dim)

        if self.method == 'transformer_encoder':
            if sub_identifiers_mask is not None:
                sub_identifiers_mask = ~sub_identifiers_mask.flatten(0, 1)  # (bs*nr_identifiers, nr_sub_identifiers)
            sub_identifiers_encoded = self.transformer_encoder(
                sub_identifiers_embeddings, src_key_padding_mask=sub_identifiers_mask).sum(dim=0)  # (bs*nr_identifiers, embedding_dim)
            assert sub_identifiers_encoded.size() == (batch_size * nr_identifiers_in_example, self.embedding_dim)
            return sub_identifiers_encoded.view(batch_size, nr_identifiers_in_example, self.embedding_dim)

        assert self.method == 'bi-lstm'
        lengths = None if sub_identifiers_mask is None else sub_identifiers_mask.flatten(0, 1).long().sum(dim=1)
        packed_input = pack_padded_sequence(sub_identifiers_embeddings, lengths=lengths, enforce_sorted=False)
        _, (last_hidden_out, _) = self.lstm_layer(packed_input)
        assert last_hidden_out.size() == (2 * 2, batch_size * nr_identifiers_in_example, self.embedding_dim)
        last_hidden_out = last_hidden_out.view(2, 2, batch_size * nr_identifiers_in_example, self.embedding_dim)[-1, :, :, :] \
            .squeeze(0).sum(dim=0)
        assert last_hidden_out.size() == (batch_size * nr_identifiers_in_example, self.embedding_dim)
        identifiers_encoded = last_hidden_out.view(batch_size, nr_identifiers_in_example, self.embedding_dim)
        return identifiers_encoded

        # TODO: fix this implementation. it is currently not correct.
        # assert isinstance(sub_identifiers_indices, list)
        # assert all(isinstance(s, list) for s in sub_identifiers_indices)
        # packed_sub_identifiers = nn.utils.rnn.pack_sequence(sub_identifiers_indices, enforce_sorted=False)
        # return self.lstm_layer(packed_sub_identifiers)[1][0]
