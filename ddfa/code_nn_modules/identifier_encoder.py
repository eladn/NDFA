import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Union, Optional

from ddfa.code_nn_modules.vocabulary import Vocabulary


class IdentifierEncoder(nn.Module):
    def __init__(self, sub_identifiers_vocab: Vocabulary, method: str = 'bi-lstm', embedding_dim: int = 256,
                 nr_rnn_layers: int = 2, apply_attn: bool = True):
        assert method in {'bi-lstm', 'transformer_encoder'}
        self.method = method
        super(IdentifierEncoder, self).__init__()
        self.sub_identifiers_vocab = sub_identifiers_vocab
        self.embedding_dim = embedding_dim
        self.apply_attn = apply_attn
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
            self.nr_rnn_layers = nr_rnn_layers
            self.lstm_layer = nn.LSTM(
                self.embedding_dim, self.embedding_dim, bidirectional=True, num_layers=self.nr_rnn_layers)
            self.key_linear_projection_layer = nn.Linear(
                in_features=self.embedding_dim, out_features=self.embedding_dim)

    def forward(self, sub_identifiers_indices: Union[torch.Tensor, nn.utils.rnn.PackedSequence],
                sub_identifiers_mask: Optional[torch.BoolTensor] = None):
        assert isinstance(sub_identifiers_indices, torch.Tensor)
        assert sub_identifiers_indices.dtype == torch.long
        assert len(sub_identifiers_indices.size()) == 3
        assert sub_identifiers_mask is None or (sub_identifiers_mask.dtype == torch.bool and
                                                sub_identifiers_mask.size() == sub_identifiers_indices.size())
        batch_size, nr_identifiers_in_example, nr_sub_identifiers_in_identifier = sub_identifiers_indices.size()

        sub_identifiers_indices = sub_identifiers_indices.flatten(0, 1)\
            .permute(1, 0)  # (nr_sub_identifiers, bsz * nr_identifiers_in_example)
        assert sub_identifiers_indices.size() == \
               (nr_sub_identifiers_in_identifier, batch_size * nr_identifiers_in_example)
        sub_identifiers_embeddings = self.sub_identifiers_embedding_layer(
            sub_identifiers_indices)  # (nr_sub_identifiers, bsz * nr_identifiers_in_example, embedding_dim)
        assert sub_identifiers_embeddings.size() == \
               (nr_sub_identifiers_in_identifier, batch_size * nr_identifiers_in_example, self.embedding_dim)

        if self.method == 'transformer_encoder':
            if sub_identifiers_mask is not None:
                sub_identifiers_mask = ~sub_identifiers_mask.flatten(0, 1)  # (bsz * nr_identifiers_in_example, nr_sub_identifiers)
            sub_identifiers_encoded = self.transformer_encoder(
                sub_identifiers_embeddings, src_key_padding_mask=sub_identifiers_mask).sum(dim=0)  # (bsz * nr_identifiers_in_example, embedding_dim)
            assert sub_identifiers_encoded.size() == (batch_size * nr_identifiers_in_example, self.embedding_dim)
            return sub_identifiers_encoded.view(batch_size, nr_identifiers_in_example, self.embedding_dim)

        assert self.method == 'bi-lstm'
        sub_identifiers_mask = None if sub_identifiers_mask is None else sub_identifiers_mask.flatten(0, 1)  # (bsz * nr_identifiers_in_example, nr_sub_identifiers)
        lengths = None if sub_identifiers_mask is None else sub_identifiers_mask.long().sum(dim=1)
        lengths = torch.where(lengths <= torch.zeros(1, dtype=torch.long, device=lengths.device),
                              torch.ones(1, dtype=torch.long, device=lengths.device), lengths)
        packed_input = pack_padded_sequence(sub_identifiers_embeddings, lengths=lengths, enforce_sorted=False)
        sub_identifiers_rnn_outputs, (last_hidden_out, _) = self.lstm_layer(packed_input)
        assert last_hidden_out.size() == \
               (self.nr_rnn_layers * 2, batch_size * nr_identifiers_in_example, self.embedding_dim)
        last_hidden_out = last_hidden_out\
            .view(self.nr_rnn_layers, 2, batch_size * nr_identifiers_in_example, self.embedding_dim)[-1, :, :, :]\
            .squeeze(0).sum(dim=0)
        assert last_hidden_out.size() == (batch_size * nr_identifiers_in_example, self.embedding_dim)

        # Apply attention over encoded sub-identifiers using the last hidden state as the attn-key.
        if self.apply_attn:
            sub_identifiers_attn_key_vector = F.relu(self.key_linear_projection_layer(last_hidden_out))  # (bsz * nr_identifiers_in_example, embedding_dim)
            sub_identifiers_rnn_outputs, _ = pad_packed_sequence(sequence=sub_identifiers_rnn_outputs)
            max_nr_sub_identifiers = sub_identifiers_rnn_outputs.size()[0]
            assert sub_identifiers_rnn_outputs.size() == \
                   (max_nr_sub_identifiers, batch_size * nr_identifiers_in_example, 2 * self.embedding_dim)
            sub_identifiers_rnn_outputs = sub_identifiers_rnn_outputs\
                .view(max_nr_sub_identifiers, batch_size * nr_identifiers_in_example, 2, self.embedding_dim).sum(dim=-2)
            assert sub_identifiers_rnn_outputs.size() == \
                   (max_nr_sub_identifiers, batch_size * nr_identifiers_in_example, self.embedding_dim)
            attn_weights = torch.bmm(
                sub_identifiers_rnn_outputs.flatten(0, 1)
                    .unsqueeze(dim=1),  # (max_nr_sub_identifiers * bsz * nr_identifiers_in_example, 1, embedding_dim)
                sub_identifiers_attn_key_vector.unsqueeze(dim=0)
                    .expand(max_nr_sub_identifiers, batch_size * nr_identifiers_in_example, self.embedding_dim)
                    .flatten(0, 1).unsqueeze(dim=-1))  # (max_nr_sub_identifiers * bsz * nr_identifiers_in_example, embedding_dim, 1)
            assert attn_weights.size() == (max_nr_sub_identifiers * batch_size * nr_identifiers_in_example, 1, 1)
            attn_weights = attn_weights.view(max_nr_sub_identifiers, batch_size * nr_identifiers_in_example).permute(1, 0)  # (batch_size * nr_identifiers_in_example, max_nr_sub_identifiers)
            if sub_identifiers_mask is not None:
                # sub_identifiers_mask = sub_identifiers_mask[:, :max_nr_sub_identifiers]
                attn_weights = attn_weights + torch.where(
                    sub_identifiers_mask,  # (bsz * nr_identifiers_in_example, nr_sub_identifiers)
                    torch.zeros(1, dtype=torch.float, device=attn_weights.device),
                    torch.full(size=(1,), fill_value=float('-inf'), dtype=torch.float, device=attn_weights.device))
            attn_probs = F.softmax(attn_weights, dim=1)  # (bsz * nr_identifiers_in_example, max_nr_sub_identifiers)
            # (bsz * nr_identifiers_in_example, 1, max_nr_sub_identifiers) * (bsz * nr_identifiers_in_example, max_nr_sub_identifiers, embedding_dim)
            # = (bsz * nr_identifiers_in_example, 1, embedding_dim)
            attn_applied = torch.bmm(
                attn_probs.unsqueeze(1), sub_identifiers_rnn_outputs.permute(1, 0, 2)).squeeze(1)  # (bsz * nr_identifiers_in_example, embedding_dim)
            identifiers_encoded = attn_applied.view(batch_size, nr_identifiers_in_example, self.embedding_dim)
        else:
            identifiers_encoded = last_hidden_out.view(batch_size, nr_identifiers_in_example, self.embedding_dim)

        return identifiers_encoded

        # TODO: fix this implementation. it is currently not correct.
        # assert isinstance(sub_identifiers_indices, list)
        # assert all(isinstance(s, list) for s in sub_identifiers_indices)
        # packed_sub_identifiers = nn.utils.rnn.pack_sequence(sub_identifiers_indices, enforce_sorted=False)
        # return self.lstm_layer(packed_sub_identifiers)[1][0]
