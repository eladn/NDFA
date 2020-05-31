import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
from typing import Union, Optional

from ddfa.code_nn_modules.vocabulary import Vocabulary
from ddfa.nn_utils.attn_rnn_encoder import AttnRNNEncoder


class IdentifierEncoder(nn.Module):
    def __init__(self, sub_identifiers_vocab: Vocabulary, method: str = 'bi-lstm', embedding_dim: int = 256,
                 nr_rnn_layers: int = 2):
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
            self.attn_rnn_encoder = AttnRNNEncoder(
                input_dim=self.embedding_dim, hidden_dim=self.embedding_dim, rnn_type='lstm',
                nr_rnn_layers=nr_rnn_layers, rnn_bi_direction=True)

    def forward(self, sub_identifiers_indices: Union[torch.Tensor, nn.utils.rnn.PackedSequence],
                sub_identifiers_mask: Optional[torch.BoolTensor] = None):
        assert isinstance(sub_identifiers_indices, torch.Tensor)
        assert sub_identifiers_indices.dtype == torch.long
        assert len(sub_identifiers_indices.size()) == 3
        assert sub_identifiers_mask is None or (sub_identifiers_mask.dtype == torch.bool and
                                                sub_identifiers_mask.size() == sub_identifiers_indices.size())
        batch_size, nr_identifiers_in_example, nr_sub_identifiers_in_identifier = sub_identifiers_indices.size()

        sub_identifiers_indices = sub_identifiers_indices.flatten(0, 1)
        assert sub_identifiers_indices.size() == \
               (batch_size * nr_identifiers_in_example, nr_sub_identifiers_in_identifier)
        sub_identifiers_embeddings = self.sub_identifiers_embedding_layer(sub_identifiers_indices)
        assert sub_identifiers_embeddings.size() == \
               (batch_size * nr_identifiers_in_example, nr_sub_identifiers_in_identifier, self.embedding_dim)

        if self.method == 'transformer_encoder':
            sub_identifiers_embeddings_SNE = sub_identifiers_embeddings.permute(1, 0, 2)  # (nr_sub_identifiers, bsz * nr_identifiers_in_example, embedding_dim)
            if sub_identifiers_mask is not None:
                sub_identifiers_mask = ~sub_identifiers_mask.flatten(0, 1)  # (bsz * nr_identifiers_in_example, nr_sub_identifiers)
            sub_identifiers_encoded = self.transformer_encoder(
                sub_identifiers_embeddings_SNE, src_key_padding_mask=sub_identifiers_mask).sum(dim=0)  # (bsz * nr_identifiers_in_example, embedding_dim)
            assert sub_identifiers_encoded.size() == (batch_size * nr_identifiers_in_example, self.embedding_dim)
            return sub_identifiers_encoded.view(batch_size, nr_identifiers_in_example, self.embedding_dim)

        assert self.method == 'bi-lstm'
        if sub_identifiers_mask is not None:
            sub_identifiers_mask = sub_identifiers_mask.flatten(0, 1)  # (bsz * nr_identifiers_in_example, nr_sub_identifiers)
            # quick fix for padding (no identifiers there) to avoid later attn softmax of only -inf values.
            sub_identifiers_mask[:, 0] = True
        encoded_identifiers = self.attn_rnn_encoder(
            sequence_input=sub_identifiers_embeddings, mask=sub_identifiers_mask, batch_first=True)
        assert encoded_identifiers.size() == (batch_size * nr_identifiers_in_example, self.embedding_dim)
        return encoded_identifiers.view(batch_size, nr_identifiers_in_example, self.embedding_dim)
