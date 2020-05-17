import torch
import torch.nn as nn
from functools import reduce
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ddfa.code_nn_modules.vocabulary import Vocabulary
from ddfa.code_data_structure_api import *
from ddfa.nn_utils.apply_batched_embeddings import apply_batched_embeddings
from ddfa.nn_utils.attention import Attention


class ExpressionEncoder(nn.Module):
    def __init__(self, tokens_vocab: Vocabulary, tokens_kinds_vocab: Vocabulary, tokens_embedding_dim: int = 256,
                 expr_encoding_dim: int = 1028, token_kind_embedding_dim: int = 4,
                 method: str = 'bi-lstm', apply_attn: bool = True):
        assert method in {'bi-lstm', 'transformer_encoder'}
        super(ExpressionEncoder, self).__init__()
        self.tokens_vocab = tokens_vocab
        self.tokens_kinds_vocab = tokens_kinds_vocab
        self.tokens_embedding_dim = tokens_embedding_dim
        self.expr_encoding_dim = expr_encoding_dim
        self.method = method
        self.apply_attn = apply_attn
        self.tokens_embedding_layer = nn.Embedding(
            num_embeddings=len(tokens_vocab), embedding_dim=self.tokens_embedding_dim,
            padding_idx=tokens_vocab.get_word_idx_or_unk('<PAD>'))
        self.token_kind_embedding_dim = token_kind_embedding_dim
        self.tokens_kinds_embedding_layer = nn.Embedding(
            num_embeddings=len(tokens_kinds_vocab), embedding_dim=self.token_kind_embedding_dim,
            padding_idx=tokens_kinds_vocab.get_word_idx_or_unk('<PAD>'))

        self.projection_linear_layer = nn.Linear(
            self.tokens_embedding_dim + self.token_kind_embedding_dim, self.expr_encoding_dim)

        if method == 'transformer_encoder':
            transformer_encoder_layer = TransformerEncoderLayer(
                d_model=self.expr_encoding_dim, nhead=1)
            encoder_norm = LayerNorm(self.expr_encoding_dim)
            self.transformer_encoder = TransformerEncoder(
                encoder_layer=transformer_encoder_layer, num_layers=3, norm=encoder_norm)
        elif method == 'bi-lstm':
            self.lstm_layer = nn.LSTM(self.expr_encoding_dim, self.expr_encoding_dim, bidirectional=True, num_layers=2)
            if self.apply_attn:
                self.attn_layer = Attention(nr_features=self.expr_encoding_dim, project_key=True)

    def forward(self, expressions: torch.Tensor, expressions_mask: Optional[torch.BoolTensor],
                encoded_identifiers: torch.Tensor):  # Union[torch.Tensor, nn.utils.rnn.PackedSequence]
        assert len(expressions.size()) == 4 and expressions.size()[-1] == 2
        batch_size, nr_exprs, nr_tokens_in_expr, _ = expressions.size()
        assert len(encoded_identifiers.size()) == 3
        nr_identifiers_in_example = encoded_identifiers.size()[1]
        assert encoded_identifiers.size() == (batch_size, nr_identifiers_in_example, self.tokens_embedding_dim)
        assert expressions_mask is None or expressions_mask.size() == (batch_size, nr_exprs, nr_tokens_in_expr)

        expressions_tokens_kinds = expressions[:, :, :, 0]  # (batch_size, nr_exprs, nr_tokens_in_expr)
        expressions_idxs = expressions[:, :, :, 1]  # (batch_size, nr_exprs, nr_tokens_in_expr)
        assert expressions_tokens_kinds.size() == expressions_idxs.size() == (batch_size, nr_exprs, nr_tokens_in_expr)

        use_identifier_vocab_condition = expressions_tokens_kinds == self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.IDENTIFIER.value)
        identifiers_idxs = torch.where(
            use_identifier_vocab_condition,
            expressions_idxs, torch.zeros_like(expressions_idxs))  # (batch_size, nr_exprs, nr_tokens_in_expr)
        assert identifiers_idxs.size() == expressions.size()[:-1]

        token_kinds_for_tokens_vocab = (
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.OPERATOR.value),
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.SEPARATOR.value),
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.KEYWORD.value))
        use_tokens_vocab_condition = reduce(
            torch.Tensor.__or__, (expressions_tokens_kinds == token_kind for token_kind in token_kinds_for_tokens_vocab))
        tokens_idxs = torch.where(
            use_tokens_vocab_condition,
            expressions_idxs, torch.zeros_like(expressions_idxs))  # (batch_size, nr_exprs, nr_tokens_in_expr)
        assert tokens_idxs.size() == expressions.size()[:-1]

        selected_tokens_encoding = self.tokens_embedding_layer(tokens_idxs.flatten())\
            .view(batch_size, nr_exprs, nr_tokens_in_expr, self.tokens_embedding_dim)

        selected_encoded_identifiers = apply_batched_embeddings(
            batched_embeddings=encoded_identifiers, indices=identifiers_idxs)
        assert selected_encoded_identifiers.size() == identifiers_idxs.size() + (encoded_identifiers.size()[-1],)

        # TODO: we could treat the tokens-embedding & identifiers-encodings as PERPETUAL to each others (concat - other dims for each)
        assert selected_encoded_identifiers.size() == selected_tokens_encoding.size()
        use_identifier_vocab_condition_expanded_to_encodings = use_identifier_vocab_condition.unsqueeze(-1)\
            .expand_as(selected_encoded_identifiers)
        embeddings = torch.where(
            use_identifier_vocab_condition_expanded_to_encodings,
            selected_encoded_identifiers, torch.where(
                use_identifier_vocab_condition_expanded_to_encodings,
                selected_tokens_encoding, torch.zeros_like(selected_tokens_encoding)))  # (batch_size, nr_exprs, nr_tokens_in_expr, embedding_dim)
        assert embeddings.size() == (batch_size, nr_exprs, nr_tokens_in_expr, self.tokens_embedding_dim)

        token_kinds_embeddings = self.tokens_kinds_embedding_layer(expressions_tokens_kinds.flatten())\
            .view(expressions_tokens_kinds.size() + (self.token_kind_embedding_dim,))  # (batch_size, nr_exprs, nr_tokens_in_expr, token_kind_embedding_dim)

        expr_embeddings = torch.cat([token_kinds_embeddings, embeddings], dim=-1)  # (batch_size, nr_exprs, nr_tokens_in_expr, embedding_dim + token_kind_embedding_dim)
        expr_embeddings_projected = self.projection_linear_layer(expr_embeddings.flatten(0, 2))\
            .view(batch_size, nr_exprs, nr_tokens_in_expr, self.expr_encoding_dim)
        expr_embeddings_projected_SNE = expr_embeddings_projected.flatten(0, 1).permute(1, 0, 2)  # (nr_tokens_in_expr, bsz, embedding_dim)
        if self.method == 'transformer_encoder':
            if expressions_mask is not None:
                expressions_mask = ~expressions_mask.flatten(0, 1)  # (bsz * nr_exprs, nr_tokens_in_expr)
            expr_encoded = self.transformer_encoder(
                expr_embeddings_projected_SNE,
                src_key_padding_mask=expressions_mask)\
                .sum(dim=0).view(batch_size, nr_exprs, -1)
        elif self.method == 'bi-lstm':
            expressions_mask = None if expressions_mask is None else expressions_mask.flatten(0, 1)
            lengths = None if expressions_mask is None else expressions_mask.long().sum(dim=1)
            lengths = torch.where(lengths <= torch.zeros(1, dtype=torch.long, device=lengths.device),
                                  torch.ones(1, dtype=torch.long, device=lengths.device), lengths)
            packed_input = pack_padded_sequence(expr_embeddings_projected_SNE, lengths=lengths, enforce_sorted=False)
            expr_rnn_outputs, (last_hidden_out, _) = self.lstm_layer(packed_input)
            assert last_hidden_out.size() == (2*2, batch_size * nr_exprs, self.expr_encoding_dim)
            last_hidden_out = last_hidden_out.view(2, 2, batch_size * nr_exprs, self.expr_encoding_dim)[-1, :, :, :]\
                .squeeze(0).sum(dim=0)
            assert last_hidden_out.size() == (batch_size * nr_exprs, self.expr_encoding_dim)

            if self.apply_attn:
                if expressions_mask is not None:
                    # quick fix for padding (no expressions there) to avoid later attn softmax of only -inf values.
                    expressions_mask[:, 0] = True

                expr_rnn_outputs, _ = pad_packed_sequence(sequence=expr_rnn_outputs)
                max_expr_len = expr_rnn_outputs.size()[0]
                assert max_expr_len == nr_tokens_in_expr
                assert expr_rnn_outputs.size() == (max_expr_len, batch_size * nr_exprs, 2 * self.expr_encoding_dim)
                expr_rnn_outputs = expr_rnn_outputs \
                    .view(max_expr_len, batch_size * nr_exprs, 2, self.expr_encoding_dim).sum(dim=-2).permute(1, 0, 2)
                assert expr_rnn_outputs.size() == (batch_size * nr_exprs, max_expr_len, self.expr_encoding_dim)

                expr_encoded = self.attn_layer(
                    sequences=expr_rnn_outputs, attn_key_from=last_hidden_out, mask=expressions_mask)
            else:
                expr_encoded = last_hidden_out

        return expr_encoded.view(batch_size, nr_exprs, self.expr_encoding_dim)
