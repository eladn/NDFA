import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm

from ddfa.code_nn_modules.vocabulary import Vocabulary
from ddfa.misc.code_data_structure_api import *
from ddfa.nn_utils.apply_batched_embeddings import apply_batched_embeddings
from ddfa.nn_utils.attn_rnn_encoder import AttnRNNEncoder


class ExpressionEncoder(nn.Module):
    def __init__(self, tokens_vocab: Vocabulary, tokens_kinds_vocab: Vocabulary,
                 expressions_special_words_vocab: Vocabulary, identifiers_special_words_vocab: Vocabulary,
                 token_embedding_dim: int = 256, identifiers_dim: int = 256, expr_encoding_dim: int = 1024,
                 token_kind_embedding_dim: int = 8, method: str = 'bi-lstm', nr_rnn_layers: int = 2,
                 nr_out_linear_layers: int = 2, dropout_rate: float = 0.3):
        assert method in {'bi-lstm', 'transformer_encoder'}
        assert nr_out_linear_layers >= 1
        super(ExpressionEncoder, self).__init__()
        self.tokens_vocab = tokens_vocab
        self.tokens_kinds_vocab = tokens_kinds_vocab
        self.expressions_special_words_vocab = expressions_special_words_vocab
        self.identifiers_special_words_vocab = identifiers_special_words_vocab
        self.token_embedding_dim = token_embedding_dim
        self.identifier_embedding_dim = identifiers_dim
        self.expr_encoding_dim = expr_encoding_dim
        self.method = method
        self.tokens_embedding_layer = nn.Embedding(
            num_embeddings=len(tokens_vocab), embedding_dim=self.token_embedding_dim,
            padding_idx=tokens_vocab.get_word_idx('<PAD>'))
        self.token_kind_embedding_dim = token_kind_embedding_dim
        self.tokens_kinds_embedding_layer = nn.Embedding(
            num_embeddings=len(tokens_kinds_vocab), embedding_dim=self.token_kind_embedding_dim,
            padding_idx=tokens_kinds_vocab.get_word_idx('<PAD>'))
        self.expressions_special_words_embedding_layer = nn.Embedding(
            num_embeddings=len(self.expressions_special_words_vocab), embedding_dim=self.expr_encoding_dim)
        self.identifiers_special_words_embedding_layer = nn.Embedding(
            num_embeddings=len(self.identifiers_special_words_vocab), embedding_dim=self.identifier_embedding_dim)

        self.projection_linear_layer = nn.Linear(
            self.token_kind_embedding_dim + self.token_embedding_dim + self.identifier_embedding_dim, self.expr_encoding_dim)
        self.additional_linear_layers = nn.ModuleList(
            [nn.Linear(self.expr_encoding_dim, self.expr_encoding_dim) for _ in range(nr_out_linear_layers - 1)])

        self.dropout_layer = nn.Dropout(p=dropout_rate)

        if method == 'transformer_encoder':
            transformer_encoder_layer = TransformerEncoderLayer(
                d_model=self.expr_encoding_dim, nhead=1)
            encoder_norm = LayerNorm(self.expr_encoding_dim)
            self.transformer_encoder = TransformerEncoder(
                encoder_layer=transformer_encoder_layer, num_layers=3, norm=encoder_norm)
        elif method == 'bi-lstm':
            self.attn_rnn_encoder = AttnRNNEncoder(
                input_dim=self.expr_encoding_dim, hidden_dim=self.expr_encoding_dim, rnn_type='lstm',
                nr_rnn_layers=nr_rnn_layers, rnn_bi_direction=True)

    def forward(self, expressions: torch.Tensor, expressions_mask: Optional[torch.BoolTensor],
                encoded_identifiers: torch.Tensor):  # Union[torch.Tensor, nn.utils.rnn.PackedSequence]
        assert len(expressions.size()) == 4 and expressions.size()[-1] == 2
        batch_size, nr_exprs, nr_tokens_in_expr, _ = expressions.size()
        assert len(encoded_identifiers.size()) == 3
        nr_identifiers_in_example = encoded_identifiers.size()[1]
        assert encoded_identifiers.size() == (batch_size, nr_identifiers_in_example, self.token_embedding_dim)
        assert expressions_mask is None or expressions_mask.size() == (batch_size, nr_exprs, nr_tokens_in_expr)

        expressions_tokens_kinds = expressions[:, :, :, 0]  # (batch_size, nr_exprs, nr_tokens_in_expr)
        expressions_idxs = expressions[:, :, :, 1]  # (batch_size, nr_exprs, nr_tokens_in_expr)
        assert expressions_tokens_kinds.size() == expressions_idxs.size() == (batch_size, nr_exprs, nr_tokens_in_expr)

        token_kinds_for_tokens_vocab = (
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.OPERATOR.value),
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.SEPARATOR.value),
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.KEYWORD.value))
        use_tokens_vocab_condition = reduce(
            torch.Tensor.logical_or,
            ((expressions_tokens_kinds == token_kind) for token_kind in token_kinds_for_tokens_vocab))
        tokens_idxs = torch.where(
            use_tokens_vocab_condition,
            expressions_idxs,
            torch.tensor(
                [self.tokens_vocab.get_word_idx('<NONE>')],
                dtype=expressions_idxs.dtype, device=expressions_idxs.device))  # (batch_size, nr_exprs, nr_tokens_in_expr)
        assert tokens_idxs.size() == expressions_idxs.size()
        selected_tokens_encoding = self.tokens_embedding_layer(tokens_idxs.flatten())\
            .view(batch_size, nr_exprs, nr_tokens_in_expr, self.token_embedding_dim)

        use_identifier_vocab_condition = expressions_tokens_kinds == self.tokens_kinds_vocab.get_word_idx_or_unk(
            SerTokenKind.IDENTIFIER.value)
        none_identifier_emb = self.identifiers_special_words_embedding_layer(
                torch.tensor([self.identifiers_special_words_vocab.get_word_idx('<NONE>')],
                             dtype=expressions_idxs.dtype, device=expressions_idxs.device))\
            .view(self.identifier_embedding_dim)
        selected_encoded_identifiers = apply_batched_embeddings(
            batched_embeddings=encoded_identifiers, indices=expressions_idxs, mask=use_identifier_vocab_condition,
            padding_embedding_vector=none_identifier_emb)
        assert self.identifier_embedding_dim == encoded_identifiers.size()[-1]
        assert selected_encoded_identifiers.size() == expressions_idxs.size() + (self.identifier_embedding_dim,)

        token_kinds_embeddings = self.tokens_kinds_embedding_layer(expressions_tokens_kinds.flatten())\
            .view(expressions_tokens_kinds.size() + (self.token_kind_embedding_dim,))  # (batch_size, nr_exprs, nr_tokens_in_expr, token_kind_embedding_dim)

        expr_embeddings = torch.cat(
            [token_kinds_embeddings, selected_tokens_encoding, selected_encoded_identifiers], dim=-1)  # (batch_size, nr_exprs, nr_tokens_in_expr, token_kind_embedding_dim + token_embedding_dim + identifier_embedding_dim)
        assert expr_embeddings.size() == (batch_size, nr_exprs, nr_tokens_in_expr, self.token_kind_embedding_dim +
                                          self.token_embedding_dim + self.identifier_embedding_dim)
        expr_embeddings = self.dropout_layer(expr_embeddings.flatten(0, 2))
        expr_embeddings_projected = self.dropout_layer(F.relu(
            self.projection_linear_layer(expr_embeddings)))
        for linear_layer in self.additional_linear_layers:
            expr_embeddings_projected = self.dropout_layer(F.relu(linear_layer(expr_embeddings_projected)))
        expr_embeddings_projected = expr_embeddings_projected.view(
            batch_size * nr_exprs, nr_tokens_in_expr, self.expr_encoding_dim)

        if self.method == 'transformer_encoder':
            expr_embeddings_projected_SNE = expr_embeddings_projected.permute(1, 0, 2)  # (nr_tokens_in_expr, bsz, embedding_dim)
            if expressions_mask is not None:
                expressions_mask = ~expressions_mask.flatten(0, 1)  # (bsz * nr_exprs, nr_tokens_in_expr)
            expr_encoded = self.transformer_encoder(
                expr_embeddings_projected_SNE,
                src_key_padding_mask=expressions_mask)\
                .sum(dim=0).view(batch_size, nr_exprs, -1)
        elif self.method == 'bi-lstm':
            if expressions_mask is not None:
                expressions_mask = expressions_mask.flatten(0, 1)  # (bsz * nr_exprs, nr_tokens_in_expr)
                # quick fix for padding (no expressions there) to avoid later attn softmax of only -inf values.
                expressions_mask[:, 0] = True
            expr_encoded = self.attn_rnn_encoder(
                sequence_input=expr_embeddings_projected, mask=expressions_mask, batch_first=True)

        return expr_encoded.view(batch_size, nr_exprs, self.expr_encoding_dim)
