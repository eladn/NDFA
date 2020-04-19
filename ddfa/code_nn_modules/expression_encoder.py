import torch
import torch.nn as nn
from functools import reduce
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm

from ddfa.code_nn_modules.vocabulary import Vocabulary
from ddfa.code_data_structure_api import *


class ExpressionEncoder(nn.Module):
    def __init__(self, tokens_vocab: Vocabulary, tokens_kinds_vocab: Vocabulary, tokens_embedding_dim: int = 256, expr_encoding_dim: int = 1028):
        super(ExpressionEncoder, self).__init__()
        self.tokens_vocab = tokens_vocab
        self.tokens_kinds_vocab = tokens_kinds_vocab
        self.tokens_embedding_dim = tokens_embedding_dim
        self.expr_encoding_dim = expr_encoding_dim
        self.tokens_embedding_layer = nn.Embedding(
            num_embeddings=len(tokens_vocab), embedding_dim=self.tokens_embedding_dim)
        self.token_kind_embedding_dim = 4
        self.tokens_kinds_embedding_layer = nn.Embedding(
            num_embeddings=len(tokens_kinds_vocab), embedding_dim=self.token_kind_embedding_dim)

        self.projection_linear_layer = nn.Linear(self.tokens_embedding_dim, self.expr_encoding_dim)
        transformer_encoder_layer = TransformerEncoderLayer(
            d_model=self.expr_encoding_dim, nhead=1)
        encoder_norm = LayerNorm(self.expr_encoding_dim)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=transformer_encoder_layer, num_layers=3, norm=encoder_norm)

    def forward(self, expressions: torch.Tensor, encoded_identifiers: torch.Tensor):  # Union[torch.Tensor, nn.utils.rnn.PackedSequence]
        assert len(expressions.size()) == 4 and expressions.size()[-1] == 2
        batch_size, nr_exprs, nr_tokens_in_expr, _ = expressions.size()
        assert len(encoded_identifiers.size()) == 3
        nr_identifiers_in_example = encoded_identifiers.size()[1]
        assert encoded_identifiers.size() == (batch_size, nr_identifiers_in_example, self.tokens_embedding_dim)

        expressions_tokens_kinds = expressions[:, :, :, 0]  # (batch_size, nr_exprs, nr_tokens_in_expr)
        expressions_idxs = expressions[:, :, :, 1]  # (batch_size, nr_exprs, nr_tokens_in_expr)
        assert expressions_tokens_kinds.size() == expressions_idxs.size() == (batch_size, nr_exprs, nr_tokens_in_expr)

        use_identifier_vocab_condition = expressions_tokens_kinds == self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.IDENTIFIER.value)
        identifiers_idxs = torch.where(
            use_identifier_vocab_condition,
            expressions_idxs, torch.zeros(expressions_idxs.size()))  # (batch_size, nr_exprs, nr_tokens_in_expr)
        assert identifiers_idxs.size() == expressions.size()[:-1]

        token_kinds_for_tokens_vocab = (
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.OPERATOR.value),
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.SEPARATOR.value),
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.KEYWORD.value))
        use_tokens_vocab_condition = reduce(
            torch.Tensor.__or__, (expressions_tokens_kinds == token_kind for token_kind in token_kinds_for_tokens_vocab))
        tokens_idxs = torch.where(
            use_tokens_vocab_condition,
            expressions_idxs, torch.zeros(expressions_idxs.size()))  # (batch_size, nr_exprs, nr_tokens_in_expr)
        assert tokens_idxs.size() == expressions.size()[:-1]

        selected_tokens_encoding = self.tokens_embedding_layer(tokens_idxs.flatten())\
            .view(batch_size, nr_exprs, nr_tokens_in_expr, self.tokens_embedding_dim)

        identifiers_idxs_flattened = identifiers_idxs.flatten()  # (batch_size * nr_exprs * nr_tokens_in_expr,)
        assert identifiers_idxs_flattened.size() == (batch_size * nr_exprs * nr_tokens_in_expr,)
        identifiers_idxs_offsets_fix_step = nr_exprs * nr_tokens_in_expr
        identifiers_idxs_offsets_fixes = (torch.range(start=0, end=batch_size-1) * nr_identifiers_in_example)\
            .repeat((identifiers_idxs_offsets_fix_step, 1)).T.flatten()  #  = [0,0,...0,1,1,...,1, ...]
        assert identifiers_idxs_flattened.size() == identifiers_idxs_offsets_fixes.size()
        identifiers_idxs_flattened_with_fixed_offsets = identifiers_idxs_flattened + identifiers_idxs_offsets_fixes
        encoded_identifiers_flattened = encoded_identifiers.flatten(0, 1)  # (batch_size*encoded_identifiers, embedding_dim)
        selected_encoded_identifiers_flattened = encoded_identifiers_flattened[identifiers_idxs_flattened_with_fixed_offsets]  # (batch_size * nr_exprs * nr_tokens_in_expr, embedding_dim)
        assert selected_encoded_identifiers_flattened.size() == (
            batch_size * nr_exprs * nr_tokens_in_expr, self.tokens_embedding_dim)
        selected_encoded_identifiers = selected_encoded_identifiers_flattened.view(
            batch_size, nr_exprs, nr_tokens_in_expr, self.tokens_embedding_dim)
        # TODO: we could thread the tokens-embedding & identifiers-encodings as PERPETUAL to each others (concat - other dims for each)
        embeddings = torch.where(
            use_identifier_vocab_condition, selected_encoded_identifiers, torch.where(
                use_tokens_vocab_condition, selected_tokens_encoding, torch.zeros(selected_tokens_encoding.size())))  # (batch_size, nr_exprs, nr_tokens_in_expr, embedding_dim)
        assert embeddings.size() == (batch_size, nr_exprs, nr_tokens_in_expr, self.tokens_embedding_dim)

        token_kinds_embeddings = self.tokens_kinds_embedding_layer(expressions_tokens_kinds.flatten())\
            .view(expressions_tokens_kinds.size() + (self.token_kind_embedding_dim,))  # (batch_size, nr_exprs, nr_tokens_in_expr, token_kind_embedding_dim)

        expr_embeddings = torch.cat([token_kinds_embeddings, embeddings], dim=-1)  # (batch_size, nr_exprs, nr_tokens_in_expr, embedding_dim + token_kind_embedding_dim)
        expr_embeddings_projected = self.projection_linear_layer(expr_embeddings.flatten(0, 2))\
            .view(batch_size, nr_exprs, nr_tokens_in_expr, -1)
        expr_encoded = self.transformer_encoder(expr_embeddings_projected.flatten(0, 1).permute(0, 1))\
            .sum(dim=0).view(batch_size, nr_exprs, -1)
        return expr_encoded
