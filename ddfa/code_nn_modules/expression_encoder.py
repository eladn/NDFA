import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm

from ddfa.code_nn_modules.vocabulary import Vocabulary
from ddfa.code_data_structure_api import *
from ddfa.nn_utils.apply_batched_embeddings import apply_batched_embeddings
from ddfa.nn_utils.attn_rnn_encoder import AttnRNNEncoder


class ExpressionEncoder(nn.Module):
    def __init__(self, tokens_vocab: Vocabulary, tokens_kinds_vocab: Vocabulary, tokens_embedding_dim: int = 256,
                 expr_encoding_dim: int = 1028, token_kind_embedding_dim: int = 4,
                 method: str = 'bi-lstm', nr_rnn_layers: int = 2):
        assert method in {'bi-lstm', 'transformer_encoder'}
        super(ExpressionEncoder, self).__init__()
        self.tokens_vocab = tokens_vocab
        self.tokens_kinds_vocab = tokens_kinds_vocab
        self.tokens_embedding_dim = tokens_embedding_dim
        self.expr_encoding_dim = expr_encoding_dim
        self.method = method
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
            self.attn_rnn_encoder = AttnRNNEncoder(
                input_dim=self.expr_encoding_dim, hidden_dim=self.expr_encoding_dim, rnn_type='lstm',
                nr_rnn_layers=nr_rnn_layers, bi_direction=True)

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
            expressions_idxs,
            torch.zeros(size=(1,), dtype=expressions_idxs.dtype, device=expressions_idxs.device))  # (batch_size, nr_exprs, nr_tokens_in_expr)
        assert identifiers_idxs.size() == expressions.size()[:-1]

        token_kinds_for_tokens_vocab = (
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.OPERATOR.value),
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.SEPARATOR.value),
            self.tokens_kinds_vocab.get_word_idx_or_unk(SerTokenKind.KEYWORD.value))
        use_tokens_vocab_condition = reduce(
            torch.Tensor.__or__, (expressions_tokens_kinds == token_kind for token_kind in token_kinds_for_tokens_vocab))
        tokens_idxs = torch.where(
            use_tokens_vocab_condition,
            expressions_idxs,
            torch.tensor([self.tokens_vocab.get_word_idx_or_unk('<UNK>')],
                         dtype=expressions_idxs.dtype, device=expressions_idxs.device))  # (batch_size, nr_exprs, nr_tokens_in_expr)
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
            selected_encoded_identifiers, selected_tokens_encoding)  # (batch_size, nr_exprs, nr_tokens_in_expr, embedding_dim)
        assert embeddings.size() == (batch_size, nr_exprs, nr_tokens_in_expr, self.tokens_embedding_dim)

        token_kinds_embeddings = self.tokens_kinds_embedding_layer(expressions_tokens_kinds.flatten())\
            .view(expressions_tokens_kinds.size() + (self.token_kind_embedding_dim,))  # (batch_size, nr_exprs, nr_tokens_in_expr, token_kind_embedding_dim)

        expr_embeddings = torch.cat([token_kinds_embeddings, embeddings], dim=-1)  # (batch_size, nr_exprs, nr_tokens_in_expr, embedding_dim + token_kind_embedding_dim)
        expr_embeddings_projected = F.relu(self.projection_linear_layer(expr_embeddings.flatten(0, 2)))\
            .view(batch_size, nr_exprs, nr_tokens_in_expr, self.expr_encoding_dim)
        expr_embeddings_projected = expr_embeddings_projected.flatten(0, 1)  # (bsz * nr_exprs, nr_tokens_in_expr, embedding_dim)
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
