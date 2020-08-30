import torch
import torch.nn as nn
import torch.nn.functional as F

from ndfa.code_nn_modules.code_task_input import SymbolsInputTensors
from ndfa.code_nn_modules.vocabulary import Vocabulary
from ndfa.code_nn_modules.expression_encoder import EncodedExpression


class SymbolsEncoder(nn.Module):
    def __init__(self, symbols_special_words_vocab: Vocabulary, symbol_embedding_dim: int, expr_encoding_dim: int,
                 dropout_rate: float = 0.3):
        super(SymbolsEncoder, self).__init__()
        self.symbols_special_words_vocab = symbols_special_words_vocab
        self.symbol_embedding_dim = symbol_embedding_dim
        # FIXME: might be problematic because 2 different modules hold `symbols_special_words_embedding` (both SymbolsEncoder and SymbolsDecoder).
        self.symbols_special_words_embedding = nn.Embedding(
            num_embeddings=len(self.symbols_special_words_vocab),
            embedding_dim=symbol_embedding_dim,
            padding_idx=self.symbols_special_words_vocab.get_word_idx('<PAD>'))
        self.symbols_token_occurrences_and_identifiers_embeddings_combiner = nn.Linear(
            in_features=expr_encoding_dim + symbol_embedding_dim, out_features=symbol_embedding_dim, bias=False)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, encoded_identifiers: torch.Tensor,
                symbols: SymbolsInputTensors,
                encoded_cfg_expressions: EncodedExpression):
        encoded_symbols_wo_commons = encoded_identifiers[symbols.symbols_identifier_indices.indices]

        if encoded_cfg_expressions is not None:
            max_nr_tokens_per_expression = encoded_cfg_expressions.full_expr_encoded.size(1)
            cfg_expr_tokens_indices_of_symbols_occurrences = \
                max_nr_tokens_per_expression * symbols.symbols_appearances_cfg_expression_idx.indices + \
                symbols.symbols_appearances_expression_token_idx.tensor
            cfg_expr_tokens_encodings_of_symbols_occurrences = \
                encoded_cfg_expressions.full_expr_encoded\
                .flatten(0, 1)[cfg_expr_tokens_indices_of_symbols_occurrences]
            symbols_occurrences_encodings_sum = \
                torch.zeros(
                    size=symbols.symbols_identifier_indices.indices.size() +
                         (cfg_expr_tokens_encodings_of_symbols_occurrences.size(-1),),
                    dtype=cfg_expr_tokens_encodings_of_symbols_occurrences.dtype,
                    device=cfg_expr_tokens_encodings_of_symbols_occurrences.device)
            symbols_occurrences_encodings = symbols_occurrences_encodings_sum.scatter_add(
                dim=0,
                index=symbols.symbols_appearances_symbol_idx.indices.unsqueeze(-1)
                .expand(cfg_expr_tokens_encodings_of_symbols_occurrences.size()),
                src=cfg_expr_tokens_encodings_of_symbols_occurrences)
            # TODO: we might want to normalize the `symbols_occurrences_encodings` (maybe using `nn.BatchNorm1d`)

            assert encoded_symbols_wo_commons.size()[:-1] == symbols_occurrences_encodings.size()[:-1]
            combined_symbols_encoding = torch.cat(
                [encoded_symbols_wo_commons, symbols_occurrences_encodings], dim=-1)
            combined_symbols_encoding = \
                self.symbols_token_occurrences_and_identifiers_embeddings_combiner(combined_symbols_encoding)
            combined_symbols_encoding = self.dropout_layer(F.relu(combined_symbols_encoding))
        else:
            combined_symbols_encoding = encoded_symbols_wo_commons

        unflattened_combined_symbols_encoding = \
            symbols.symbols_identifier_indices.unflatten(combined_symbols_encoding)
        assert unflattened_combined_symbols_encoding.ndim == 3
        return unflattened_combined_symbols_encoding
