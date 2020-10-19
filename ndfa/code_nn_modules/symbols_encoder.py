import torch
import torch.nn as nn
from typing import Optional

from ndfa.nn_utils.misc import get_activation_layer
from ndfa.code_nn_modules.code_task_input import SymbolsInputTensors
from ndfa.nn_utils.scatter_combiner import ScatterCombiner


__all__ = ['SymbolsEncoder']


class SymbolsEncoder(nn.Module):
    def __init__(self, symbol_embedding_dim: int,
                 expression_encoding_dim: int,
                 combining_method: str = 'sum',
                 dropout_rate: float = 0.3,
                 activation_fn: str = 'relu'):
        super(SymbolsEncoder, self).__init__()
        self.symbol_embedding_dim = symbol_embedding_dim
        self.scatter_combiner = ScatterCombiner(
            encoding_dim=self.symbol_embedding_dim, combining_method=combining_method)
        self.symbols_token_occurrences_and_identifiers_embeddings_combiner = nn.Linear(
            in_features=expression_encoding_dim + symbol_embedding_dim, out_features=symbol_embedding_dim, bias=False)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, encoded_identifiers: torch.Tensor,
                symbols: SymbolsInputTensors,
                encoded_cfg_expressions: Optional[torch.Tensor] = None):
        symbols_identifiers_encodings = encoded_identifiers[symbols.symbols_identifier_indices.indices]

        if encoded_cfg_expressions is not None:
            max_nr_tokens_per_expression = encoded_cfg_expressions.size(1)
            cfg_expr_tokens_indices_of_symbols_occurrences = \
                max_nr_tokens_per_expression * symbols.symbols_appearances_cfg_expression_idx.indices + \
                symbols.symbols_appearances_expression_token_idx.tensor
            cfg_expr_tokens_encodings_of_symbols_occurrences = \
                encoded_cfg_expressions.flatten(0, 1)[cfg_expr_tokens_indices_of_symbols_occurrences]
            nr_symbols = symbols.symbols_identifier_indices.indices.size(0)

            symbols_occurrences_encodings = self.scatter_combiner(
                scattered_input=cfg_expr_tokens_encodings_of_symbols_occurrences,
                indices=symbols.symbols_appearances_symbol_idx.indices,
                dim_size=nr_symbols, attn_keys=symbols_identifiers_encodings)

            # symbols_occurrences_encodings = scatter_add(
            #     src=cfg_expr_tokens_encodings_of_symbols_occurrences,
            #     index=symbols.symbols_appearances_symbol_idx.indices,
            #     dim=0, dim_size=nr_symbols)
            # symbols_occurrences_encodings = scatter_sum(
            #     src=cfg_expr_tokens_encodings_of_symbols_occurrences,
            #     index=symbols.symbols_appearances_symbol_idx.indices.unsqueeze(-1)
            #         .expand(cfg_expr_tokens_encodings_of_symbols_occurrences.size()),
            #     dim=0, dim_size=nr_symbols)

            assert symbols_identifiers_encodings.size()[:-1] == symbols_occurrences_encodings.size()[:-1]
            final_symbols_encoding = torch.cat(
                [symbols_identifiers_encodings, symbols_occurrences_encodings], dim=-1)
            final_symbols_encoding = \
                self.symbols_token_occurrences_and_identifiers_embeddings_combiner(final_symbols_encoding)
            final_symbols_encoding = self.dropout_layer(self.activation_layer(final_symbols_encoding))
        else:
            final_symbols_encoding = symbols_identifiers_encodings
        return final_symbols_encoding
