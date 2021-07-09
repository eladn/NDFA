import torch
import torch.nn as nn
from typing import Optional

from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.code_nn_modules.code_task_input import SymbolsInputTensors
from ndfa.nn_utils.modules.scatter_combiner import ScatterCombiner
from ndfa.code_nn_modules.params.symbols_encoder_params import SymbolsEncoderParams


__all__ = ['SymbolsEncoder']


class SymbolsEncoder(nn.Module):
    def __init__(self,
                 symbol_embedding_dim: int,
                 identifier_embedding_dim: int,
                 expression_encoding_dim: int,
                 encoder_params: SymbolsEncoderParams,
                 dropout_rate: float = 0.3,
                 activation_fn: str = 'relu'):
        super(SymbolsEncoder, self).__init__()
        self.symbol_embedding_dim = symbol_embedding_dim
        self.encoder_params = encoder_params
        if self.encoder_params.use_symbols_occurrences:
            self.scatter_combiner = ScatterCombiner(
                encoding_dim=expression_encoding_dim,
                applied_attn_output_dim=self.symbol_embedding_dim,
                combiner_params=self.encoder_params.combining_params)
            # TODO: add `output_dim` to `ScatterCombiner` and use it
            self.scatter_combiner_output_dim = self.symbol_embedding_dim \
                if self.scatter_combiner.combiner_params.method == 'attn' else expression_encoding_dim
        if self.encoder_params.use_symbols_occurrences and self.encoder_params.use_identifier_encoding:
            self.symbols_token_occurrences_and_identifiers_embeddings_combiner = nn.Linear(
                in_features=self.scatter_combiner_output_dim + identifier_embedding_dim,
                out_features=self.symbol_embedding_dim, bias=False)
        else:
            assert self.scatter_combiner_output_dim == self.symbol_embedding_dim
            # TODO: impl linear projection if it's not the case
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, encoded_identifiers: torch.Tensor,
                symbols: SymbolsInputTensors,
                encodings_of_symbols_occurrences: Optional[torch.Tensor] = None,
                symbols_indices_of_symbols_occurrences: Optional[torch.LongTensor] = None):
        symbols_identifiers_encodings = encoded_identifiers[symbols.symbols_identifier_indices.indices]

        if self.encoder_params.use_symbols_occurrences:
            assert encodings_of_symbols_occurrences is not None
            assert symbols_indices_of_symbols_occurrences is not None

            # max_nr_tokens_per_expression = encoded_expressions.size(1)
            # cfg_expr_tokens_indices_of_symbols_occurrences = \
            #     max_nr_tokens_per_expression * symbols.symbols_appearances_cfg_expression_idx.indices + \
            #     symbols.symbols_appearances_expression_token_idx.tensor
            # cfg_expr_tokens_encodings_of_symbols_occurrences = \
            #     encoded_expressions.flatten(0, 1)[cfg_expr_tokens_indices_of_symbols_occurrences]
            nr_symbols = symbols.symbols_identifier_indices.indices.size(0)

            symbols_occurrences_encodings = self.scatter_combiner(
                scattered_input=encodings_of_symbols_occurrences,
                indices=symbols_indices_of_symbols_occurrences,
                dim_size=nr_symbols, attn_queries=symbols_identifiers_encodings)

            # symbols_occurrences_encodings = scatter_add(
            #     src=cfg_expr_tokens_encodings_of_symbols_occurrences,
            #     index=symbols.symbols_appearances_symbol_idx.indices,
            #     dim=0, dim_size=nr_symbols)
            # symbols_occurrences_encodings = scatter_sum(
            #     src=cfg_expr_tokens_encodings_of_symbols_occurrences,
            #     index=symbols.symbols_appearances_symbol_idx.indices.unsqueeze(-1)
            #         .expand(cfg_expr_tokens_encodings_of_symbols_occurrences.size()),
            #     dim=0, dim_size=nr_symbols)

            if self.encoder_params.use_identifier_encoding:
                assert symbols_identifiers_encodings.size()[:-1] == symbols_occurrences_encodings.size()[:-1]
                final_symbols_encoding = torch.cat(
                    [symbols_identifiers_encodings, symbols_occurrences_encodings], dim=-1)
                final_symbols_encoding = \
                    self.symbols_token_occurrences_and_identifiers_embeddings_combiner(final_symbols_encoding)
            else:
                assert self.symbol_embedding_dim == symbols_occurrences_encodings.size(-1)
                final_symbols_encoding = symbols_occurrences_encodings
            final_symbols_encoding = self.dropout_layer(self.activation_layer(final_symbols_encoding))
        else:
            assert self.encoder_params.use_identifier_encoding
            assert self.symbol_embedding_dim == symbols_identifiers_encodings.size(-1)
            final_symbols_encoding = symbols_identifiers_encodings
        return final_symbols_encoding
