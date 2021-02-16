import torch
import torch.nn as nn
from typing import Optional

from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors
from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.code_nn_modules.code_task_input import SymbolsInputTensors
from ndfa.nn_utils.modules.scatter_combiner import ScatterCombiner


__all__ = ['SymbolsEncoder']


class SymbolsEncoder(nn.Module):
    def __init__(self,
                 symbol_embedding_dim: int,
                 identifier_embedding_dim: int,
                 expression_encoding_dim: int,
                 combining_method: str = 'sum',
                 dropout_rate: float = 0.3,
                 activation_fn: str = 'relu'):
        super(SymbolsEncoder, self).__init__()
        self.symbol_embedding_dim = symbol_embedding_dim
        self.scatter_combiner = ScatterCombiner(
            encoding_dim=self.symbol_embedding_dim, combining_method=combining_method)
        self.symbols_token_occurrences_and_identifiers_embeddings_combiner = nn.Linear(
            in_features=expression_encoding_dim + identifier_embedding_dim,
            out_features=symbol_embedding_dim, bias=False)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, encoded_identifiers: torch.Tensor,
                symbols: SymbolsInputTensors,
                encoded_expressions: Optional[torch.Tensor] = None,
                tokenized_expressions_input: Optional[CodeExpressionTokensSequenceInputTensors] = None,
                encoded_ast_nodes: Optional[torch.Tensor] = None,
                ast_nodes_with_symbol_leaf_nodes_indices: Optional[torch.LongTensor] = None,
                ast_nodes_with_symbol_leaf_symbol_idx: Optional[torch.LongTensor] = None):
        symbols_identifiers_encodings = encoded_identifiers[symbols.symbols_identifier_indices.indices]

        if encoded_expressions is not None or encoded_ast_nodes is not None:
            if encoded_expressions is not None:
                assert tokenized_expressions_input.is_symbol_mask.sequences.shape == encoded_expressions.shape[:-1]
                encodings_of_symbols_occurrences = encoded_expressions[tokenized_expressions_input.is_symbol_mask.sequences]
                assert encodings_of_symbols_occurrences.shape[:-1] == tokenized_expressions_input.symbol_index.indices.shape
                symbols_indices_of_symbols_occurrences = tokenized_expressions_input.symbol_index.indices  #symbols.symbols_appearances_symbol_idx.indices
            elif encoded_ast_nodes is not None:
                assert ast_nodes_with_symbol_leaf_nodes_indices.shape == ast_nodes_with_symbol_leaf_symbol_idx.shape
                encodings_of_symbols_occurrences = encoded_ast_nodes[ast_nodes_with_symbol_leaf_nodes_indices]
                symbols_indices_of_symbols_occurrences = ast_nodes_with_symbol_leaf_symbol_idx
            else:
                assert False

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

            assert symbols_identifiers_encodings.size()[:-1] == symbols_occurrences_encodings.size()[:-1]
            final_symbols_encoding = torch.cat(
                [symbols_identifiers_encodings, symbols_occurrences_encodings], dim=-1)
            final_symbols_encoding = \
                self.symbols_token_occurrences_and_identifiers_embeddings_combiner(final_symbols_encoding)
            final_symbols_encoding = self.dropout_layer(self.activation_layer(final_symbols_encoding))
        else:
            final_symbols_encoding = symbols_identifiers_encodings
        return final_symbols_encoding
