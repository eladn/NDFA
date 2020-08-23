import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, Optional

from ndfa.nn_utils.scattered_encodings import ScatteredEncodings
from ndfa.code_nn_modules.code_task_input import MethodCodeInputPaddedTensors
from ndfa.code_nn_modules.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.expression_encoder import ExpressionEncoder
from ndfa.code_nn_modules.identifier_encoder import IdentifierEncoder
from ndfa.code_nn_modules.cfg_node_encoder import CFGNodeEncoder, EncodedCFGNode
from ndfa.code_nn_modules.symbols_encoder import SymbolsEncoder


__all__ = ['MethodCodeEncoder', 'EncodedMethodCode']


class EncodedMethodCode(NamedTuple):
    encoded_identifiers: torch.Tensor
    encoded_cfg_nodes: torch.Tensor
    encoded_symbols: torch.Tensor
    encoded_cfg_nodes_after_bridge: torch.Tensor
    encoded_symbols_occurrences: Optional[ScatteredEncodings] = None


class MethodCodeEncoder(nn.Module):
    def __init__(self, code_task_vocabs: CodeTaskVocabs, identifier_embedding_dim: int = 256,
                 expr_encoding_dim: int = 1028, nr_encoder_decoder_bridge_layers: int = 0, dropout_p: float = 0.3):
        super(MethodCodeEncoder, self).__init__()
        self.identifier_embedding_dim = identifier_embedding_dim
        self.expr_encoding_dim = expr_encoding_dim
        self.identifier_encoder = IdentifierEncoder(
            sub_identifiers_vocab=code_task_vocabs.sub_identifiers, embedding_dim=self.identifier_embedding_dim)
        expression_encoder = ExpressionEncoder(
            kos_tokens_vocab=code_task_vocabs.kos_tokens, tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
            expressions_special_words_vocab=code_task_vocabs.expressions_special_words,
            identifiers_special_words_vocab=code_task_vocabs.identifiers_special_words,
            kos_token_embedding_dim=self.identifier_embedding_dim, expr_encoding_dim=self.expr_encoding_dim)
        self.cfg_node_encoder = CFGNodeEncoder(
            expression_encoder=expression_encoder, pdg_node_control_kinds_vocab=code_task_vocabs.pdg_node_control_kinds)
        self.encoder_decoder_bridge_dense_layers = nn.ModuleList([
            nn.Linear(in_features=self.cfg_node_encoder.output_dim, out_features=self.cfg_node_encoder.output_dim)
            for _ in range(nr_encoder_decoder_bridge_layers)]) if nr_encoder_decoder_bridge_layers else None
        self.symbols_encoder = SymbolsEncoder(
            symbols_special_words_vocab=code_task_vocabs.symbols_special_words,
            symbol_embedding_dim=self.identifier_embedding_dim)  # it might change...
        self.dropout_layer = nn.Dropout(dropout_p)

    def forward(self, code_task_input: MethodCodeInputPaddedTensors) -> EncodedMethodCode:
        encoded_identifiers = self.identifier_encoder(
            sub_identifiers_indices=code_task_input.identifiers,
            sub_identifiers_mask=code_task_input.sub_identifiers_mask)  # (batch_size, nr_identifiers, identifier_encoding_dim)
        encoded_cfg_nodes: EncodedCFGNode = self.cfg_node_encoder(
            encoded_identifiers=encoded_identifiers, cfg_nodes_expressions=code_task_input.cfg_nodes_expressions,
            cfg_nodes_expressions_mask=code_task_input.cfg_nodes_expressions_mask,
            cfg_nodes_mask=code_task_input.cfg_nodes_mask,
            cfg_nodes_control_kind=code_task_input.cfg_nodes_control_kind)  # (batch_size, nr_cfg_nodes, cfg_node_encoding_dim)

        encoded_symbols = self.symbols_encoder(
            encoded_identifiers=encoded_identifiers,
            identifiers_idxs_of_all_symbols=code_task_input.identifiers_idxs_of_all_symbols,
            identifiers_idxs_of_all_symbols_mask=code_task_input.identifiers_idxs_of_all_symbols_mask)

        assert code_task_input.indices_of_symbols_occurrences_in_cfg_nodes_expressions.tensor.ndim == 3
        assert code_task_input.indices_of_symbols_occurrences_in_cfg_nodes_expressions.tensor.size(0) == \
               code_task_input.batch_size
        assert code_task_input.indices_of_symbols_occurrences_in_cfg_nodes_expressions.tensor.size(2) == 2
        flattened_indices_of_symbols_occurrences_in_cfg_nodes_expressions = \
            encoded_cfg_nodes.encoded_cfg_nodes_expressions.full_expr_encoded.size(2) * \
            code_task_input.indices_of_symbols_occurrences_in_cfg_nodes_expressions.tensor[:, :, 0] + \
            code_task_input.indices_of_symbols_occurrences_in_cfg_nodes_expressions.tensor[:, :, 1]
        flattened_indices_of_symbols_occurrences_in_cfg_nodes_expressions = \
            flattened_indices_of_symbols_occurrences_in_cfg_nodes_expressions.unsqueeze(-1)\
                .expand(flattened_indices_of_symbols_occurrences_in_cfg_nodes_expressions.size() +
                        (encoded_cfg_nodes.encoded_cfg_nodes_expressions.full_expr_encoded.size(-1),))
        symbol_occurrences_encodings = torch.gather(
            encoded_cfg_nodes.encoded_cfg_nodes_expressions.full_expr_encoded.flatten(1, 2),
            dim=1, index=flattened_indices_of_symbols_occurrences_in_cfg_nodes_expressions)
        symbols_occurrences_scattered_encodings = ScatteredEncodings(
            encodings=symbol_occurrences_encodings,
            indices=code_task_input.symbols_idxs_of_symbols_occurrences_in_cfg_nodes_expressions.tensor,
            mask=code_task_input.symbols_idxs_of_symbols_occurrences_in_cfg_nodes_expressions.collate_mask)

        encoded_cfg_nodes_after_bridge = encoded_cfg_nodes.encoded_cfg_nodes
        if self.encoder_decoder_bridge_dense_layers:
            encoded_cfg_nodes_after_bridge = functools.reduce(
                lambda last_res, cur_layer: self.dropout_layer(F.relu(cur_layer(last_res))),
                self.encoder_decoder_bridge_dense_layers,
                encoded_cfg_nodes.encoded_cfg_nodes.flatten(0, 1))\
                .view(encoded_cfg_nodes.encoded_cfg_nodes.size()[:-1] + (-1,))

        return EncodedMethodCode(
            encoded_identifiers=encoded_identifiers,
            encoded_cfg_nodes=encoded_cfg_nodes.encoded_cfg_nodes,
            encoded_symbols=encoded_symbols,
            encoded_cfg_nodes_after_bridge=encoded_cfg_nodes_after_bridge,
            encoded_symbols_occurrences=symbols_occurrences_scattered_encodings)
