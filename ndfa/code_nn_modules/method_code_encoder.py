import functools
import torch
import torch.nn as nn
from typing import NamedTuple, Optional

from ndfa.code_nn_modules.params.method_code_encoder_params import MethodCodeEncoderParams
from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.nn_utils.modules.attn_rnn_decoder import ScatteredEncodings
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.identifier_encoder import IdentifierEncoder
from ndfa.code_nn_modules.method_cfg_encoder import MethodCFGEncoder, EncodedMethodCFG
from ndfa.code_nn_modules.method_cfg_encoder_v2 import MethodCFGEncoderV2, EncodedMethodCFGV2
from ndfa.code_nn_modules.code_expression_embedder import CodeExpressionEmbedder
from ndfa.code_nn_modules.code_expression_encoder import CodeExpressionEncoder
from ndfa.code_nn_modules.symbols_encoder import SymbolsEncoder
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors


__all__ = ['MethodCodeEncoder', 'EncodedMethodCode']


class EncodedMethodCode(NamedTuple):
    encoded_identifiers: torch.Tensor
    whole_method_ast_nodes_encoding: torch.Tensor
    whole_method_token_seqs_encoding: torch.Tensor
    encoded_cfg_nodes: torch.Tensor
    encoded_cfg_nodes_after_bridge: torch.Tensor
    encoded_symbols: torch.Tensor
    encoded_symbols_occurrences: Optional[ScatteredEncodings] = None


class MethodCodeEncoder(nn.Module):
    def __init__(self, code_task_vocabs: CodeTaskVocabs, encoder_params: MethodCodeEncoderParams,
                 nr_encoder_decoder_bridge_layers: int = 0,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(MethodCodeEncoder, self).__init__()
        self.encoder_params = encoder_params
        self.activation_layer = get_activation_layer(activation_fn)()

        self.identifier_encoder = IdentifierEncoder(
            sub_identifiers_vocab=code_task_vocabs.sub_identifiers,
            encoder_params=self.encoder_params.identifier_encoder,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

        # TODO: use `encoder_params.method_encoder_type` in forward()!
        if self.encoder_params.method_encoder_type == 'method-cfg':
            self.method_cfg_encoder = MethodCFGEncoder(
                code_task_vocabs=code_task_vocabs,
                encoder_params=self.encoder_params.method_cfg_encoder,
                identifier_embedding_dim=self.encoder_params.identifier_encoder.identifier_embedding_dim,
                symbol_embedding_dim=self.encoder_params.symbol_embedding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.method_encoder_type == 'method-cfg-v2':
            self.method_cfg_encoder_v2 = MethodCFGEncoderV2(
                code_task_vocabs=code_task_vocabs,
                encoder_params=self.encoder_params.method_cfg_encoder,
                identifier_embedding_dim=self.encoder_params.identifier_encoder.identifier_embedding_dim,
                symbol_embedding_dim=self.encoder_params.symbol_embedding_dim,
                symbols_encoder_params=self.encoder_params.symbols_encoder_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.method_encoder_type == 'whole-method':
            self.ast_node_embedding_dim = self.encoder_params.method_linear_seq_expression_encoder.token_encoding_dim  # TODO: FIXME: plug-in correct HPs
            self.whole_method_code_embedder = CodeExpressionEmbedder(
                code_task_vocabs=code_task_vocabs,
                encoder_params=self.encoder_params.method_linear_seq_expression_encoder,
                identifier_embedding_dim=self.encoder_params.identifier_encoder.identifier_embedding_dim,
                ast_node_embedding_dim=self.ast_node_embedding_dim,
                nr_final_embeddings_linear_layers=1,  # TODO: plug HP here
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            self.whole_method_code_encoder = CodeExpressionEncoder(
                encoder_params=self.encoder_params.method_linear_seq_expression_encoder,
                code_task_vocabs=code_task_vocabs,
                identifier_embedding_dim=self.encoder_params.identifier_encoder.identifier_embedding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            self.symbols_encoder = SymbolsEncoder(
                symbol_embedding_dim=self.encoder_params.symbol_embedding_dim,
                expression_encoding_dim=self.encoder_params.method_cfg_encoder.cfg_node_expression_encoder.token_encoding_dim,
                identifier_embedding_dim=self.encoder_params.identifier_encoder.identifier_embedding_dim,
                encoder_params=self.encoder_params.symbols_encoder_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        else:
            raise ValueError(f'Unexpected method code encoder type `{self.encoder_params.method_encoder_type}`.')

        self.encoder_decoder_bridge_dense_layers = nn.ModuleList([
            nn.Linear(in_features=self.encoder_params.method_cfg_encoder.cfg_node_encoding_dim,
                      out_features=self.encoder_params.method_cfg_encoder.cfg_node_encoding_dim)  # TODO: make it abstract `encoder.output_dim` to support additional encoders
            for _ in range(nr_encoder_decoder_bridge_layers)])
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, code_task_input: MethodCodeInputTensors) -> EncodedMethodCode:
        # (nr_identifiers_in_batch, identifier_encoding_dim)
        encoded_identifiers = self.identifier_encoder(
            identifiers=code_task_input.identifiers)  # (nr_identifiers_in_batch, identifier_encoding_dim)

        whole_method_code_encoded = None
        unflattened_cfg_nodes_encodings = None
        encoded_cfg_nodes_after_bridge = None
        if self.encoder_params.method_encoder_type == 'method-cfg':
            encoded_method_cfg: EncodedMethodCFG = self.method_cfg_encoder(
                code_task_input=code_task_input, encoded_identifiers=encoded_identifiers)
            unflattened_cfg_nodes_encodings = code_task_input.pdg.cfg_nodes_control_kind.unflatten(
                encoded_method_cfg.encoded_cfg_nodes)
            encoded_symbols = encoded_method_cfg.encoded_symbols
            encoded_cfg_nodes_after_bridge = unflattened_cfg_nodes_encodings
            if len(self.encoder_decoder_bridge_dense_layers) > 0:
                encoded_cfg_nodes_after_bridge = functools.reduce(
                    lambda last_res, cur_layer: self.dropout_layer(self.activation_layer(cur_layer(last_res))),
                    self.encoder_decoder_bridge_dense_layers,
                    encoded_method_cfg.encoded_cfg_nodes.flatten(0, 1))\
                    .view(encoded_cfg_nodes_after_bridge.size()[:-1] + (-1,))
        elif self.encoder_params.method_encoder_type == 'method-cfg-v2':
            encoded_method_cfg: EncodedMethodCFGV2 = self.method_cfg_encoder_v2(
                code_task_input=code_task_input, encoded_identifiers=encoded_identifiers)
            unflattened_cfg_nodes_encodings = code_task_input.pdg.cfg_nodes_control_kind.unflatten(
                encoded_method_cfg.encoded_cfg_nodes)
            encoded_symbols = encoded_method_cfg.encoded_symbols
            encoded_cfg_nodes_after_bridge = unflattened_cfg_nodes_encodings
            if len(self.encoder_decoder_bridge_dense_layers) > 0:
                encoded_cfg_nodes_after_bridge = functools.reduce(
                    lambda last_res, cur_layer: self.dropout_layer(self.activation_layer(cur_layer(last_res))),
                    self.encoder_decoder_bridge_dense_layers,
                    encoded_method_cfg.encoded_cfg_nodes.flatten(0, 1))\
                    .view(encoded_cfg_nodes_after_bridge.size()[:-1] + (-1,))
        elif self.encoder_params.method_encoder_type == 'whole-method':
            embedded_method_code: CodeExpressionEncodingsTensors = self.whole_method_code_embedder(
                encoded_identifiers=encoded_identifiers,
                tokenized_expressions_input=code_task_input.method_tokenized_code,
                method_ast_input=code_task_input.ast)
            whole_method_code_encoded: CodeExpressionEncodingsTensors = self.whole_method_code_encoder(
                previous_code_expression_encodings=embedded_method_code,
                tokenized_expressions_input=code_task_input.method_tokenized_code,
                sub_ast_input=code_task_input.ast)
            encoded_symbols = self.symbols_encoder(
                encoded_identifiers=encoded_identifiers,
                symbols=code_task_input.symbols,
                encoded_expressions=whole_method_code_encoded.token_seqs
                if self.encoder_params.symbols_encoder_params.use_symbols_occurrences else None,
                tokenized_expressions_input=code_task_input.method_tokenized_code,
                encoded_ast_nodes=whole_method_code_encoded.ast_nodes
                if self.encoder_params.symbols_encoder_params.use_symbols_occurrences else None,
                ast_nodes_with_symbol_leaf_nodes_indices=code_task_input.ast.ast_nodes_with_symbol_leaf_nodes_indices.indices,
                ast_nodes_with_symbol_leaf_symbol_idx=code_task_input.ast.ast_nodes_with_symbol_leaf_symbol_idx.indices)
        else:
            assert False

        return EncodedMethodCode(
            encoded_identifiers=encoded_identifiers,
            whole_method_ast_nodes_encoding=None if whole_method_code_encoded is None else whole_method_code_encoded.ast_nodes,
            whole_method_token_seqs_encoding=None if whole_method_code_encoded is None else whole_method_code_encoded.token_seqs,
            encoded_cfg_nodes=unflattened_cfg_nodes_encodings,
            encoded_symbols=encoded_symbols,
            encoded_cfg_nodes_after_bridge=encoded_cfg_nodes_after_bridge)
