import functools
import torch
import torch.nn as nn
from typing import NamedTuple, Optional

from ndfa.ndfa_model_hyper_parameters import MethodCodeEncoderParams
from ndfa.nn_utils.misc import get_activation_layer
from ndfa.nn_utils.scattered_encodings import ScatteredEncodings
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_nn_modules.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.identifier_encoder import IdentifierEncoder
from ndfa.code_nn_modules.method_cfg_encoder import MethodCFGEncoder, EncodedMethodCFG
from ndfa.code_nn_modules.expression_encoder import ExpressionEncoder


__all__ = ['MethodCodeEncoder', 'EncodedMethodCode']


class EncodedMethodCode(NamedTuple):
    encoded_identifiers: torch.Tensor
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
                use_symbols_occurrences_for_symbols_encodings=
                self.encoder_params.use_symbols_occurrences_for_symbols_encodings,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.method_encoder_type == 'method-linear-seq':
            self.linear_seq_method_code_encoder = ExpressionEncoder(
                kos_tokens_vocab=code_task_vocabs.kos_tokens,
                tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
                expressions_special_words_vocab=code_task_vocabs.expressions_special_words,
                identifiers_special_words_vocab=code_task_vocabs.identifiers_special_words,
                encoder_params=self.encoder_params.method_linear_seq_expression_encoder_type,
                identifier_embedding_dim=self.encoder_params.identifier_encoder.identifier_embedding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        else:
            raise ValueError(f'Unexpected method code encoder type `{self.encoder_params.method_encoder_type}`.')

        # TODO: remove!
        # expression_encoder = ExpressionEncoder(
        #     kos_tokens_vocab=code_task_vocabs.kos_tokens, tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
        #     expressions_special_words_vocab=code_task_vocabs.expressions_special_words,
        #     identifiers_special_words_vocab=code_task_vocabs.identifiers_special_words,
        #     kos_token_embedding_dim=self.identifier_embedding_dim, expression_encoding_dim=self.expr_encoding_dim)
        # self.cfg_node_encoder = CFGNodeEncoder(
        #     expression_encoder=expression_encoder, pdg_node_control_kinds_vocab=code_task_vocabs.pdg_node_control_kinds)
        # self.symbols_encoder = SymbolsEncoder(
        #     symbols_special_words_vocab=code_task_vocabs.symbols_special_words,
        #     symbol_embedding_dim=self.identifier_embedding_dim,
        #     expression_encoding_dim=self.expr_encoding_dim)  # it might change...

        self.encoder_decoder_bridge_dense_layers = nn.ModuleList([
            nn.Linear(in_features=self.encoder_params.method_cfg_encoder.cfg_node_encoding_dim,
                      out_features=self.encoder_params.method_cfg_encoder.cfg_node_encoding_dim)  # TODO: make it abstract `encoder.output_dim` to support additional encoders
            for _ in range(nr_encoder_decoder_bridge_layers)])
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, code_task_input: MethodCodeInputTensors) -> EncodedMethodCode:
        # (nr_identifiers_in_batch, identifier_encoding_dim)
        encoded_identifiers = self.identifier_encoder(
            identifiers_sub_parts=code_task_input.identifiers_sub_parts,
            identifiers_sub_parts_hashings=code_task_input.identifiers_sub_parts_hashings)  # (nr_identifiers_in_batch, identifier_encoding_dim)

        encoded_method_cfg: EncodedMethodCFG = self.method_cfg_encoder(
            code_task_input=code_task_input, encoded_identifiers=encoded_identifiers)

        unflattened_cfg_nodes_encodings = code_task_input.pdg.cfg_nodes_control_kind.unflatten(
            encoded_method_cfg.encoded_cfg_nodes)

        encoded_cfg_nodes_after_bridge = unflattened_cfg_nodes_encodings
        if len(self.encoder_decoder_bridge_dense_layers) > 0:
            encoded_cfg_nodes_after_bridge = functools.reduce(
                lambda last_res, cur_layer: self.dropout_layer(self.activation_layer(cur_layer(last_res))),
                self.encoder_decoder_bridge_dense_layers,
                encoded_method_cfg.encoded_cfg_nodes.flatten(0, 1))\
                .view(encoded_cfg_nodes_after_bridge.size()[:-1] + (-1,))

        return EncodedMethodCode(
            encoded_identifiers=encoded_identifiers,
            encoded_cfg_nodes=unflattened_cfg_nodes_encodings,
            encoded_symbols=encoded_method_cfg.encoded_symbols,
            encoded_cfg_nodes_after_bridge=encoded_cfg_nodes_after_bridge)

        # TODO: REMOVE!
        # encoded_cfg_nodes: EncodedCFGNode = self.cfg_node_encoder(
        #     encoded_identifiers=encoded_identifiers,
        #     pdg=code_task_input.pdg)  # (nr_cfg_nodes_in_batch, cfg_node_encoding_dim)
        # encoded_symbols = self.symbols_encoder(
        #     encoded_identifiers=encoded_identifiers,
        #     symbols=code_task_input.symbols,
        #     encoded_cfg_expressions=encoded_cfg_nodes.encoded_cfg_nodes_expressions
        #     if self.use_copy_attn_with_symbols_occurrences_in_cfg_expressions else None)
        # return EncodedMethodCode(
        #     encoded_identifiers=encoded_identifiers,
        #     encoded_cfg_nodes=encoded_cfg_nodes.encoded_cfg_nodes,
        #     encoded_symbols=encoded_symbols,
        #     encoded_cfg_nodes_after_bridge=encoded_cfg_nodes_after_bridge)
