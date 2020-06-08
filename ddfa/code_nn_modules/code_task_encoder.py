import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

from ddfa.code_nn_modules.code_task_input import CodeTaskInput
from ddfa.code_nn_modules.code_task_vocabs import CodeTaskVocabs
from ddfa.code_nn_modules.expression_encoder import ExpressionEncoder
from ddfa.code_nn_modules.identifier_encoder import IdentifierEncoder
from ddfa.code_nn_modules.cfg_node_encoder import CFGNodeEncoder
from ddfa.nn_utils.apply_batched_embeddings import apply_batched_embeddings


__all__ = ['CodeTaskEncoder', 'EncodedCode']


class EncodedCode(NamedTuple):
    encoded_identifiers: torch.Tensor
    encoded_cfg_nodes: torch.Tensor
    all_symbols_encodings: torch.Tensor
    encoded_cfg_nodes_after_bridge: torch.Tensor


class CodeTaskEncoder(nn.Module):
    def __init__(self, code_task_vocabs: CodeTaskVocabs, identifier_embedding_dim: int = 256,
                 expr_encoding_dim: int = 1028, nr_encoder_decoder_bridge_layers: int = 0):
        super(CodeTaskEncoder, self).__init__()
        self.identifier_embedding_dim = identifier_embedding_dim
        self.expr_encoding_dim = expr_encoding_dim
        self.identifier_encoder = IdentifierEncoder(
            sub_identifiers_vocab=code_task_vocabs.sub_identifiers, embedding_dim=self.identifier_embedding_dim)
        expression_encoder = ExpressionEncoder(
            tokens_vocab=code_task_vocabs.tokens, tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
            expressions_special_words_vocab=code_task_vocabs.expressions_special_words,
            identifiers_special_words_vocab=code_task_vocabs.identifiers_special_words,
            token_embedding_dim=self.identifier_embedding_dim, expr_encoding_dim=self.expr_encoding_dim)
        self.cfg_node_encoder = CFGNodeEncoder(
            expression_encoder=expression_encoder, pdg_node_control_kinds_vocab=code_task_vocabs.pdg_node_control_kinds)
        self.encoder_decoder_bridge_dense_layers = nn.ModuleList([
            nn.Linear(in_features=self.cfg_node_encoder.output_dim, out_features=self.cfg_node_encoder.output_dim)
            for _ in range(nr_encoder_decoder_bridge_layers)]) if nr_encoder_decoder_bridge_layers else None

    def forward(self, code_task_input: CodeTaskInput) -> EncodedCode:
        encoded_identifiers = self.identifier_encoder(
            sub_identifiers_indices=code_task_input.identifiers,
            sub_identifiers_mask=code_task_input.sub_identifiers_mask)  # (batch_size, nr_identifiers, identifier_encoding_dim)
        encoded_cfg_nodes = self.cfg_node_encoder(
            encoded_identifiers=encoded_identifiers, cfg_nodes_expressions=code_task_input.cfg_nodes_expressions,
            cfg_nodes_expressions_mask=code_task_input.cfg_nodes_expressions_mask,
            cfg_nodes_mask=code_task_input.cfg_nodes_mask,
            cfg_nodes_control_kind=code_task_input.cfg_nodes_control_kind)  # (batch_size, nr_cfg_nodes, cfg_node_encoding_dim)

        symbol_pad_embed = self.symbols_special_words_embedding(
            torch.tensor([self.vocabs.symbols_special_words.get_word_idx_or_unk('<PAD>')],
                         dtype=torch.long, device=encoded_identifiers.device)).view(-1)
        all_symbols_encodings = apply_batched_embeddings(
            batched_embeddings=encoded_identifiers, indices=code_task_input.identifiers_idxs_of_all_symbols,
            mask=code_task_input.identifiers_idxs_of_all_symbols_mask,
            padding_embedding_vector=symbol_pad_embed)  # (batch_size, nr_symbols, identifier_encoding_dim)
        assert all_symbols_encodings.size() == (code_task_input.batch_size, MAX_NR_SYMBOLS, self.identifier_embedding_dim)

        encoded_cfg_nodes_after_bridge = encoded_cfg_nodes
        if self.encoder_decoder_inbetween_dense_layers:
            encoded_cfg_nodes_after_bridge = functools.reduce(
                lambda last_res, cur_layer: F.relu(cur_layer(last_res)),
                self.encoder_decoder_inbetween_dense_layers,
                encoded_cfg_nodes.flatten(0, 1)).view(encoded_cfg_nodes.size()[:-1] + (-1,))

        return EncodedCode(
            encoded_identifiers=encoded_identifiers,
            encoded_cfg_nodes=encoded_cfg_nodes,
            all_symbols_encodings=all_symbols_encodings,
            encoded_cfg_nodes_after_bridge=encoded_cfg_nodes_after_bridge)
