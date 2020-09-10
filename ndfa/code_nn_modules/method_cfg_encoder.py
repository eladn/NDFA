import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, Optional
from torch_scatter import scatter_mean

from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_nn_modules.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.expression_encoder import ExpressionEncoder
from ndfa.code_nn_modules.expression_combiner import ExpressionCombiner
from ndfa.code_nn_modules.identifier_encoder import IdentifierEncoder
from ndfa.code_nn_modules.cfg_node_encoder import CFGNodeEncoder
from ndfa.code_nn_modules.cfg_paths_encoder import CFGPathEncoder
from ndfa.code_nn_modules.symbols_encoder import SymbolsEncoder
from ndfa.nn_utils.rnn_encoder import RNNEncoder
from ndfa.nn_utils.seq_context_adder import SeqContextAdder


__all__ = ['MethodCFGEncoder', 'EncodedMethodCFG']


class EncodedMethodCFG(NamedTuple):
    encoded_identifiers: torch.Tensor
    encoded_cfg_nodes: torch.Tensor
    encoded_symbols: torch.Tensor


class MethodCFGEncoder(nn.Module):
    def __init__(self, code_task_vocabs: CodeTaskVocabs, identifier_embedding_dim: int = 256,
                 expression_encoding_dim: int = 512, cfg_combined_expression_dim: int = 1024,
                 cfg_node_dim: int = 1024, dropout_p: float = 0.3, nr_layers: int = 2,
                 nr_rnn_layers: int = 2, use_symbols_occurrences_for_symbols_encodings: bool = True):
        super(MethodCFGEncoder, self).__init__()
        self.identifier_embedding_dim = identifier_embedding_dim
        self.expression_encoding_dim = expression_encoding_dim
        self.cfg_combined_expression_dim = cfg_combined_expression_dim
        self.cfg_node_dim = cfg_node_dim
        self.nr_rnn_layers = nr_rnn_layers
        self.identifier_encoder = IdentifierEncoder(
            sub_identifiers_vocab=code_task_vocabs.sub_identifiers,
            embedding_dim=self.identifier_embedding_dim)
        self.first_expression_encoder = ExpressionEncoder(
            kos_tokens_vocab=code_task_vocabs.kos_tokens,
            tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
            expressions_special_words_vocab=code_task_vocabs.expressions_special_words,
            identifiers_special_words_vocab=code_task_vocabs.identifiers_special_words,
            kos_token_embedding_dim=self.identifier_embedding_dim,
            expression_encoding_dim=self.expression_encoding_dim)
        self.expression_encoders = nn.ModuleList([
            RNNEncoder(
                input_dim=self.expression_encoding_dim,
                hidden_dim=self.expression_encoding_dim,
                nr_rnn_layers=self.nr_rnn_layers)
            for _ in range(nr_layers - 1)])
        self.expression_combiner = ExpressionCombiner(
            expression_encoding_dim=self.expression_encoding_dim,
            combined_expression_dim=self.cfg_combined_expression_dim,
            nr_attn_heads=8, nr_dim_reduction_layers=3)
        self.first_cfg_node_encoder = CFGNodeEncoder(
            cfg_node_dim=cfg_node_dim,
            cfg_combined_expression_dim=cfg_combined_expression_dim,
            pdg_node_control_kinds_vocab=code_task_vocabs.pdg_node_control_kinds)
        self.cfg_node_encoders = nn.ModuleList([
            CFGNodeEncoderExpressionUpdateLayer(
                cfg_node_dim=self.cfg_node_dim,
                cfg_combined_expression_dim=cfg_combined_expression_dim)
            for _ in range(nr_layers - 1)])
        self.cfg_path_encoders = nn.ModuleList([
            CFGPathEncoder(
                cfg_node_dim=cfg_combined_expression_dim)
            for _ in range(nr_layers)])
        self.scatter_cfg_encoded_paths_to_cfg_node_encodings = ScatterCFGEncodedPathsToCFGNodeEncodings()
        self.expression_context_adder = SeqContextAdder(
            main_dim=self.expression_encoding_dim,
            ctx_dim=self.cfg_combined_expression_dim)
        self.symbols_encoder = SymbolsEncoder(
            symbols_special_words_vocab=code_task_vocabs.symbols_special_words,
            symbol_embedding_dim=self.identifier_embedding_dim,
            expression_encoding_dim=self.expression_encoding_dim)  # it might change...
        self.dropout_layer = nn.Dropout(dropout_p)
        self.use_symbols_occurrences_for_symbols_encodings = \
            use_symbols_occurrences_for_symbols_encodings

    def forward(self, code_task_input: MethodCodeInputTensors) -> EncodedMethodCFG:
        # (nr_identifiers_in_batch, identifier_encoding_dim)
        encoded_identifiers = self.identifier_encoder(
            identifiers_sub_parts=code_task_input.identifiers_sub_parts,
            identifiers_sub_parts_hashings=code_task_input.identifiers_sub_parts_hashings)

        encoded_cfg_nodes = self.first_cfg_node_encoder(
            encoded_identifiers=encoded_identifiers, pdg=code_task_input.pdg)
        encoded_expressions = self.first_expression_encoder(
            expressions=code_task_input.pdg.cfg_nodes_tokenized_expressions,
            encoded_identifiers=encoded_identifiers)
        encoded_expressions_with_context = None
        for expression_encoder, cfg_node_encoder, cfg_path_encoder \
                in zip(itertools.chain((None,), self.expression_encoders),
                       itertools.chain((None,), self.cfg_node_encoders),
                       self.cfg_path_encoders):
            assert not (encoded_expressions_with_context is None) ^ (expression_encoder is None)
            if expression_encoder is not None:
                _, encoded_expressions = expression_encoder(
                    sequence_input=encoded_expressions_with_context,
                    lengths=code_task_input.pdg.cfg_nodes_tokenized_expressions.token_type.sequences_lengths,
                    batch_first=True)
            combined_expressions = self.expression_combiner(
                expressions_encodings=encoded_expressions,
                expressions_lengths=code_task_input.pdg.cfg_nodes_tokenized_expressions.token_type.sequences_lengths)
            if cfg_node_encoder is not None:
                encoded_cfg_nodes = self.dropout_layer(cfg_node_encoder(
                    previous_cfg_nodes_encodings=encoded_cfg_nodes,
                    cfg_combined_expressions_encodings=combined_expressions))
            encoded_cfg_paths = self.dropout_layer(cfg_path_encoder(
                cfg_nodes_encodings=encoded_cfg_nodes,
                cfg_paths_nodes_indices=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences,
                cfg_paths_lengths=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences_lengths))
            encoded_cfg_nodes = self.scatter_cfg_encoded_paths_to_cfg_node_encodings(
                encoded_cfg_paths=encoded_cfg_paths,
                cfg_paths_lengths=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences_lengths,
                cfg_paths_node_indices=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences)
            encoded_expressions_with_context = self.expression_context_adder(
                encoded_expressions=encoded_expressions,
                encoded_cfg_nodes=encoded_cfg_nodes[code_task_input.pdg.cfg_nodes_has_expression_mask])

        encoded_symbols = self.symbols_encoder(
            encoded_identifiers=encoded_identifiers,
            symbols=code_task_input.symbols,
            encoded_cfg_expressions=encoded_expressions_with_context
            if self.use_symbols_occurrences_for_symbols_encodings else None)

        return EncodedMethodCFG(
            encoded_identifiers=encoded_identifiers,
            encoded_cfg_nodes=encoded_cfg_nodes,
            encoded_symbols=encoded_symbols)


class CFGNodeEncoderExpressionUpdateLayer(nn.Module):
    def __init__(self, cfg_node_dim: int, cfg_combined_expression_dim: int):
        super(CFGNodeEncoderExpressionUpdateLayer, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.cfg_combined_expression_dim = cfg_combined_expression_dim

    def forward(self, previous_cfg_nodes_encodings: torch.Tensor,
                cfg_combined_expressions_encodings: torch.Tensor):
        raise NotImplementedError


class ScatterCFGEncodedPathsToCFGNodeEncodings(nn.Module):
    def __init__(self):
        super(ScatterCFGEncodedPathsToCFGNodeEncodings, self).__init__()

    def forward(self, encoded_cfg_paths: torch.tensor,
                cfg_paths_lengths: torch.LongTensor,
                cfg_paths_node_indices: torch.LongTensor):
        # TODO: complete impl! `encoded_cfg_paths` is in form of sequences. should flatten! should ignore edges (take only nodes)!
        raise NotImplementedError
        # cfg_nodes_encodings = scatter_mean(
        #     src=encoded_cfg_paths, index=cfg_paths_node_indices, dim=0)
        # return cfg_nodes_encodings
