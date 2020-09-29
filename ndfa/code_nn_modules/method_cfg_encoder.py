import itertools
import torch
import torch.nn as nn
from typing import NamedTuple
from torch_scatter import scatter_mean

from ndfa.ndfa_model_hyper_parameters import MethodCFGEncoderParams
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_nn_modules.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.expression_encoder import ExpressionEncoder
from ndfa.code_nn_modules.expression_combiner import ExpressionCombiner
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
    def __init__(self, code_task_vocabs: CodeTaskVocabs, identifier_embedding_dim: int,
                 symbol_embedding_dim: int, encoder_params: MethodCFGEncoderParams,
                 use_symbols_occurrences_for_symbols_encodings: bool,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu', nr_layers: int = 1,
                 nr_rnn_layers: int = 2):
        super(MethodCFGEncoder, self).__init__()
        assert nr_layers >= 1
        self.identifier_embedding_dim = identifier_embedding_dim
        self.symbol_embedding_dim = symbol_embedding_dim
        self.encoder_params = encoder_params
        self.nr_rnn_layers = nr_rnn_layers
        self.first_expression_encoder = ExpressionEncoder(
            kos_tokens_vocab=code_task_vocabs.kos_tokens,
            tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
            expressions_special_words_vocab=code_task_vocabs.expressions_special_words,
            identifiers_special_words_vocab=code_task_vocabs.identifiers_special_words,
            encoder_params=self.encoder_params.cfg_node_expression_encoder,
            identifier_embedding_dim=self.identifier_embedding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.expression_encoders = nn.ModuleList([
            RNNEncoder(
                input_dim=self.expression_encoding_dim,
                hidden_dim=self.expression_encoding_dim,
                nr_rnn_layers=self.nr_rnn_layers)
            for _ in range(nr_layers - 1)])
        self.expression_combiner = ExpressionCombiner(
            expression_encoding_dim=self.encoder_params.cfg_node_expression_encoder.token_encoding_dim,
            combined_expression_dim=self.encoder_params.cfg_node_expression_encoder.combined_expression_encoding_dim,
            nr_attn_heads=4, nr_dim_reduction_layers=3,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.first_cfg_node_encoder = CFGNodeEncoder(
            cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
            cfg_combined_expression_dim=self.encoder_params.cfg_node_expression_encoder.combined_expression_encoding_dim,
            pdg_node_control_kinds_vocab=code_task_vocabs.pdg_node_control_kinds,
            pdg_node_control_kinds_embedding_dim=self.encoder_params.cfg_node_control_kinds_embedding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.cfg_node_encoders = nn.ModuleList([
            CFGNodeEncoderExpressionUpdateLayer(
                cfg_node_dim=self.cfg_node_dim,
                cfg_combined_expression_dim=self.cfg_combined_expression_dim,
                dropout_rate=dropout_rate)
            for _ in range(nr_layers - 1)])
        self.cfg_path_encoders = nn.ModuleList([
            CFGPathEncoder(cfg_node_dim=self.encoder_params.cfg_node_encoding_dim)
            for _ in range(nr_layers)])
        self.scatter_cfg_encoded_paths_to_cfg_node_encodings = ScatterCFGEncodedPathsToCFGNodeEncodings()
        self.expression_context_adder = SeqContextAdder(
            main_dim=self.encoder_params.cfg_node_expression_encoder.token_encoding_dim,
            ctx_dim=self.encoder_params.cfg_node_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.symbols_encoder = SymbolsEncoder(
            symbols_special_words_vocab=code_task_vocabs.symbols_special_words,
            symbol_embedding_dim=self.symbol_embedding_dim,
            expression_encoding_dim=self.encoder_params.cfg_node_expression_encoder.token_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.use_symbols_occurrences_for_symbols_encodings = \
            use_symbols_occurrences_for_symbols_encodings

    def forward(self, code_task_input: MethodCodeInputTensors, encoded_identifiers: torch.Tensor) -> EncodedMethodCFG:
        encoded_expressions_with_context = None
        # TODO: add skip-connections [AddNorm]
        for expression_encoder, cfg_node_encoder, cfg_path_encoder \
                in zip(itertools.chain((None,), self.expression_encoders),
                       itertools.chain((None,), self.cfg_node_encoders),
                       self.cfg_path_encoders):
            assert not (encoded_expressions_with_context is None) ^ (expression_encoder is None)
            if expression_encoder is None:
                encoded_expressions = self.first_expression_encoder(
                    expressions=code_task_input.pdg.cfg_nodes_tokenized_expressions,
                    encoded_identifiers=encoded_identifiers)
            else:
                _, new_encoded_expressions = expression_encoder(
                    sequence_input=encoded_expressions_with_context,
                    lengths=code_task_input.pdg.cfg_nodes_tokenized_expressions.token_type.sequences_lengths,
                    batch_first=True)
                # TODO: use AddNorm for skip-connections here
                encoded_expressions = encoded_expressions + new_encoded_expressions
            combined_expressions = self.expression_combiner(
                expressions_encodings=encoded_expressions,
                expressions_lengths=code_task_input.pdg.cfg_nodes_tokenized_expressions.token_type.sequences_lengths)
            if cfg_node_encoder is None:
                encoded_cfg_nodes = self.first_cfg_node_encoder(
                    combined_cfg_expressions_encodings=combined_expressions, pdg=code_task_input.pdg)
            else:
                assert encoded_cfg_nodes is not None
                new_encoded_cfg_nodes = cfg_node_encoder(
                    previous_cfg_nodes_encodings=encoded_cfg_nodes,
                    cfg_combined_expressions_encodings=combined_expressions,
                    cfg_nodes_has_expression_mask=code_task_input.pdg.cfg_nodes_has_expression_mask.tensor)
                encoded_cfg_nodes = encoded_cfg_nodes + new_encoded_cfg_nodes  # TODO: use AddNorm for skip-connections here

            if self.encoder_params.encoder_type == 'control-flow-paths-folded-to-nodes':
                # TODO: use AddNorm for skip-connections here
                encoded_cfg_paths = self.dropout_layer(cfg_path_encoder(
                    cfg_nodes_encodings=encoded_cfg_nodes,
                    cfg_paths_nodes_indices=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences,
                    cfg_paths_lengths=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences_lengths))
                # TODO: use AddNorm for skip-connections here
                encoded_cfg_nodes = encoded_cfg_nodes + self.scatter_cfg_encoded_paths_to_cfg_node_encodings(
                    encoded_cfg_paths=encoded_cfg_paths,
                    cfg_paths_mask=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences_mask,
                    cfg_paths_node_indices=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences,
                    nr_cfg_nodes=code_task_input.pdg.cfg_nodes_has_expression_mask.batch_size)
            elif self.encoder_params.encoder_type == 'all-nodes-single-seq':
                raise NotImplementedError  # TODO: impl
            elif self.encoder_params.encoder_type == 'set-of-control-flow-paths':
                raise NotImplementedError  # TODO: impl
            elif self.encoder_params.encoder_type == 'gnn':
                raise NotImplementedError  # TODO: impl
            elif self.encoder_params.encoder_type == 'control-flow-paths-ngrams':
                raise NotImplementedError  # TODO: impl
            elif self.encoder_params.encoder_type == 'set-of-nodes':
                raise NotImplementedError  # TODO: impl
            else:
                raise ValueError(f'Unsupported method-CFG encoding type `{self.encoder_params.encoder_type}`.')

            encoded_expressions_with_context = self.expression_context_adder(
                sequence=encoded_expressions,
                sequence_mask=code_task_input.pdg.cfg_nodes_tokenized_expressions.token_type.sequences_mask,
                context=encoded_cfg_nodes[code_task_input.pdg.cfg_nodes_has_expression_mask.tensor])

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
    def __init__(self, cfg_node_dim: int, cfg_combined_expression_dim: int, dropout_rate: float = 0.3):
        super(CFGNodeEncoderExpressionUpdateLayer, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.cfg_combined_expression_dim = cfg_combined_expression_dim
        self.projection_layer = nn.Linear(
            in_features=self.cfg_node_dim + self.cfg_combined_expression_dim, out_features=self.cfg_node_dim)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, previous_cfg_nodes_encodings: torch.Tensor,
                cfg_combined_expressions_encodings: torch.Tensor,
                cfg_nodes_has_expression_mask: torch.BoolTensor):
        # TODO: consider adding another layer
        new_cfg_node_embeddings_for_nodes_with_expressions = self.dropout_layer(self.activation_layer(self.projection_layer(torch.cat([
            previous_cfg_nodes_encodings[cfg_nodes_has_expression_mask], cfg_combined_expressions_encodings], dim=-1))))
        return previous_cfg_nodes_encodings.masked_scatter(
            cfg_nodes_has_expression_mask.unsqueeze(-1).expand(previous_cfg_nodes_encodings.size()),
            new_cfg_node_embeddings_for_nodes_with_expressions)


class ScatterCFGEncodedPathsToCFGNodeEncodings(nn.Module):
    def __init__(self):
        super(ScatterCFGEncodedPathsToCFGNodeEncodings, self).__init__()

    def forward(self, encoded_cfg_paths: torch.tensor,
                cfg_paths_mask: torch.BoolTensor,
                cfg_paths_node_indices: torch.LongTensor,
                nr_cfg_nodes: int):
        # `encoded_cfg_paths` is in form of sequences. We flatten it by applying a mask selector.
        # The mask also helps to ignore paddings.
        # TODO: ignore edges (take only nodes)! after we actually add the edges..
        cfg_nodes_encodings = scatter_mean(
            src=encoded_cfg_paths[cfg_paths_mask],
            index=cfg_paths_node_indices[cfg_paths_mask],
            dim=0, dim_size=nr_cfg_nodes)
        assert cfg_nodes_encodings.size() == (nr_cfg_nodes, encoded_cfg_paths.size(2))
        return cfg_nodes_encodings
