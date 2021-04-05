import torch
import torch.nn as nn
from typing import NamedTuple, Dict, Optional, Mapping

from ndfa.nn_utils.misc.misc import get_activation_layer
from ndfa.code_nn_modules.params.method_cfg_encoder_params import MethodCFGEncoderParams
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors, PDGInputTensors, \
    CFGPathsNGramsInputTensors, PDGExpressionsSubASTInputTensors
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.cfg_node_encoder import CFGNodeEncoder
from ndfa.code_nn_modules.cfg_paths_encoder import CFGPathEncoder, EncodedCFGPaths
from ndfa.code_nn_modules.symbols_encoder import SymbolsEncoder
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.nn_utils.modules.seq_context_adder import SeqContextAdder
from ndfa.nn_utils.modules.scatter_combiner import ScatterCombiner
from ndfa.nn_utils.modules.params.sequence_encoder_params import SequenceEncoderParams
from ndfa.code_nn_modules.code_task_input import SymbolsInputTensors
from ndfa.nn_utils.functions.weave_tensors import weave_tensors, unweave_tensor
from ndfa.nn_utils.modules.state_updater import StateUpdater
from ndfa.nn_utils.modules.norm_wrapper import NormWrapper
from ndfa.nn_utils.modules.module_repeater import ModuleRepeater
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.code_nn_modules.code_expression_encoder import CodeExpressionEncoder
from ndfa.code_nn_modules.code_expression_embedder import CodeExpressionEmbedder
from ndfa.code_nn_modules.code_expression_combiner import CodeExpressionCombiner
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.code_nn_modules.cfg_gnn_encoder import CFGGNNEncoder
from ndfa.code_nn_modules.params.symbols_encoder_params import SymbolsEncoderParams
from ndfa.nn_utils.modules.params.scatter_combiner_params import ScatterCombinerParams


__all__ = ['MethodCFGEncoderV2', 'EncodedMethodCFGV2']


class EncodedMethodCFGV2(NamedTuple):
    encoded_identifiers: torch.Tensor
    encoded_cfg_nodes: torch.Tensor
    encoded_symbols: torch.Tensor


class MethodCFGEncoderV2(nn.Module):
    def __init__(self, code_task_vocabs: CodeTaskVocabs, identifier_embedding_dim: int,
                 symbol_embedding_dim: int, encoder_params: MethodCFGEncoderParams,
                 symbols_encoder_params: SymbolsEncoderParams,
                 use_norm: bool = True, affine_norm: bool = False, norm_type: str = 'layer',  # TODO: put in HPs
                 share_norm_between_usage_points: bool = True,  # TODO: put in HPs
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(MethodCFGEncoderV2, self).__init__()
        self.identifier_embedding_dim = identifier_embedding_dim
        self.symbol_embedding_dim = symbol_embedding_dim
        self.encoder_params = encoder_params

        self.code_expression_embedder = CodeExpressionEmbedder(
            code_task_vocabs=code_task_vocabs,
            encoder_params=self.encoder_params.cfg_node_expression_encoder,
            identifier_embedding_dim=self.identifier_embedding_dim,
            nr_final_embeddings_linear_layers=1,  # TODO: plug HP here
            dropout_rate=dropout_rate, activation_fn=activation_fn)

        self.code_expression_encoders_before_macro = nn.ModuleList([
            CodeExpressionEncoder(
                encoder_params=self.encoder_params.cfg_node_expression_encoder,
                code_task_vocabs=code_task_vocabs,
                identifier_embedding_dim=self.identifier_embedding_dim,
                is_first_encoder_layer=True,
                dropout_rate=dropout_rate, activation_fn=activation_fn),
            # CodeExpressionEncoder(
            #     encoder_params=self.encoder_params.cfg_node_expression_encoder,
            #     code_task_vocabs=code_task_vocabs,
            #     identifier_embedding_dim=self.identifier_embedding_dim,
            #     is_first_encoder_layer=False,
            #     dropout_rate=dropout_rate, activation_fn=activation_fn),
            # CodeExpressionEncoder(
            #     encoder_params=self.encoder_params.cfg_node_expression_encoder,
            #     code_task_vocabs=code_task_vocabs,
            #     identifier_embedding_dim=self.identifier_embedding_dim,
            #     is_first_encoder_layer=False,
            #     dropout_rate=dropout_rate, activation_fn=activation_fn)
        ])

        self.code_expression_encoders_after_macro = nn.ModuleList([
            CodeExpressionEncoder(
                encoder_params=self.encoder_params.cfg_node_expression_encoder,
                code_task_vocabs=code_task_vocabs,
                identifier_embedding_dim=self.identifier_embedding_dim,
                is_first_encoder_layer=False,
                dropout_rate=dropout_rate, activation_fn=activation_fn),
            # CodeExpressionEncoder(
            #     encoder_params=self.encoder_params.cfg_node_expression_encoder,
            #     code_task_vocabs=code_task_vocabs,
            #     identifier_embedding_dim=self.identifier_embedding_dim,
            #     is_first_encoder_layer=False,
            #     dropout_rate=dropout_rate, activation_fn=activation_fn),
            # CodeExpressionEncoder(
            #     encoder_params=self.encoder_params.cfg_node_expression_encoder,
            #     code_task_vocabs=code_task_vocabs,
            #     identifier_embedding_dim=self.identifier_embedding_dim,
            #     is_first_encoder_layer=False,
            #     dropout_rate=dropout_rate, activation_fn=activation_fn)
        ])

        self.code_expression_combiners_before_macro = nn.ModuleList([
            CodeExpressionCombiner(
                encoder_params=self.encoder_params.cfg_node_expression_encoder,
                tokenized_expression_combiner_params=self.encoder_params.cfg_node_tokenized_expression_combiner,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for _ in range(3)])
        self.code_expression_combiners_after_macro = nn.ModuleList([
            CodeExpressionCombiner(
                encoder_params=self.encoder_params.cfg_node_expression_encoder,
                tokenized_expression_combiner_params=self.encoder_params.cfg_node_tokenized_expression_combiner,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for _ in range(3)])

        if self.encoder_params.cfg_node_expression_encoder.encoder_type == 'tokens-seq':
            self.tokenized_expression_context_adder = SeqContextAdder(
                main_dim=self.encoder_params.cfg_node_expression_encoder.tokens_seq_encoder.token_encoding_dim,
                ctx_dim=self.encoder_params.cfg_node_encoding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.cfg_node_expression_encoder.encoder_type == 'ast':
            self.macro_context_adder_to_sub_ast = MacroContextAdderToSubAST(
                ast_node_encoding_dim=self.encoder_params.cfg_node_expression_encoder.ast_encoder.ast_node_embedding_dim,
                cfg_node_encoding_dim=self.encoder_params.cfg_node_encoding_dim)
        else:
            raise ValueError(
                f'Unsupported expression encoder type '
                f'`{self.encoder_params.cfg_node_expression_encoder.encoder_type}`.')

        self.cfg_node_encoder = CFGNodeEncoder(
            cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
            cfg_combined_expression_dim=self.encoder_params.cfg_node_expression_encoder.combined_expression_encoding_dim,
            pdg_node_control_kinds_vocab=code_task_vocabs.pdg_node_control_kinds,
            pdg_node_control_kinds_embedding_dim=self.encoder_params.cfg_node_control_kinds_embedding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

        self.cfg_node_encoding_mixer_with_expression_encoding = \
            CFGNodeEncodingMixerWithExpressionEncoding(
                cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
                cfg_combined_expression_dim=self.encoder_params.cfg_node_expression_encoder.combined_expression_encoding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)

        if self.encoder_params.encoder_type in {'pdg-paths-folded-to-nodes', 'control-flow-paths-folded-to-nodes', 'set-of-control-flow-paths'}:
            self.cfg_paths_encoder = CFGPathEncoder(
                cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
                cfg_paths_sequence_encoder_params=self.encoder_params.cfg_paths_sequence_encoder,
                control_flow_edge_types_vocab=code_task_vocabs.pdg_control_flow_edge_types,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        if self.encoder_params.encoder_type in \
                {'control-flow-paths-ngrams-folded-to-nodes', 'set-of-control-flow-paths-ngrams'}:
            self.cfg_paths_ngrams_encoder = CFGPathsNGramsEncoder(
                cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
                cfg_paths_sequence_encoder_params=self.encoder_params.cfg_paths_sequence_encoder,
                control_flow_edge_types_vocab=code_task_vocabs.pdg_control_flow_edge_types, is_first=True,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        if self.encoder_params.encoder_type == 'gnn':
            self.cfg_gnn_encoder = CFGGNNEncoder(
                cfg_node_encoding_dim=self.encoder_params.cfg_node_encoding_dim,
                encoder_params=self.encoder_params.cfg_gnn_encoder,
                pdg_control_flow_edge_types_vocab=code_task_vocabs.pdg_control_flow_edge_types,
                norm_layer_ctor=lambda: NormWrapper(
                    self.encoder_params.cfg_node_encoding_dim,
                    affine=affine_norm, norm_type=norm_type),
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        if self.encoder_params.encoder_type == 'control-flow-paths-ngrams-folded-to-nodes':
            self.scatter_cfg_encoded_ngrams_to_cfg_node_encodings = \
                ScatterCFGEncodedNGramsToCFGNodeEncodings(
                    cfg_node_encoding_dim=self.encoder_params.cfg_node_encoding_dim,
                    cfg_nodes_folding_params=self.encoder_params.cfg_nodes_folding_params,
                    dropout_rate=dropout_rate, activation_fn=activation_fn)
        if self.encoder_params.encoder_type in {'control-flow-paths-folded-to-nodes', 'pdg-paths-folded-to-nodes'}:
            self.scatter_cfg_encoded_paths_to_cfg_node_encodings = \
                ScatterCFGEncodedPathsToCFGNodeEncodings(
                    cfg_node_encoding_dim=self.encoder_params.cfg_node_encoding_dim,
                    cfg_nodes_folding_params=self.encoder_params.cfg_nodes_folding_params,
                    dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.encoder_type in {'all-nodes-single-unstructured-linear-seq',
                                                  'all-nodes-single-random-permutation-seq'}:
            cfg_single_path_sequence_type = \
                'linear' \
                    if self.encoder_params.encoder_type == 'all-nodes-single-unstructured-linear-seq' else \
                    'random-permutation'
            self.cfg_single_path_encoder = CFGSinglePathEncoder(
                cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
                sequence_type=cfg_single_path_sequence_type,
                cfg_paths_sequence_encoder_params=self.encoder_params.cfg_paths_sequence_encoder,
                dropout_rate=dropout_rate, activation_fn=activation_fn)

        # TODO: put in HPs
        expression_encoding_dim = \
            self.encoder_params.cfg_node_expression_encoder.tokens_seq_encoder.token_encoding_dim \
                if self.encoder_params.cfg_node_expression_encoder.encoder_type == 'tokens-seq' else \
                self.encoder_params.cfg_node_expression_encoder.ast_encoder.ast_node_embedding_dim

        self.symbols_encoder = SymbolsEncoder(
            identifier_embedding_dim=self.identifier_embedding_dim,
            symbol_embedding_dim=self.symbol_embedding_dim,
            expression_encoding_dim=expression_encoding_dim,
            encoder_params=symbols_encoder_params,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.add_symbols_encodings_to_expressions = AddSymbolsEncodingsToExpressions(
            expression_token_encoding_dim=expression_encoding_dim,
            symbol_encoding_dim=self.symbol_embedding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

        self.use_norm = use_norm
        if self.use_norm:
            self.expressions_norm = ModuleRepeater(
                module_create_fn=lambda: NormWrapper(
                    expression_encoding_dim,
                    affine=affine_norm, norm_type=norm_type),
                repeats=3, share=share_norm_between_usage_points, repeat_key='usage_point')
            self.combined_expressions_norm = NormWrapper(
                self.encoder_params.cfg_node_expression_encoder.combined_expression_encoding_dim,
                affine=affine_norm, norm_type=norm_type)
            self.symbols_norm = NormWrapper(
                self.symbol_embedding_dim,
                affine=affine_norm, norm_type=norm_type)
            self.cfg_nodes_norm = ModuleRepeater(
                module_create_fn=lambda: NormWrapper(
                    self.encoder_params.cfg_node_encoding_dim,
                    affine=affine_norm, norm_type=norm_type),
                repeats=2, share=share_norm_between_usage_points, repeat_key='usage_point')

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def apply_expression_encoder(
            self, code_task_input: MethodCodeInputTensors,
            previous_expression_encodings: CodeExpressionEncodingsTensors,
            expression_encoder: CodeExpressionEncoder,
            expression_combiner: CodeExpressionCombiner) -> CodeExpressionEncodingsTensors:
        encoded_code_expressions: CodeExpressionEncodingsTensors = expression_encoder(
            previous_code_expression_encodings=previous_expression_encodings,
            tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
            sub_ast_input=code_task_input.pdg.cfg_nodes_expressions_ast)
        if self.use_norm:
            if self.encoder_params.cfg_node_expression_encoder.encoder_type == 'tokens-seq':
                encoded_code_expressions.token_seqs = self.expressions_norm(
                    encoded_code_expressions.token_seqs, usage_point=1)
            elif self.encoder_params.cfg_node_expression_encoder.encoder_type in {'ast_paths', 'ast_treelstm'}:
                encoded_code_expressions.ast_nodes = self.expressions_norm(
                    encoded_code_expressions.ast_nodes, usage_point=1)

        encoded_code_expressions.combined_expressions = expression_combiner(
            encoded_code_expressions=encoded_code_expressions,
            tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
            cfg_nodes_expressions_ast=code_task_input.pdg.cfg_nodes_expressions_ast,
            cfg_nodes_has_expression_mask=code_task_input.pdg.cfg_nodes_has_expression_mask.tensor)
        if self.use_norm:
            encoded_code_expressions.combined_expressions = self.combined_expressions_norm(
                encoded_code_expressions.combined_expressions)

        return encoded_code_expressions

    def forward(self, code_task_input: MethodCodeInputTensors, encoded_identifiers: torch.Tensor) -> EncodedMethodCFGV2:
        encoded_cfg_paths, encoded_cfg_paths_ngrams = None, None

        embedded_code_expressions: CodeExpressionEncodingsTensors = self.code_expression_embedder(
            encoded_identifiers=encoded_identifiers,
            tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
            method_ast_input=code_task_input.ast)
        if self.use_norm:
            if self.encoder_params.cfg_node_expression_encoder.encoder_type == 'tokens-seq':
                embedded_code_expressions.token_seqs = self.expressions_norm(
                    embedded_code_expressions.token_seqs, usage_point=0)
            elif self.encoder_params.cfg_node_expression_encoder.encoder_type in {'ast_paths', 'ast_treelstm'}:
                embedded_code_expressions.ast_nodes = self.expressions_norm(
                    embedded_code_expressions.ast_nodes, usage_point=0)

        encoded_code_expressions = embedded_code_expressions
        for encoder, combiner in \
                zip(self.code_expression_encoders_before_macro, self.code_expression_combiners_before_macro):
            encoded_code_expressions = self.apply_expression_encoder(
                code_task_input=code_task_input,
                previous_expression_encodings=encoded_code_expressions,
                expression_encoder=encoder,
                expression_combiner=combiner)

        encoded_cfg_nodes = self.cfg_node_encoder(
            combined_cfg_expressions_encodings=encoded_code_expressions.combined_expressions,
            pdg=code_task_input.pdg)
        if self.use_norm:
            encoded_cfg_nodes = self.cfg_nodes_norm(encoded_cfg_nodes, usage_point=0)

        if self.encoder_params.encoder_type in {'control-flow-paths-folded-to-nodes', 'pdg-paths-folded-to-nodes', 'set-of-control-flow-paths'}:
            cfg_paths_input = code_task_input.pdg.cfg_pdg_paths \
                if self.encoder_params.encoder_type == 'pdg-paths-folded-to-nodes' else \
                code_task_input.pdg.cfg_control_flow_paths
            encoded_cfg_paths = self.cfg_paths_encoder(
                cfg_nodes_encodings=encoded_cfg_nodes,
                cfg_paths_nodes_indices=cfg_paths_input.nodes_indices.sequences,
                cfg_paths_edge_types=cfg_paths_input.edges_types.sequences,
                cfg_paths_lengths=cfg_paths_input.nodes_indices.sequences_lengths)
        if self.encoder_params.encoder_type in \
                {'control-flow-paths-ngrams-folded-to-nodes', 'set-of-control-flow-paths-ngrams'}:
            ngrams_min_n = self.encoder_params.cfg_paths_ngrams_min_n
            ngrams_max_n = self.encoder_params.cfg_paths_ngrams_max_n
            all_ngram_ns = \
                set(code_task_input.pdg.cfg_control_flow_paths_exact_ngrams.keys()) | \
                set(code_task_input.pdg.cfg_control_flow_paths_partial_ngrams.keys())
            assert len(all_ngram_ns) > 0
            ngrams_n_range_start = min(all_ngram_ns) if ngrams_min_n is None else max(min(all_ngram_ns), ngrams_min_n)
            ngrams_n_range_end = max(all_ngram_ns) if ngrams_max_n is None else min(max(all_ngram_ns), ngrams_max_n)
            assert ngrams_n_range_start <= ngrams_n_range_end
            cfg_control_flow_paths_ngrams_input = {
                ngram_n: ngrams
                for ngram_n, ngrams in code_task_input.pdg.cfg_control_flow_paths_exact_ngrams.items()
                if ngrams_n_range_start <= ngram_n <= ngrams_n_range_end}
            if ngrams_max_n in code_task_input.pdg.cfg_control_flow_paths_partial_ngrams:
                cfg_control_flow_paths_ngrams_input[ngrams_max_n + 1] = \
                    code_task_input.pdg.cfg_control_flow_paths_partial_ngrams[ngrams_max_n]

            encoded_cfg_paths_ngrams = self.cfg_paths_ngrams_encoder(
                cfg_nodes_encodings=encoded_cfg_nodes,
                cfg_control_flow_paths_ngrams_input=cfg_control_flow_paths_ngrams_input)

        if self.encoder_params.encoder_type in {'control-flow-paths-folded-to-nodes', 'pdg-paths-folded-to-nodes'}:
            encoded_cfg_nodes = self.scatter_cfg_encoded_paths_to_cfg_node_encodings(
                encoded_cfg_node_occurrences_in_paths=encoded_cfg_paths.nodes_occurrences,
                cfg_paths_mask=cfg_paths_input.nodes_indices.sequences_mask,
                cfg_paths_node_indices=cfg_paths_input.nodes_indices.sequences,
                previous_cfg_nodes_encodings=encoded_cfg_nodes,
                nr_cfg_nodes=code_task_input.pdg.cfg_nodes_has_expression_mask.batch_size)
            if self.use_norm:
                encoded_cfg_nodes = self.cfg_nodes_norm(encoded_cfg_nodes, usage_point=1)
        elif self.encoder_params.encoder_type in {'all-nodes-single-unstructured-linear-seq',
                                                  'all-nodes-single-random-permutation-seq'}:
            encoded_cfg_nodes = self.cfg_single_path_encoder(
                pdg_input=code_task_input.pdg,
                cfg_nodes_encodings=encoded_cfg_nodes)
            if self.use_norm:
                encoded_cfg_nodes = self.cfg_nodes_norm(encoded_cfg_nodes, usage_point=1)
        elif self.encoder_params.encoder_type == 'set-of-control-flow-paths':
            raise NotImplementedError  # TODO: impl
        elif self.encoder_params.encoder_type == 'gnn':
            encoded_cfg_nodes = self.cfg_gnn_encoder(
                encoded_cfg_nodes=encoded_cfg_nodes,
                cfg_control_flow_graph=code_task_input.pdg.cfg_control_flow_graph)
        elif self.encoder_params.encoder_type == 'set-of-control-flow-paths-ngrams':
            raise NotImplementedError  # TODO: impl
        elif self.encoder_params.encoder_type == 'control-flow-paths-ngrams-folded-to-nodes':
            encoded_cfg_nodes = self.scatter_cfg_encoded_ngrams_to_cfg_node_encodings(
                encoded_cfg_paths_ngrams=encoded_cfg_paths_ngrams,
                cfg_control_flow_paths_ngrams_input=cfg_control_flow_paths_ngrams_input,
                previous_cfg_nodes_encodings=encoded_cfg_nodes,
                nr_cfg_nodes=code_task_input.pdg.cfg_nodes_has_expression_mask.batch_size)
            if self.use_norm:
                encoded_cfg_nodes = self.cfg_nodes_norm(encoded_cfg_nodes, usage_point=1)
        elif self.encoder_params.encoder_type == 'set-of-nodes':
            pass  # We actually do not need to do anything in this case.
        else:
            raise ValueError(f'Unsupported method-CFG encoding type `{self.encoder_params.encoder_type}`.')

        # Add CFG-node macro context to its own sub-AST.
        if self.encoder_params.cfg_node_expression_encoder.encoder_type == 'tokens-seq':
            # TODO: maybe we do not need this for `self.encoder_params.encoder_type == 'set-of-nodes'`
            # FIXME: notice there is a skip-connection built-in here that is not conditioned
            #  by the flag `self.use_skip_connections`.
            encoded_code_expressions.token_seqs = self.tokenized_expression_context_adder(
                sequence=encoded_code_expressions.token_seqs,
                sequence_mask=code_task_input.pdg.cfg_nodes_tokenized_expressions.token_type.sequences_mask,
                context=encoded_cfg_nodes[code_task_input.pdg.cfg_nodes_has_expression_mask.tensor])
            if self.use_norm:
                encoded_code_expressions.token_seqs = self.expressions_norm(
                    encoded_code_expressions.token_seqs, usage_point=2)
        elif self.encoder_params.cfg_node_expression_encoder.encoder_type == 'ast':
            encoded_code_expressions.ast_nodes = self.macro_context_adder_to_sub_ast(
                previous_ast_nodes_encodings=encoded_code_expressions.ast_nodes,
                new_cfg_nodes_encodings=encoded_cfg_nodes,
                cfg_expressions_sub_ast_input=code_task_input.pdg.cfg_nodes_expressions_ast)
            if self.use_norm:
                encoded_code_expressions.ast_nodes = self.expressions_norm(
                    encoded_code_expressions.ast_nodes, usage_point=2)
        else:
            assert False

        # TODO: do we want to apply this?
        # encoded_cfg_nodes = self.cfg_node_encoding_mixer_with_expression_encoding(
        #     previous_cfg_nodes_encodings=encoded_cfg_nodes,
        #     cfg_combined_expressions_encodings=combined_expressions,
        #     cfg_nodes_has_expression_mask=code_task_input.pdg.cfg_nodes_has_expression_mask.tensor)
        # if self.use_norm:
        #     encoded_cfg_nodes = self.cfg_nodes_norm(encoded_cfg_nodes, usage_point=0)

        for encoder, combiner in \
                zip(self.code_expression_encoders_after_macro, self.code_expression_combiners_after_macro):
            encoded_code_expressions = self.apply_expression_encoder(
                code_task_input=code_task_input,
                previous_expression_encodings=encoded_code_expressions,
                expression_encoder=encoder,
                expression_combiner=combiner)

        encoded_symbols = self.symbols_encoder(
            encoded_identifiers=encoded_identifiers,
            symbols=code_task_input.symbols,
            encoded_expressions=encoded_code_expressions.token_seqs
            if self.symbols_encoder.encoder_params.use_symbols_occurrences else None,
            tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
            encoded_ast_nodes=encoded_code_expressions.ast_nodes
            if self.symbols_encoder.encoder_params.use_symbols_occurrences else None,
            ast_nodes_with_symbol_leaf_nodes_indices=code_task_input.ast.ast_nodes_with_symbol_leaf_nodes_indices.indices,
            ast_nodes_with_symbol_leaf_symbol_idx=code_task_input.ast.ast_nodes_with_symbol_leaf_symbol_idx.indices)

        return EncodedMethodCFGV2(
            encoded_identifiers=encoded_identifiers,
            encoded_cfg_nodes=encoded_cfg_nodes,
            encoded_symbols=encoded_symbols)


class MacroContextAdderToSubAST(nn.Module):
    def __init__(self, ast_node_encoding_dim: int, cfg_node_encoding_dim: int,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(MacroContextAdderToSubAST, self).__init__()
        self.ast_node_encoding_dim = ast_node_encoding_dim
        self.cfg_node_encoding_dim = cfg_node_encoding_dim
        self.gate = StateUpdater(
            state_dim=self.ast_node_encoding_dim, update_dim=self.cfg_node_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(
            self, previous_ast_nodes_encodings: torch.Tensor,
            new_cfg_nodes_encodings: torch.Tensor,
            cfg_expressions_sub_ast_input: PDGExpressionsSubASTInputTensors):
        ast_nodes_encodings_updates = self.gate(
            previous_state=previous_ast_nodes_encodings[
                cfg_expressions_sub_ast_input.ast_node_idx_to_pdg_node_idx_mapping_key.indices],
            state_update=new_cfg_nodes_encodings[
                cfg_expressions_sub_ast_input.ast_node_idx_to_pdg_node_idx_mapping_value.indices])
        return previous_ast_nodes_encodings.scatter(
            dim=0,
            index=cfg_expressions_sub_ast_input.ast_node_idx_to_pdg_node_idx_mapping_key.indices
                .unsqueeze(-1).expand(ast_nodes_encodings_updates.shape),
            src=ast_nodes_encodings_updates)


class CFGSinglePathEncoder(nn.Module):
    def __init__(self, cfg_node_dim: int, sequence_type: str,
                 cfg_paths_sequence_encoder_params: SequenceEncoderParams,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGSinglePathEncoder, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        assert sequence_type in {'linear', 'random-permutation'}
        self.sequence_type = sequence_type
        self.sequence_encoder_layer = SequenceEncoder(
            encoder_params=cfg_paths_sequence_encoder_params,
            input_dim=self.cfg_node_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(self, pdg_input: PDGInputTensors,
                cfg_nodes_encodings: torch.Tensor):
        if self.sequence_type == 'linear':
            unflattened_nodes_encodings = pdg_input.cfg_nodes_control_kind.unflatten(cfg_nodes_encodings)
        elif self.sequence_type == 'random-permutation':
            # TODO: fix to use `BatchedFlattenedIndicesPseudoRandomPermutation.permute()`!
            unflattened_nodes_encodings = cfg_nodes_encodings[pdg_input.cfg_nodes_random_permutation.permutations]
            unflattened_nodes_encodings = unflattened_nodes_encodings.masked_fill(
                ~pdg_input.cfg_nodes_control_kind.unflattener_mask.unsqueeze(-1)
                    .expand(unflattened_nodes_encodings.shape), 0)
        else:
            assert False
        path_encodings = self.sequence_encoder_layer(
            sequence_input=unflattened_nodes_encodings,
            lengths=pdg_input.cfg_nodes_control_kind.nr_items_per_example,
            batch_first=True).sequence
        assert path_encodings.shape == unflattened_nodes_encodings.shape

        if self.sequence_type == 'linear':
            reflattened_nodes_encodings = pdg_input.cfg_nodes_control_kind.flatten(path_encodings)
        elif self.sequence_type == 'random-permutation':
            nr_cfg_nodes = cfg_nodes_encodings.size(0)
            # TODO: fix to use `BatchedFlattenedIndicesPseudoRandomPermutation.inverse_permute()`!
            cfg_nodes_permuted_indices = pdg_input.cfg_nodes_random_permutation.permutations
            cfg_nodes_permuted_indices = cfg_nodes_permuted_indices.masked_fill(
                ~pdg_input.cfg_nodes_control_kind.unflattener_mask, nr_cfg_nodes)
            new_cfg_nodes_encodings = \
                cfg_nodes_encodings.new_zeros(size=(nr_cfg_nodes + 1, cfg_nodes_encodings.size(1)))
            path_encodings_flattened = path_encodings.flatten(0, 1)
            cfg_nodes_permuted_indices_flattened = cfg_nodes_permuted_indices.flatten(0, 1)
            new_cfg_nodes_encodings.scatter_(
                dim=0,
                index=cfg_nodes_permuted_indices_flattened.unsqueeze(-1).expand(path_encodings_flattened.shape),
                src=path_encodings_flattened)
            reflattened_nodes_encodings = new_cfg_nodes_encodings[:-1]
        else:
            assert False

        assert reflattened_nodes_encodings.shape == cfg_nodes_encodings.shape
        return reflattened_nodes_encodings


class AddSymbolsEncodingsToExpressions(nn.Module):
    def __init__(self, expression_token_encoding_dim: int, symbol_encoding_dim: int,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(AddSymbolsEncodingsToExpressions, self).__init__()
        self.expression_token_encoding_dim = expression_token_encoding_dim
        self.symbol_encoding_dim = symbol_encoding_dim
        self.gate = StateUpdater(
            state_dim=expression_token_encoding_dim, update_dim=symbol_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, expressions_encodings: torch.Tensor,
                symbols_encodings: torch.Tensor,
                symbols: SymbolsInputTensors) -> torch.Tensor:
        assert expressions_encodings.ndim == 3
        orig_expressions_encodings_shape = expressions_encodings.shape
        max_nr_tokens_per_expression = expressions_encodings.size(1)
        cfg_expr_tokens_indices_of_symbols_occurrences = \
            max_nr_tokens_per_expression * symbols.symbols_appearances_cfg_expression_idx.indices + \
            symbols.symbols_appearances_expression_token_idx.tensor
        expressions_encodings = expressions_encodings.flatten(0, 1)
        symbols_occurrences = expressions_encodings.index_select(
            dim=0,
            index=cfg_expr_tokens_indices_of_symbols_occurrences)
        symbols_encodings_for_occurrences = symbols_encodings[symbols.symbols_appearances_symbol_idx.indices]
        assert symbols_occurrences.ndim == 2 and symbols_encodings_for_occurrences.ndim == 2
        assert symbols_occurrences.size(0) == symbols_encodings_for_occurrences.size(0)

        updated_occurrences = self.gate(
            previous_state=symbols_occurrences, state_update=symbols_encodings_for_occurrences)
        assert updated_occurrences.shape == symbols_occurrences.shape

        updated_expressions_encodings = expressions_encodings.scatter(
            dim=0,
            index=cfg_expr_tokens_indices_of_symbols_occurrences.unsqueeze(-1).expand(updated_occurrences.shape),
            src=updated_occurrences)
        return updated_expressions_encodings.view(orig_expressions_encodings_shape)


class CFGNodeEncodingMixerWithExpressionEncoding(nn.Module):
    def __init__(self, cfg_node_dim: int, cfg_combined_expression_dim: int,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGNodeEncodingMixerWithExpressionEncoding, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.cfg_combined_expression_dim = cfg_combined_expression_dim
        # self.projection_layer = nn.Linear(
        #     in_features=self.cfg_node_dim + self.cfg_combined_expression_dim, out_features=self.cfg_node_dim)
        self.gate = StateUpdater(
            state_dim=self.cfg_node_dim, update_dim=self.cfg_combined_expression_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.activation_layer = get_activation_layer(activation_fn)()

    def forward(self, previous_cfg_nodes_encodings: torch.Tensor,
                cfg_combined_expressions_encodings: torch.Tensor,
                cfg_nodes_has_expression_mask: torch.BoolTensor):
        # use gating-mechanism here (it's a classic state-update use-case here)
        previous_encodings_of_cfg_nodes_with_expressions = previous_cfg_nodes_encodings[cfg_nodes_has_expression_mask]
        new_cfg_node_encodings_for_nodes_with_expressions = self.gate(
            previous_state=previous_encodings_of_cfg_nodes_with_expressions,
            state_update=cfg_combined_expressions_encodings)

        # TODO: consider adding another layer
        # new_cfg_node_encodings_for_nodes_with_expressions = self.projection_layer(torch.cat([
        #     previous_cfg_nodes_encodings[cfg_nodes_has_expression_mask], cfg_combined_expressions_encodings], dim=-1))
        # new_cfg_node_encodings_for_nodes_with_expressions = self.dropout_layer(self.activation_layer(
        #     new_cfg_node_encodings_for_nodes_with_expressions))

        return previous_cfg_nodes_encodings.masked_scatter(
            cfg_nodes_has_expression_mask.unsqueeze(-1).expand(previous_cfg_nodes_encodings.size()),
            new_cfg_node_encodings_for_nodes_with_expressions)


class CFGPathsNGramsEncoder(nn.Module):
    def __init__(self, cfg_node_dim: int,
                 cfg_paths_sequence_encoder_params: SequenceEncoderParams,
                 control_flow_edge_types_vocab: Optional[Vocabulary] = None, is_first: bool = True,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGPathsNGramsEncoder, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.control_flow_edge_types_vocab = control_flow_edge_types_vocab
        self.is_first = is_first
        if self.is_first:
            self.control_flow_edge_types_embeddings = nn.Embedding(
                num_embeddings=len(self.control_flow_edge_types_vocab),
                embedding_dim=self.cfg_node_dim,
                padding_idx=self.control_flow_edge_types_vocab.get_word_idx('<PAD>'))
        self.sequence_encoder_layer = SequenceEncoder(
            encoder_params=cfg_paths_sequence_encoder_params,
            input_dim=self.cfg_node_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.nodes_occurrences_encodings_gate = StateUpdater(
            state_dim=self.cfg_node_dim, update_dim=self.cfg_node_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(
            self, cfg_nodes_encodings: torch.Tensor,
            cfg_control_flow_paths_ngrams_input: Mapping[int, CFGPathsNGramsInputTensors],
            previous_encoding_layer_output: Optional[Dict[int, EncodedCFGPaths]] = None) \
            -> Dict[int, EncodedCFGPaths]:
        assert (previous_encoding_layer_output is not None) ^ self.is_first
        results = {}
        for ngrams_n, ngrams in cfg_control_flow_paths_ngrams_input.items():
            if self.is_first:
                ngrams_edges_types_embeddings = self.control_flow_edge_types_embeddings(ngrams.edges_types.sequences)
                ngrams_nodes_encodings = cfg_nodes_encodings[ngrams.nodes_indices.sequences]
            else:
                ngrams_edges_types_embeddings = previous_encoding_layer_output[ngrams_n].edges_occurrences

                # Update encodings of cfg nodes occurrences in paths:
                updated_ngrams_nodes_encodings = cfg_nodes_encodings[ngrams.nodes_indices.sequences]
                assert updated_ngrams_nodes_encodings.shape == \
                       previous_encoding_layer_output[ngrams_n].nodes_occurrences.shape
                ngrams_nodes_encodings = self.nodes_occurrences_encodings_gate(
                    previous_state=previous_encoding_layer_output[ngrams_n].nodes_occurrences,
                    state_update=updated_ngrams_nodes_encodings)

            # weave nodes & edge-types in each path
            interwoven_nodes_and_edge_types_embeddings = weave_tensors(
                tensors=[ngrams_nodes_encodings, ngrams_edges_types_embeddings], dim=1)
            assert interwoven_nodes_and_edge_types_embeddings.shape == \
                   (ngrams_nodes_encodings.size(0),
                    2 * ngrams_nodes_encodings.size(1),
                    self.cfg_node_dim)

            ngrams_encodings = self.sequence_encoder_layer(
                sequence_input=interwoven_nodes_and_edge_types_embeddings,
                batch_first=True).sequence
            assert ngrams_encodings.shape == interwoven_nodes_and_edge_types_embeddings.shape

            # separate nodes encodings and edge types embeddings from paths
            nodes_occurrences_encodings, edges_occurrences_encodings = unweave_tensor(
                woven_tensor=ngrams_encodings, dim=1, nr_target_tensors=2)
            assert nodes_occurrences_encodings.shape == ngrams_nodes_encodings.shape
            results[ngrams_n] = EncodedCFGPaths(
                nodes_occurrences=nodes_occurrences_encodings,
                edges_occurrences=edges_occurrences_encodings)

        return results


class ScatterCFGEncodedNGramsToCFGNodeEncodings(nn.Module):
    def __init__(self, cfg_node_encoding_dim: int,
                 cfg_nodes_folding_params: ScatterCombinerParams,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(ScatterCFGEncodedNGramsToCFGNodeEncodings, self).__init__()
        self.cfg_nodes_folding_params = cfg_nodes_folding_params
        self.cfg_node_encoding_dim = cfg_node_encoding_dim
        self.scatter_combiner_layer = ScatterCombiner(
            encoding_dim=cfg_node_encoding_dim, combiner_params=cfg_nodes_folding_params)
        self.gate = StateUpdater(
            state_dim=cfg_node_encoding_dim, update_dim=cfg_node_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(self, encoded_cfg_paths_ngrams: Dict[int, EncodedCFGPaths],
                cfg_control_flow_paths_ngrams_input: Mapping[int, CFGPathsNGramsInputTensors],
                previous_cfg_nodes_encodings: torch.Tensor, nr_cfg_nodes: int):
        # `encoded_cfg_paths` is in form of sequences. We flatten it by applying a mask selector.
        # The mask also helps to ignore paddings.
        sorted_ngrams_n = sorted(list(
            set(encoded_cfg_paths_ngrams.keys()) & set(cfg_control_flow_paths_ngrams_input.keys())))
        assert all(
            encoded_cfg_paths_ngrams[ngrams_n].nodes_occurrences.shape[:-1] ==
            cfg_control_flow_paths_ngrams_input[ngrams_n].nodes_indices.sequences.shape
            for ngrams_n in sorted_ngrams_n)
        flattened_nodes_occurrences = torch.cat(
            [encoded_cfg_paths_ngrams[ngrams_n].nodes_occurrences.flatten(0, 1)
             for ngrams_n in sorted_ngrams_n], dim=0)
        flattened_nodes_indices = torch.cat(
            [cfg_control_flow_paths_ngrams_input[ngrams_n].nodes_indices.sequences.flatten(0, 1)
             for ngrams_n in sorted_ngrams_n], dim=0)
        if self.cfg_nodes_folding_params.method == 'attn':
            assert previous_cfg_nodes_encodings is not None
            assert previous_cfg_nodes_encodings.size(0) == nr_cfg_nodes
        updated_cfg_nodes_encodings = self.scatter_combiner_layer(
            scattered_input=flattened_nodes_occurrences,
            indices=flattened_nodes_indices,
            dim_size=nr_cfg_nodes,
            attn_queries=previous_cfg_nodes_encodings)
        assert updated_cfg_nodes_encodings.size() == (nr_cfg_nodes, self.cfg_node_encoding_dim)

        # Note: This gate here is the last we added so far (in the paths case).
        new_cfg_nodes_encodings = self.gate(
            previous_state=previous_cfg_nodes_encodings,
            state_update=updated_cfg_nodes_encodings)
        return new_cfg_nodes_encodings


class ScatterCFGEncodedPathsToCFGNodeEncodings(nn.Module):
    def __init__(self, cfg_node_encoding_dim: int,
                 cfg_nodes_folding_params: ScatterCombinerParams,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(ScatterCFGEncodedPathsToCFGNodeEncodings, self).__init__()
        self.cfg_nodes_folding_params = cfg_nodes_folding_params
        self.scatter_combiner_layer = ScatterCombiner(
            encoding_dim=cfg_node_encoding_dim, combiner_params=cfg_nodes_folding_params)
        self.gate = StateUpdater(
            state_dim=cfg_node_encoding_dim, update_dim=cfg_node_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(self, encoded_cfg_node_occurrences_in_paths: torch.Tensor,
                cfg_paths_mask: torch.BoolTensor,
                cfg_paths_node_indices: torch.LongTensor,
                previous_cfg_nodes_encodings: torch.Tensor,
                nr_cfg_nodes: int):
        # `encoded_cfg_paths` is in form of sequences. We flatten it by applying a mask selector.
        # The mask also helps to ignore paddings.
        if self.cfg_nodes_folding_params.method == 'attn':
            assert previous_cfg_nodes_encodings is not None
            assert previous_cfg_nodes_encodings.size(0) == nr_cfg_nodes
        updated_cfg_nodes_encodings = self.scatter_combiner_layer(
            scattered_input=encoded_cfg_node_occurrences_in_paths[cfg_paths_mask],
            indices=cfg_paths_node_indices[cfg_paths_mask],
            dim_size=nr_cfg_nodes,
            attn_queries=previous_cfg_nodes_encodings)
        assert updated_cfg_nodes_encodings.size() == (nr_cfg_nodes, encoded_cfg_node_occurrences_in_paths.size(2))

        # Note: This gate here is the last we added so far.
        #       It actually reduced the results (F1 went down from 0.53 to 0.52).
        #       But maybe choosing another combination of places (to apply the gating-mechanism)
        #         which includes this place might yield better results.
        #       It made the first evaluation result (before training, initial random weights)
        #         much higher to around F1=0.26.
        new_cfg_nodes_encodings = self.gate(
            previous_state=previous_cfg_nodes_encodings,
            state_update=updated_cfg_nodes_encodings)
        return new_cfg_nodes_encodings
