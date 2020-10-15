import itertools
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from typing import NamedTuple

from ndfa.nn_utils.misc import get_activation_layer
from ndfa.ndfa_model_hyper_parameters import MethodCFGEncoderParams
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors, PDGInputTensors
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.expression_encoder import ExpressionEncoder
from ndfa.nn_utils.sequence_combiner import SequenceCombiner
from ndfa.code_nn_modules.cfg_node_encoder import CFGNodeEncoder
from ndfa.code_nn_modules.cfg_paths_encoder import CFGPathEncoder, EncodedCFGPaths
from ndfa.code_nn_modules.symbols_encoder import SymbolsEncoder
from ndfa.nn_utils.sequence_encoder import SequenceEncoder
from ndfa.nn_utils.seq_context_adder import SeqContextAdder
from ndfa.nn_utils.scatter_combiner import ScatterCombiner
from ndfa.ndfa_model_hyper_parameters import SequenceEncoderParams
from ndfa.code_nn_modules.code_task_input import SymbolsInputTensors
from ndfa.nn_utils.weave_tensors import weave_tensors, unweave_tensor
from ndfa.nn_utils.gate import Gate


__all__ = ['MethodCFGEncoder', 'EncodedMethodCFG']


class EncodedMethodCFG(NamedTuple):
    encoded_identifiers: torch.Tensor
    encoded_cfg_nodes: torch.Tensor
    encoded_symbols: torch.Tensor


class MethodCFGEncoder(nn.Module):
    def __init__(self, code_task_vocabs: CodeTaskVocabs, identifier_embedding_dim: int,
                 symbol_embedding_dim: int, encoder_params: MethodCFGEncoderParams,
                 use_symbols_occurrences_for_symbols_encodings: bool,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu',
                 nr_layers: int = 2, use_skip_connections: bool = False,
                 use_norm: bool = True):
        super(MethodCFGEncoder, self).__init__()
        assert nr_layers >= 1
        self.identifier_embedding_dim = identifier_embedding_dim
        self.symbol_embedding_dim = symbol_embedding_dim
        self.encoder_params = encoder_params
        self.first_expression_encoder = ExpressionEncoder(
            kos_tokens_vocab=code_task_vocabs.kos_tokens,
            tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
            expressions_special_words_vocab=code_task_vocabs.expressions_special_words,
            identifiers_special_words_vocab=code_task_vocabs.identifiers_special_words,
            encoder_params=self.encoder_params.cfg_node_expression_encoder,
            identifier_embedding_dim=self.identifier_embedding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.expression_updaters = nn.ModuleList([
            SequenceEncoder(
                encoder_params=self.encoder_params.cfg_node_expression_encoder.sequence_encoder,
                input_dim=self.encoder_params.cfg_node_expression_encoder.token_encoding_dim,
                hidden_dim=self.encoder_params.cfg_node_expression_encoder.token_encoding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for _ in range(nr_layers - 1)])
        self.expression_combiner = SequenceCombiner(
            encoding_dim=self.encoder_params.cfg_node_expression_encoder.token_encoding_dim,
            combined_dim=self.encoder_params.cfg_node_expression_encoder.combined_expression_encoding_dim,
            combiner_params=self.encoder_params.cfg_node_expression_combiner,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.first_cfg_node_encoder = CFGNodeEncoder(
            cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
            cfg_combined_expression_dim=self.encoder_params.cfg_node_expression_encoder.combined_expression_encoding_dim,
            pdg_node_control_kinds_vocab=code_task_vocabs.pdg_node_control_kinds,
            pdg_node_control_kinds_embedding_dim=self.encoder_params.cfg_node_control_kinds_embedding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.cfg_node_updaters = nn.ModuleList([
            CFGNodeEncoderExpressionUpdateLayer(
                cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
                cfg_combined_expression_dim=self.encoder_params.cfg_node_expression_encoder.combined_expression_encoding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for _ in range(nr_layers - 1)])
        if self.encoder_params.encoder_type in {'control-flow-paths-folded-to-nodes', 'set-of-control-flow-paths'}:
            self.first_cfg_paths_encoder = CFGPathEncoder(
                cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
                cfg_paths_sequence_encoder_params=self.encoder_params.cfg_paths_sequence_encoder,
                control_flow_edge_types_vocab=code_task_vocabs.pdg_control_flow_edge_types,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            self.cfg_path_updaters = nn.ModuleList([
                CFGPathsUpdater(
                    cfg_node_dim=self.encoder_params.cfg_node_encoding_dim,
                    cfg_paths_sequence_encoder_params=self.encoder_params.cfg_paths_sequence_encoder,
                    dropout_rate=dropout_rate, activation_fn=activation_fn)
                for _ in range(nr_layers - 1)])
        if self.encoder_params.encoder_type == 'control-flow-paths-folded-to-nodes':
            self.scatter_cfg_encoded_paths_to_cfg_node_encodings = ScatterCFGEncodedPathsToCFGNodeEncodings(
                cfg_node_encoding_dim=self.encoder_params.cfg_node_encoding_dim,
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
        self.expression_context_adder = SeqContextAdder(
            main_dim=self.encoder_params.cfg_node_expression_encoder.token_encoding_dim,
            ctx_dim=self.encoder_params.cfg_node_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.symbols_encoder = SymbolsEncoder(
            symbols_special_words_vocab=code_task_vocabs.symbols_special_words,
            symbol_embedding_dim=self.symbol_embedding_dim,
            expression_encoding_dim=self.encoder_params.cfg_node_expression_encoder.token_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.add_symbols_encodings_to_expressions = AddSymbolsEncodingsToExpressions(
            expression_token_encoding_dim=self.encoder_params.cfg_node_expression_encoder.token_encoding_dim,
            symbol_encoding_dim=self.symbol_embedding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.dropout_layer = nn.Dropout(dropout_rate)

        self.use_norm = use_norm
        if self.use_norm:
            self.expressions_layer_norm = nn.ModuleList([
                LayerNorm(self.encoder_params.cfg_node_expression_encoder.token_encoding_dim)
                for _ in range(nr_layers)])
            self.combined_expressions_layer_norm = nn.ModuleList([
                LayerNorm(self.encoder_params.cfg_node_expression_encoder.combined_expression_encoding_dim)
                for _ in range(nr_layers)])
            self.cfg_nodes_layer_norm = nn.ModuleList([
                LayerNorm(self.encoder_params.cfg_node_encoding_dim)
                for _ in range(nr_layers)])

        self.use_symbols_occurrences_for_symbols_encodings = \
            use_symbols_occurrences_for_symbols_encodings
        self.use_skip_connections = use_skip_connections  # TODO: move to HPs
        # self.update_cfg_nodes_within_encoding_gate = Gate(
        #     state_dim=self.encoder_params.cfg_node_encoding_dim,
        #     update_dim=self.encoder_params.cfg_node_encoding_dim,
        #     dropout_rate=dropout_rate, activation_fn=activation_fn)
        # self.update_cfg_nodes_cross_encoding_gate = Gate(
        #     state_dim=self.encoder_params.cfg_node_encoding_dim,
        #     update_dim=self.encoder_params.cfg_node_encoding_dim,
        #     dropout_rate=dropout_rate, activation_fn=activation_fn)
        # self.update_cfg_nodes_encoding_gates = nn.ModuleList([
        #     Gate(state_dim=self.encoder_params.cfg_node_encoding_dim,
        #          update_dim=self.encoder_params.cfg_node_encoding_dim,
        #          dropout_rate=dropout_rate, activation_fn=activation_fn)
        #     for _ in range(nr_layers - 1)])
        # self.update_cfg_nodes_encoding_from_cf_paths_gates = nn.ModuleList([
        #     Gate(state_dim=self.encoder_params.cfg_node_encoding_dim,
        #          update_dim=self.encoder_params.cfg_node_encoding_dim,
        #          dropout_rate=dropout_rate, activation_fn=activation_fn)
        #     for _ in range(nr_layers)])

    def forward(self, code_task_input: MethodCodeInputTensors, encoded_identifiers: torch.Tensor) -> EncodedMethodCFG:
        encoded_expressions_with_context = None
        for layer_idx, (expression_updater, cfg_node_updater, cfg_path_updater) \
                in enumerate(zip(itertools.chain((None,), self.expression_updaters),
                       itertools.chain((None,), self.cfg_node_updaters),
                       itertools.chain((None,), self.cfg_path_updaters))):
            assert not (encoded_expressions_with_context is None) ^ (expression_updater is None)
            if expression_updater is None:
                encoded_expressions = self.first_expression_encoder(
                    expressions=code_task_input.pdg.cfg_nodes_tokenized_expressions,
                    encoded_identifiers=encoded_identifiers)
            else:
                assert encoded_symbols is not None
                encoded_expressions_with_context = self.add_symbols_encodings_to_expressions(
                    expressions_encodings=encoded_expressions_with_context,
                    symbols_encodings=encoded_symbols,
                    symbols=code_task_input.symbols)
                if self.use_norm:
                    encoded_expressions_with_context = self.expressions_layer_norm[layer_idx](encoded_expressions_with_context)

                new_encoded_expressions = expression_updater(
                    sequence_input=encoded_expressions_with_context,
                    lengths=code_task_input.pdg.cfg_nodes_tokenized_expressions.token_type.sequences_lengths,
                    batch_first=True).sequence
                if self.use_skip_connections:
                    # TODO: use AddNorm for skip-connections here
                    encoded_expressions = encoded_expressions + new_encoded_expressions  # skip-connection
                else:
                    encoded_expressions = new_encoded_expressions
            if self.use_norm:
                encoded_expressions = self.expressions_layer_norm[layer_idx](encoded_expressions)
            combined_expressions = self.expression_combiner(
                sequence_encodings=encoded_expressions,
                sequence_lengths=code_task_input.pdg.cfg_nodes_tokenized_expressions.token_type.sequences_lengths)
            if self.use_norm:
                combined_expressions = self.combined_expressions_layer_norm[layer_idx](combined_expressions)
            if cfg_node_updater is None:
                encoded_cfg_nodes = self.first_cfg_node_encoder(
                    combined_cfg_expressions_encodings=combined_expressions, pdg=code_task_input.pdg)
            else:
                assert encoded_cfg_nodes is not None
                new_encoded_cfg_nodes = cfg_node_updater(
                    previous_cfg_nodes_encodings=encoded_cfg_nodes,
                    cfg_combined_expressions_encodings=combined_expressions,
                    cfg_nodes_has_expression_mask=code_task_input.pdg.cfg_nodes_has_expression_mask.tensor)
                if self.use_skip_connections:
                    # TODO: use AddNorm for skip-connections here
                    encoded_cfg_nodes = encoded_cfg_nodes + new_encoded_cfg_nodes  # skip-connection
                else:
                    # encoded_cfg_nodes = self.update_cfg_nodes_encoding_gate(
                    #     previous_state=encoded_cfg_nodes, state_update=new_encoded_cfg_nodes)
                    # encoded_cfg_nodes = self.update_cfg_nodes_encoding_gates[layer_idx - 1](
                    #     previous_state=encoded_cfg_nodes, state_update=new_encoded_cfg_nodes)
                    encoded_cfg_nodes = new_encoded_cfg_nodes
            if self.use_norm:
                encoded_cfg_nodes = self.cfg_nodes_layer_norm[layer_idx](encoded_cfg_nodes)

            if self.encoder_params.encoder_type in {'control-flow-paths-folded-to-nodes', 'set-of-control-flow-paths'}:
                if cfg_path_updater is None:
                    encoded_cfg_paths = self.first_cfg_paths_encoder(
                        cfg_nodes_encodings=encoded_cfg_nodes,
                        cfg_paths_nodes_indices=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences,
                        cfg_paths_edge_types=code_task_input.pdg.cfg_control_flow_paths.edges_types.sequences,
                        cfg_paths_lengths=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences_lengths)
                else:
                    encoded_cfg_paths = cfg_path_updater(
                        previous_cfg_paths_encodings=encoded_cfg_paths,
                        updated_cfg_nodes_encodings=encoded_cfg_nodes,
                        cfg_paths_nodes_indices=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences,
                        cfg_paths_mask=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences_mask,
                        cfg_paths_lengths=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences_lengths)
            if self.encoder_params.encoder_type == 'control-flow-paths-folded-to-nodes':
                new_encoded_cfg_nodes = self.scatter_cfg_encoded_paths_to_cfg_node_encodings(
                    encoded_cfg_node_occurrences_in_paths=encoded_cfg_paths.nodes_occurrences,
                    cfg_paths_mask=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences_mask,
                    cfg_paths_node_indices=code_task_input.pdg.cfg_control_flow_paths.nodes_indices.sequences,
                    previous_cfg_nodes_encodings=encoded_cfg_nodes,
                    nr_cfg_nodes=code_task_input.pdg.cfg_nodes_has_expression_mask.batch_size)
                if self.use_skip_connections:
                    # TODO: use AddNorm for skip-connections here
                    encoded_cfg_nodes = encoded_cfg_nodes + new_encoded_cfg_nodes  # skip-connection
                else:
                    # encoded_cfg_nodes = self.update_cfg_nodes_encoding_gate(
                    #     previous_state=encoded_cfg_nodes, state_update=new_encoded_cfg_nodes)
                    # encoded_cfg_nodes = self.update_cfg_nodes_encoding_from_cf_paths_gates[layer_idx](
                    #     previous_state=encoded_cfg_nodes, state_update=new_encoded_cfg_nodes)
                    encoded_cfg_nodes = new_encoded_cfg_nodes
                if self.use_norm:
                    encoded_cfg_nodes = self.cfg_nodes_layer_norm[layer_idx](encoded_cfg_nodes)
            elif self.encoder_params.encoder_type in {'all-nodes-single-unstructured-linear-seq',
                                                      'all-nodes-single-random-permutation-seq'}:
                new_encoded_cfg_nodes = self.cfg_single_path_encoder(
                    pdg_input=code_task_input.pdg,
                    cfg_nodes_encodings=encoded_cfg_nodes)
                # encoded_cfg_nodes = self.update_cfg_nodes_encoding_gate(
                #     previous_state=encoded_cfg_nodes, state_update=new_encoded_cfg_nodes)
                # encoded_cfg_nodes = self.update_cfg_nodes_encoding_from_cf_paths_gates[layer_idx](
                #     previous_state=encoded_cfg_nodes, state_update=new_encoded_cfg_nodes)
                encoded_cfg_nodes = new_encoded_cfg_nodes
                if self.use_norm:
                    encoded_cfg_nodes = self.cfg_nodes_layer_norm[layer_idx](encoded_cfg_nodes)
            elif self.encoder_params.encoder_type == 'set-of-control-flow-paths':
                raise NotImplementedError  # TODO: impl
            elif self.encoder_params.encoder_type == 'gnn':
                raise NotImplementedError  # TODO: impl
            elif self.encoder_params.encoder_type == 'control-flow-paths-ngrams':
                raise NotImplementedError  # TODO: impl
            elif self.encoder_params.encoder_type == 'set-of-nodes':
                pass  # We actually do not need to do anything in this case.
            else:
                raise ValueError(f'Unsupported method-CFG encoding type `{self.encoder_params.encoder_type}`.')

            # TODO: maybe we do not need this for `self.encoder_params.encoder_type == 'set-of-nodes'`
            # FIXME: notice there is a skip-connection built-in here that is not conditioned
            #  by the flag `self.use_skip_connections`.
            encoded_expressions_with_context = self.expression_context_adder(
                sequence=encoded_expressions,
                sequence_mask=code_task_input.pdg.cfg_nodes_tokenized_expressions.token_type.sequences_mask,
                context=encoded_cfg_nodes[code_task_input.pdg.cfg_nodes_has_expression_mask.tensor])
            if self.use_norm:
                encoded_expressions_with_context = self.expressions_layer_norm[layer_idx](encoded_expressions_with_context)

            encoded_symbols = self.symbols_encoder(
                encoded_identifiers=encoded_identifiers,
                symbols=code_task_input.symbols,
                encoded_cfg_expressions=encoded_expressions_with_context
                if self.use_symbols_occurrences_for_symbols_encodings else None)

        return EncodedMethodCFG(
            encoded_identifiers=encoded_identifiers,
            encoded_cfg_nodes=encoded_cfg_nodes,
            encoded_symbols=encoded_symbols)


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
            # pdg_input.cfg_nodes_random_permutation.unflatten(pdg_input.cfg_nodes_random_permutation.sequences)
            raise NotImplementedError  # TODO: implement!
        else:
            assert False
        path_encodings = self.sequence_encoder_layer(
            sequence_input=unflattened_nodes_encodings,
            lengths=pdg_input.cfg_nodes_control_kind.nr_items_per_example,
            batch_first=True).sequence
        assert path_encodings.shape == unflattened_nodes_encodings.shape
        reflattened_nodes_encodings = pdg_input.cfg_nodes_control_kind.flatten(unflattened_nodes_encodings)
        assert reflattened_nodes_encodings.shape == cfg_nodes_encodings.shape
        return reflattened_nodes_encodings


class AddSymbolsEncodingsToExpressions(nn.Module):
    def __init__(self, expression_token_encoding_dim: int, symbol_encoding_dim: int,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(AddSymbolsEncodingsToExpressions, self).__init__()
        self.expression_token_encoding_dim = expression_token_encoding_dim
        self.symbol_encoding_dim = symbol_encoding_dim
        self.gate = Gate(state_dim=expression_token_encoding_dim, update_dim=symbol_encoding_dim,
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


class CFGNodeEncoderExpressionUpdateLayer(nn.Module):
    def __init__(self, cfg_node_dim: int, cfg_combined_expression_dim: int,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGNodeEncoderExpressionUpdateLayer, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.cfg_combined_expression_dim = cfg_combined_expression_dim
        # self.projection_layer = nn.Linear(
        #     in_features=self.cfg_node_dim + self.cfg_combined_expression_dim, out_features=self.cfg_node_dim)
        self.gate = Gate(state_dim=self.cfg_node_dim, update_dim=self.cfg_combined_expression_dim,
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


class CFGPathsUpdater(nn.Module):
    def __init__(self, cfg_node_dim: int,
                 cfg_paths_sequence_encoder_params: SequenceEncoderParams,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGPathsUpdater, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.sequence_encoder_layer = SequenceEncoder(
            encoder_params=cfg_paths_sequence_encoder_params,
            input_dim=self.cfg_node_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

        self.nodes_occurrences_encodings_gate = Gate(
            state_dim=self.cfg_node_dim, update_dim=self.cfg_node_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(self,
                previous_cfg_paths_encodings: EncodedCFGPaths,
                updated_cfg_nodes_encodings: torch.Tensor,
                cfg_paths_nodes_indices: torch.LongTensor,
                cfg_paths_mask: torch.BoolTensor,
                cfg_paths_lengths: torch.LongTensor) -> EncodedCFGPaths:
        # Update encodings of cfg nodes occurrences in paths:
        updated_cfg_nodes_encodings_per_node_occurrence_in_path = updated_cfg_nodes_encodings[cfg_paths_nodes_indices]
        assert updated_cfg_nodes_encodings_per_node_occurrence_in_path.shape == \
               previous_cfg_paths_encodings.nodes_occurrences.shape

        updated_cfg_paths_nodes_encodings = self.nodes_occurrences_encodings_gate(
            previous_state=previous_cfg_paths_encodings.nodes_occurrences,
            state_update=updated_cfg_nodes_encodings_per_node_occurrence_in_path)
        updated_cfg_paths_nodes_encodings = updated_cfg_paths_nodes_encodings.masked_fill(
            ~cfg_paths_mask.unsqueeze(-1).expand(updated_cfg_paths_nodes_encodings.size()), 0)

        # OLD simple way to update encodings of cfg nodes occurrences in paths:
        # cfg_paths_nodes_embeddings = \
        #     previous_cfg_paths_encodings.nodes_occurrences + updated_cfg_nodes_encodings[cfg_paths_nodes_indices]

        cfg_paths_edge_types_embeddings = previous_cfg_paths_encodings.edges_occurrences

        # weave nodes & edge-types in each path
        cfg_paths_interwoven_nodes_and_edge_types_embeddings = weave_tensors(
            tensors=[updated_cfg_paths_nodes_encodings, cfg_paths_edge_types_embeddings], dim=1)
        assert cfg_paths_interwoven_nodes_and_edge_types_embeddings.shape == \
               (updated_cfg_paths_nodes_encodings.size(0),
                2 * updated_cfg_paths_nodes_encodings.size(1),
                self.cfg_node_dim)

        paths_encodings = self.sequence_encoder_layer(
            sequence_input=cfg_paths_interwoven_nodes_and_edge_types_embeddings,
            lengths=cfg_paths_lengths * 2, batch_first=True).sequence
        assert paths_encodings.shape == cfg_paths_interwoven_nodes_and_edge_types_embeddings.shape

        # separate nodes encodings and edge types embeddings from paths
        nodes_occurrences_encodings, edges_occurrences_encodings = unweave_tensor(
            woven_tensor=paths_encodings, dim=1, nr_target_tensors=2)
        assert nodes_occurrences_encodings.shape == edges_occurrences_encodings.shape == \
               updated_cfg_paths_nodes_encodings.shape

        return EncodedCFGPaths(
            nodes_occurrences=nodes_occurrences_encodings,
            edges_occurrences=edges_occurrences_encodings)


class ScatterCFGEncodedPathsToCFGNodeEncodings(nn.Module):
    def __init__(self, cfg_node_encoding_dim: int, combining_method: str = 'attn',
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(ScatterCFGEncodedPathsToCFGNodeEncodings, self).__init__()
        self.combining_method = combining_method
        self.scatter_combiner_layer = ScatterCombiner(
            encoding_dim=cfg_node_encoding_dim, combining_method=combining_method)
        self.gate = Gate(
            state_dim=cfg_node_encoding_dim, update_dim=cfg_node_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(self, encoded_cfg_node_occurrences_in_paths: torch.tensor,
                cfg_paths_mask: torch.BoolTensor,
                cfg_paths_node_indices: torch.LongTensor,
                previous_cfg_nodes_encodings: torch.Tensor,
                nr_cfg_nodes: int):
        # `encoded_cfg_paths` is in form of sequences. We flatten it by applying a mask selector.
        # The mask also helps to ignore paddings.
        if self.combining_method == 'attn':
            assert previous_cfg_nodes_encodings is not None
            assert previous_cfg_nodes_encodings.size(0) == nr_cfg_nodes
        updated_cfg_nodes_encodings = self.scatter_combiner_layer(
            scattered_input=encoded_cfg_node_occurrences_in_paths[cfg_paths_mask],
            indices=cfg_paths_node_indices[cfg_paths_mask],
            dim_size=nr_cfg_nodes,
            attn_keys=previous_cfg_nodes_encodings)
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
