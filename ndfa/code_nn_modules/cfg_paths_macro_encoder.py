import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Mapping, Callable

from ndfa.code_nn_modules.params.cfg_paths_macro_encoder_params import CFGPathsMacroEncoderParams
from ndfa.nn_utils.modules.graph_paths_encoder import PathsEncoder, EncodedPaths
from ndfa.nn_utils.model_wrapper.vocabulary import Vocabulary
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.code_nn_modules.code_task_input import PDGInputTensors, CFGPathsNGramsInputTensors
from ndfa.nn_utils.modules.params.state_updater_params import StateUpdaterParams
from ndfa.nn_utils.model_wrapper.flattened_tensor import FlattenedTensor


__all__ = ['CFGPathsMacroEncoder', 'CFGPathsMacroEncodings']


@dataclass
class CFGPathsMacroEncodings:
    ngrams: Optional[Dict[int, EncodedPaths]] = None
    paths: Optional[EncodedPaths] = None
    folded_nodes_encodings: Optional[torch.Tensor] = None
    combined_paths: Optional[FlattenedTensor] = None


class CFGPathsMacroEncoder(nn.Module):
    def __init__(
            self,
            params: CFGPathsMacroEncoderParams,
            cfg_node_dim: int,
            cfg_nodes_encodings_state_updater: Optional[StateUpdaterParams] = None,
            control_flow_edge_types_vocab: Optional[Vocabulary] = None,
            is_first_layer: bool = True,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CFGPathsMacroEncoder, self).__init__()
        self.params = params
        self.cfg_node_dim = cfg_node_dim
        self.is_first_layer = is_first_layer
        self.paths_encoder = PathsEncoder(
            node_dim=self.cfg_node_dim,
            paths_sequence_encoder_params=self.params.path_sequence_encoder,
            edge_types_vocab=control_flow_edge_types_vocab,
            edge_type_insertion_mode=self.params.edge_types_insertion_mode,
            is_first_layer=self.is_first_layer,
            fold_occurrences_back_to_nodes=self.params.output_type == CFGPathsMacroEncoderParams.OutputType.FoldNodeOccurrencesToNodeEncodings,
            folding_params=self.params.nodes_folding_params,
            folded_node_encodings_updater_params=cfg_nodes_encodings_state_updater,
            combine_paths=self.params.output_type == CFGPathsMacroEncoderParams.OutputType.SetOfPaths,
            paths_combining_params=self.params.paths_combining_params,
            norm_params=norm_params,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        if self.params.is_ngrams:
            raise NotImplementedError  # TODO: implement this case! the folding / set-of-paths are not implemented when ngrams are used..

    def forward(
            self,
            cfg_nodes_encodings: torch.Tensor,
            pdg_input: PDGInputTensors,
            previous_encoding_layer_output: CFGPathsMacroEncodings = None
    ) -> CFGPathsMacroEncodings:
        if self.params.is_ngrams:
            raise NotImplementedError  # TODO: implement this case!
            cfg_control_flow_paths_ngrams_input = self.get_cfg_control_flow_paths_ngrams_input(pdg_input=pdg_input)
            result_per_ngram_len: Dict[int, EncodedPaths] = {}
            for ngrams_n, ngrams in cfg_control_flow_paths_ngrams_input.items():
                result_per_ngram_len[ngrams_n] = self.paths_encoder(
                    nodes_encodings=cfg_nodes_encodings,
                    paths_nodes_indices=ngrams.nodes_indices.sequences,
                    paths_edge_types=ngrams.edges_types.sequences,
                    paths_lengths=ngrams.nodes_indices.sequences_lengths,
                    paths_mask=ngrams.nodes_indices.sequences_mask,
                    previous_encoding_layer_output=None if previous_encoding_layer_output is None else
                    previous_encoding_layer_output.ngrams[ngrams_n]
                )
            combined_paths = torch.cat([paths.combined_paths for paths in result_per_ngram_len.values()], dim=0)
            raise NotImplementedError
            return CFGPathsMacroEncodings(
                ngrams=result_per_ngram_len,
                folded_nodes_encodings=None,  # TODO!
                combined_paths=FlattenedTensor(
                    flattened=combined_paths,
                    unflattener_mask=None,  # TODO!
                    unflattener_fn=None))  # TODO!
        else:
            cfg_paths_input = pdg_input.cfg_pdg_paths \
                if self.params.paths_type == CFGPathsMacroEncoderParams.PathsType.DataDependencyAndControlFlow else \
                pdg_input.cfg_control_flow_paths
            encoded_paths: EncodedPaths = self.paths_encoder(
                nodes_encodings=cfg_nodes_encodings,
                paths_nodes_indices=cfg_paths_input.nodes_indices.sequences,
                paths_edge_types=cfg_paths_input.edges_types.sequences,
                paths_lengths=cfg_paths_input.nodes_indices.sequences_lengths,
                paths_mask=cfg_paths_input.nodes_indices.sequences_mask,
                previous_encoding_layer_output=None if previous_encoding_layer_output is None else
                previous_encoding_layer_output.paths)

            return CFGPathsMacroEncodings(
                paths=EncodedPaths(
                    nodes_occurrences=encoded_paths.nodes_occurrences,
                    edges_occurrences=encoded_paths.edges_occurrences),
                folded_nodes_encodings=encoded_paths.folded_nodes_encodings,
                combined_paths=FlattenedTensor(
                    flattened=encoded_paths.combined_paths,
                    unflattener_mask=cfg_paths_input.nodes_indices.unflattener_mask,
                    unflattener_fn=cfg_paths_input.nodes_indices.unflatten))

    def get_cfg_control_flow_paths_ngrams_input(
            self, pdg_input: PDGInputTensors) -> Mapping[int, CFGPathsNGramsInputTensors]:
        assert self.params.is_ngrams
        ngrams_min_n = self.params.ngrams.min_n
        ngrams_max_n = self.params.ngrams.max_n
        all_ngram_ns = \
            set(pdg_input.cfg_control_flow_paths_exact_ngrams.keys()) | \
            set(pdg_input.cfg_control_flow_paths_partial_ngrams.keys())
        assert len(all_ngram_ns) > 0
        ngrams_n_range_start = min(all_ngram_ns) if ngrams_min_n is None else max(min(all_ngram_ns), ngrams_min_n)
        ngrams_n_range_end = max(all_ngram_ns) if ngrams_max_n is None else min(max(all_ngram_ns), ngrams_max_n)
        assert ngrams_n_range_start <= ngrams_n_range_end
        cfg_control_flow_paths_ngrams_input = {
            ngram_n: ngrams
            for ngram_n, ngrams in pdg_input.cfg_control_flow_paths_exact_ngrams.items()
            if ngrams_n_range_start <= ngram_n <= ngrams_n_range_end}
        if ngrams_max_n in pdg_input.cfg_control_flow_paths_partial_ngrams:
            cfg_control_flow_paths_ngrams_input[ngrams_max_n + 1] = \
                pdg_input.cfg_control_flow_paths_partial_ngrams[ngrams_max_n]
        return cfg_control_flow_paths_ngrams_input
