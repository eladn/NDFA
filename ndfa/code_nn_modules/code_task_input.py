__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2020-06-08"

import torch
import dgl
import dataclasses
from torch_geometric.data import Data as TGData
from typing import Optional, Callable, Dict, List

from ndfa.misc.tensors_data_class import TensorsDataClass, BatchFlattenedTensor, BatchFlattenedSeq, \
    BatchedFlattenedIndicesFlattenedTensor, BatchedFlattenedIndicesFlattenedSeq, \
    batched_flattened_indices_flattened_tensor_field, batched_flattened_indices_flattened_seq_field, \
    BatchedFlattenedIndicesPseudoRandomPermutation, BatchFlattenedPseudoRandomSamplerFromRange, \
    batch_flattened_pseudo_random_sampler_from_range_field, BatchFlattenedSeqShuffler, \
    batch_flattened_seq_shuffler_field, TensorsDataDict, batch_flattened_tensor_field, batch_flattened_seq_field, \
    batch_flattened_indices_pseudo_random_permutation_field, FragmentSizeDistribution
from ndfa.nn_utils.model_wrapper.flattened_tensor import FlattenedTensor
from ndfa.nn_utils.modules.params.sampling_params import SamplingParams
from ndfa.code_tasks.method_code_preprocess_params import HierarchicMethodEncoderPreprocessParams, \
    MethodCodePreprocessParams, ASTPreprocessParams


__all__ = [
   'MethodCodeInputTensors', 'CodeExpressionTokensSequenceInputTensors',
   'MethodCodeTokensSequenceInputTensors', 'CFGCodeExpressionTokensSequenceInputTensors',
   'SymbolsInputTensors', 'CFGPathsInputTensors', 'CFGPathsNGramsInputTensors',
   'PDGInputTensors', 'MethodASTInputTensors', 'SubASTInputTensors', 'IdentifiersInputTensors',
   'PDGExpressionsSubASTInputTensors', 'PrunedUpperASTInputTensors'
]


@dataclasses.dataclass
class CodeExpressionTokensSequenceInputTensors(TensorsDataClass):
    @classmethod
    def get_fragmented_shuffling_params(cls, collate_data) -> Optional[FragmentSizeDistribution]:
        flat_tokens_seq_params = collate_data.model_hps.method_code_encoder.get_flat_tokens_seq_code_encoder_params()
        if not flat_tokens_seq_params or not flat_tokens_seq_params.shuffle_expressions or \
                not flat_tokens_seq_params.shuffling_options.fragmented_shuffling:
            return None
        return flat_tokens_seq_params.shuffling_options.fragmented_shuffling_distribution_params

    # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    token_type: BatchFlattenedSeq = \
        batch_flattened_seq_field(self_indexing_group='code_expressions')
    # (nr_kos_tokens_in_all_expressions_in_batch,)
    kos_token_index: BatchFlattenedTensor = \
        batch_flattened_tensor_field()
    # (nr_identifier_tokens_in_all_expressions_in_batch,)
    identifier_index: BatchedFlattenedIndicesFlattenedTensor = batched_flattened_indices_flattened_tensor_field(
        tgt_indexing_group='identifiers')
    # (nr_symbol_occurrences_in_all_expressions_in_batch,)
    symbol_index: BatchedFlattenedIndicesFlattenedTensor = batched_flattened_indices_flattened_tensor_field(
        tgt_indexing_group='symbols')
    # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    is_symbol_mask: BatchFlattenedSeq = batch_flattened_seq_field()
    # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    sequence_shuffler: BatchFlattenedSeqShuffler = batch_flattened_seq_shuffler_field(
        initial_seed_salt='code_expressions_seq_shuffler',
        fragmented_shuffling_fragments_size_distribution=get_fragmented_shuffling_params)
    token_idx_to_ast_leaf_idx_mapping_key: Optional[BatchedFlattenedIndicesFlattenedTensor] = \
        batched_flattened_indices_flattened_tensor_field(default=None)
    token_idx_to_ast_leaf_idx_mapping_value: Optional[BatchedFlattenedIndicesFlattenedTensor] = \
        batched_flattened_indices_flattened_tensor_field(default=None)


@dataclasses.dataclass
class MethodCodeTokensSequenceInputTensors(CodeExpressionTokensSequenceInputTensors):
    # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    token_type: BatchFlattenedSeq = \
        batch_flattened_seq_field(self_indexing_group='method_code')


@dataclasses.dataclass
class CFGCodeExpressionTokensSequenceInputTensors(CodeExpressionTokensSequenceInputTensors):
    # (nr_expressions_in_batch, batch_max_nr_tokens_in_expr)
    token_type: BatchFlattenedSeq = \
        batch_flattened_seq_field(self_indexing_group='cfg_code_expressions')

    def batch_flattened_tokens_seqs_as_unflattenable(
            self, tokens_seq_encodings: torch.Tensor) -> FlattenedTensor:
        return FlattenedTensor(
            flattened=tokens_seq_encodings,
            unflattener_mask_getter=self.get_expressions_per_example_unflattener_mask,
            unflattener_fn=self.flatten_expressions_per_example)

    def flatten_expressions_per_example(self, tokens_seq_encodings: torch.Tensor) -> torch.Tensor:
        assert tokens_seq_encodings.ndim == 3  # (#seqs_in_batch, seq_len, embd)
        unflattened_tokens_seq = self.token_type.unflatten(tokens_seq_encodings)
        assert unflattened_tokens_seq.ndim == 4  # (#examples_in_batch, #seqs_in_example, seq_len, embd)
        return unflattened_tokens_seq.flatten(1, 2)  # (#examples_in_batch, len_of_concatenated_seqs_per_example, embd)

    def get_expressions_per_example_unflattener_mask(self) -> torch.Tensor:
        assert self.token_type.sequences_mask.ndim == 2  # (#seqs_in_batch, seq_len)
        unflattened_tokens_seq_mask = self.token_type.unflatten(self.token_type.sequences_mask)
        assert unflattened_tokens_seq_mask.ndim == 3  # (#examples_in_batch, #seqs_in_example, seq_len)
        return unflattened_tokens_seq_mask.flatten(1, 2)  # (#examples_in_batch, len_of_concatenated_seqs_per_example)


@dataclasses.dataclass
class SymbolsInputTensors(TensorsDataClass):
    # (nr_symbols_in_batch,)
    # value meaning: identifier batched index
    symbols_identifier_indices: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(
            self_indexing_group='symbols', tgt_indexing_group='identifiers')
    # not used
    # # (nr_symbols_appearances,)
    # symbols_appearances_symbol_idx: BatchedFlattenedIndicesFlattenedTensor = \
    #     batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='symbols')
    # # (nr_symbols_appearances,)
    # symbols_appearances_expression_token_idx: Optional[BatchFlattenedTensor] = \
    #     batch_flattened_tensor_field(default=None)
    # # (nr_symbols_appearances,)
    # symbols_appearances_cfg_expression_idx: Optional[BatchedFlattenedIndicesFlattenedTensor] = \
    #     batched_flattened_indices_flattened_tensor_field(
    #         default=None, tgt_indexing_group='cfg_code_expressions')


@dataclasses.dataclass
class CFGPathsInputTensors(TensorsDataClass):
    nodes_indices: BatchedFlattenedIndicesFlattenedSeq = \
        batched_flattened_indices_flattened_seq_field(tgt_indexing_group='cfg_nodes')
    edges_types: BatchFlattenedSeq = batch_flattened_seq_field()


@dataclasses.dataclass
class CFGPathsNGramsInputTensors(TensorsDataClass):
    nodes_indices: BatchedFlattenedIndicesFlattenedSeq = \
        batched_flattened_indices_flattened_seq_field(tgt_indexing_group='cfg_nodes')
    edges_types: BatchFlattenedSeq = batch_flattened_seq_field()


@dataclasses.dataclass
class PDGInputTensors(TensorsDataClass):
    # (nr_cfg_nodes_in_batch, )
    cfg_nodes_control_kind: Optional[BatchFlattenedTensor] = \
        batch_flattened_tensor_field(default=None, self_indexing_group='cfg_nodes')
    # (nr_cfg_nodes_in_batch, )
    cfg_nodes_has_expression_mask: Optional[BatchFlattenedTensor] = \
        batch_flattened_tensor_field(default=None)
    cfg_nodes_tokenized_expressions: Optional[CFGCodeExpressionTokensSequenceInputTensors] = None
    # cfg_nodes_expressions_ref_to_method_tokenized_expressions: Optional[BatchFlattenedTensor] = \
    #     batch_flattened_tensor_field(default=None)
    cfg_nodes_expressions_ast: Optional['PDGExpressionsSubASTInputTensors'] = None
    cfg_macro_trimmed_ast: Optional['PrunedUpperASTInputTensors'] = None

    # cfg_edges: Optional[torch.LongTensor] = None
    # cfg_edges_lengths: Optional[torch.BoolTensor] = None
    # cfg_edges_attrs: Optional[torch.LongTensor] = None

    cfg_nodes_random_permutation: Optional[BatchedFlattenedIndicesPseudoRandomPermutation] = \
        batch_flattened_indices_pseudo_random_permutation_field(
            default=None, tgt_indexing_group='cfg_nodes', batch_dependent_seed=True,
            example_dependent_seed=True, initial_seed_salt='cfgn')
    cfg_control_flow_paths: Optional[CFGPathsInputTensors] = None
    cfg_pdg_paths: Optional[CFGPathsInputTensors] = None
    cfg_control_flow_paths_exact_ngrams: Optional[TensorsDataDict[int, CFGPathsNGramsInputTensors]] = None
    cfg_control_flow_paths_partial_ngrams: Optional[TensorsDataDict[int, CFGPathsNGramsInputTensors]] = None

    cfg_control_flow_graph: Optional[TGData] = None

    @property
    def nr_cfg_nodes(self) -> int:
        return self.cfg_nodes_control_kind.tensor.size(0)

    def unflatten_cfg_nodes_encodings(self, cfg_nodes_encodings: torch.Tensor) -> torch.Tensor:
        return self.cfg_nodes_control_kind.unflatten(cfg_nodes_encodings)

    def get_cfg_nodes_encodings_unflattener_mask(self):
        return self.cfg_nodes_control_kind.unflattener_mask

    def keep_only_relevant_fields_according_to_preprocess_params(
            self, preprocess_params: HierarchicMethodEncoderPreprocessParams):
        return dataclasses.replace(
            self,
            cfg_nodes_tokenized_expressions=self.cfg_nodes_tokenized_expressions
            if preprocess_params.micro_tokens_seq else None,
            cfg_nodes_expressions_ast=
            self.cfg_nodes_expressions_ast.keep_only_relevant_fields_according_to_preprocess_params(
                preprocess_params=preprocess_params.micro_ast)
            if preprocess_params.micro_ast else None,
            cfg_macro_trimmed_ast=
            self.cfg_macro_trimmed_ast.keep_only_relevant_fields_according_to_preprocess_params(
                preprocess_params=preprocess_params.macro_ast)
            if preprocess_params.macro_ast else None,
            cfg_control_flow_paths=self.cfg_control_flow_paths
            if preprocess_params.control_flow_paths and
               preprocess_params.control_flow_paths.full_paths else None,
            cfg_nodes_random_permutation=self.cfg_nodes_random_permutation
            if preprocess_params.control_flow_single_flat_seq and
               preprocess_params.control_flow_single_flat_seq.cfg_nodes_random_permutation else None,
            cfg_control_flow_paths_exact_ngrams=self.cfg_control_flow_paths_exact_ngrams
            if preprocess_params.control_flow_paths and preprocess_params.control_flow_paths.ngrams else None,
            cfg_control_flow_paths_partial_ngrams=self.cfg_control_flow_paths_partial_ngrams
            if preprocess_params.control_flow_paths and preprocess_params.control_flow_paths.ngrams else None,
            cfg_control_flow_graph=self.cfg_control_flow_graph if preprocess_params.control_flow_graph else None)


@dataclasses.dataclass
class IdentifiersInputTensors(TensorsDataClass):
    # (nr_sub_parts_in_batch, )  # TODO: is it necessary?
    sub_parts_batch: BatchFlattenedTensor = \
        batch_flattened_tensor_field(self_indexing_group='identifiers_sub_parts__')
    # (nr_sub_parts_in_batch, )
    sub_parts_vocab_word_index: BatchFlattenedTensor = \
        batch_flattened_tensor_field(self_indexing_group='identifiers_sub_parts')

    # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier)
    identifier_sub_parts_index: BatchedFlattenedIndicesFlattenedSeq = \
        batched_flattened_indices_flattened_seq_field(
            self_indexing_group='identifiers__', tgt_indexing_group='identifiers_sub_parts')
    # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier)
    identifier_sub_parts_vocab_word_index: BatchFlattenedSeq = \
        batch_flattened_seq_field(self_indexing_group='identifiers')
    # (nr_identifiers_in_batch, )
    identifiers_vocab_word_index: BatchFlattenedTensor = \
        batch_flattened_tensor_field(self_indexing_group='identifiers___')
    # (nr_identifiers_in_batch, batch_max_nr_sub_parts_in_identifier, nr_hashing_features)
    identifier_sub_parts_hashings: BatchFlattenedSeq = \
        batch_flattened_seq_field(self_indexing_group='identifiers____')
    # (nr_sub_parts_obfuscation_embeddings)
    sub_parts_obfuscation: BatchFlattenedPseudoRandomSamplerFromRange = \
        batch_flattened_pseudo_random_sampler_from_range_field(
            initial_seed_salt='idntf', replacement='wo_replacement_within_example')
    # (nr_identifiers_obfuscation_embeddings)
    identifiers_obfuscation: BatchFlattenedPseudoRandomSamplerFromRange = \
        batch_flattened_pseudo_random_sampler_from_range_field(
            initial_seed_salt='idntf', replacement='wo_replacement_within_example')


# To avoid IDE errors
def dataclasses_field_wo_defaults():
    return dataclasses.field()


# TODO: `ASTPathsInputTensors`
@dataclasses.dataclass
class SubASTInputTensors(TensorsDataClass):
    @classmethod
    def _get_ast_leaf_to_leaf_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        return cls.get_ast_leaf_to_leaf_paths_dataloading_sampling_params(collate_data)

    @classmethod
    def _get_ast_leaf_to_root_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        return cls.get_ast_leaf_to_root_paths_dataloading_sampling_params(collate_data)

    @classmethod
    def get_ast_leaf_to_root_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        raise NotImplementedError  # sub-class should implement this method

    @classmethod
    def get_ast_leaf_to_leaf_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        raise NotImplementedError  # sub-class should implement this method

    ast_leaf_to_leaf_paths_node_indices: Optional[BatchedFlattenedIndicesFlattenedSeq] = \
        batched_flattened_indices_flattened_seq_field(
            tgt_indexing_group='ast_nodes', sequences_sampling_initial_seed_salt='astpth',
            sequences_per_example_sampling=_get_ast_leaf_to_leaf_paths_dataloading_sampling_params)
    ast_leaf_to_leaf_paths_child_place: Optional[BatchFlattenedSeq] = \
        batch_flattened_seq_field(
            sequences_sampling_initial_seed_salt='astpth',
            sequences_per_example_sampling=_get_ast_leaf_to_leaf_paths_dataloading_sampling_params)
    ast_leaf_to_leaf_paths_vertical_direction: Optional[BatchFlattenedSeq] = \
        batch_flattened_seq_field(
            sequences_sampling_initial_seed_salt='astpth',
            sequences_per_example_sampling=_get_ast_leaf_to_leaf_paths_dataloading_sampling_params)
    ast_leaf_to_leaf_paths_shuffler: Optional[BatchFlattenedSeqShuffler] = \
        batch_flattened_seq_shuffler_field(initial_seed_salt='ast_leaf_to_leaf_paths_shuffler')
    ast_leaf_to_root_paths_node_indices: Optional[BatchedFlattenedIndicesFlattenedSeq] = \
        batched_flattened_indices_flattened_seq_field(
            tgt_indexing_group='ast_nodes', sequences_sampling_initial_seed_salt='astpth',
            sequences_per_example_sampling=_get_ast_leaf_to_root_paths_dataloading_sampling_params)
    ast_leaf_to_root_paths_child_place: Optional[BatchFlattenedSeq] = \
        batch_flattened_seq_field(
            sequences_sampling_initial_seed_salt='astpth',
            sequences_per_example_sampling=_get_ast_leaf_to_root_paths_dataloading_sampling_params)
    ast_leaf_to_root_paths_shuffler: Optional[BatchFlattenedSeqShuffler] = \
        batch_flattened_seq_shuffler_field(initial_seed_salt='ast_leaf_to_root_paths_shuffler')
    ast_leaves_sequence_node_indices: Optional[BatchedFlattenedIndicesFlattenedSeq] = \
        batched_flattened_indices_flattened_seq_field(tgt_indexing_group='ast_nodes')
    ast_leaves_sequence_shuffler: Optional[BatchFlattenedSeqShuffler] = \
        batch_flattened_seq_shuffler_field(initial_seed_salt='ast_leaves_sequence_shuffler')
    siblings_sequences_node_indices: Optional[BatchedFlattenedIndicesFlattenedSeq] = \
        batched_flattened_indices_flattened_seq_field(tgt_indexing_group='ast_nodes')
    siblings_w_parent_sequences_node_indices: Optional[BatchedFlattenedIndicesFlattenedSeq] = \
        batched_flattened_indices_flattened_seq_field(tgt_indexing_group='ast_nodes')
    dgl_tree: Optional[dgl.DGLGraph] = dataclasses_field_wo_defaults()  # To avoid IDE errors
    pyg_graph: Optional[TGData] = dataclasses_field_wo_defaults()  # To avoid IDE errors

    def get_ast_paths_node_indices(self, path_type: str) -> BatchedFlattenedIndicesFlattenedSeq:
        if path_type == 'leaf_to_leaf':
            return self.ast_leaf_to_leaf_paths_node_indices
        elif path_type == 'leaf_to_root':
            return self.ast_leaf_to_root_paths_node_indices
        elif path_type == 'leaves_sequence':
            return self.ast_leaves_sequence_node_indices
        elif path_type == 'siblings_sequences':
            return self.siblings_sequences_node_indices
        elif path_type == 'siblings_w_parent_sequences':
            return self.siblings_w_parent_sequences_node_indices
        else:
            raise ValueError(f'Unsupported path type `{path_type}`.')

    @classmethod
    def path_type_has_child_place(cls, path_type: str) -> bool:
        return path_type in {'leaf_to_leaf', 'leaf_to_root'}

    def get_ast_paths_child_place(self, path_type: str) -> Optional[BatchFlattenedSeq]:
        if path_type == 'leaf_to_leaf':
            return self.ast_leaf_to_leaf_paths_child_place
        elif path_type == 'leaf_to_root':
            return self.ast_leaf_to_root_paths_child_place
        elif path_type in {'leaves_sequence', 'siblings_sequences', 'siblings_w_parent_sequences'}:
            return None
        else:
            raise ValueError(f'Unsupported path type `{path_type}`.')

    @classmethod
    def path_type_has_vertical_direction(cls, path_type: str) -> bool:
        return path_type == 'leaf_to_leaf'

    def get_ast_paths_vertical_direction(self, path_type: str) -> Optional[BatchFlattenedSeq]:
        if path_type == 'leaf_to_leaf':
            return self.ast_leaf_to_leaf_paths_vertical_direction
        elif path_type in {'leaf_to_root', 'leaves_sequence', 'siblings_sequences', 'siblings_w_parent_sequences'}:
            return None
        else:
            raise ValueError(f'Unsupported path type `{path_type}`.')

    def get_ast_paths_shuffler(self, path_type: str) -> BatchFlattenedSeqShuffler:
        if path_type == 'leaf_to_leaf':
            return self.ast_leaf_to_leaf_paths_shuffler
        elif path_type == 'leaf_to_root':
            return self.ast_leaf_to_root_paths_shuffler
        elif path_type == 'leaves_sequence':
            return self.ast_leaves_sequence_shuffler
        else:
            raise ValueError(f'Unsupported path type `{path_type}`.')

    def get_ast_paths_unflattener_mask(self, paths_types: List[str]) -> torch.Tensor:
        def _get_unflatterer_mask_by_type(paths_type: str):
            flatteners_mask_dict = {
                'leaf_to_leaf': self.ast_leaf_to_leaf_paths_node_indices.unflattener_mask,
                'leaf_to_root': self.ast_leaf_to_root_paths_node_indices.unflattener_mask}
            return flatteners_mask_dict[paths_type]

        return torch.cat([_get_unflatterer_mask_by_type(paths_type) for paths_type in paths_types], dim=1)

    def get_ast_paths_unflattener(self, paths_types: List[str]) -> Callable[[torch.Tensor], torch.Tensor]:
        def _get_unflatterer_by_type(paths_type: str):
            flatteners_dict = {
                'leaf_to_leaf': self.ast_leaf_to_leaf_paths_node_indices.unflatten,
                'leaf_to_root': self.ast_leaf_to_root_paths_node_indices.unflatten}
            return flatteners_dict[paths_type]

        def unflatten(all_combined_paths_encodings: Dict[str, torch.Tensor]):
            return torch.cat([_get_unflatterer_by_type(paths_type)(all_combined_paths_encodings[paths_type]) for paths_type in paths_types], dim=1)

        return unflatten

    def batch_flattened_combined_ast_paths_as_unflattenable(
            self, combined_ast_paths_encodings_by_type: Dict[str, torch.Tensor]) -> FlattenedTensor:
        return FlattenedTensor(
            # flattened=torch.cat(list(combined_ast_paths_encodings_by_type.values()), dim=0),
            flattened=combined_ast_paths_encodings_by_type,
            unflattener_mask_getter=lambda: self.get_ast_paths_unflattener_mask(
                paths_types=list(combined_ast_paths_encodings_by_type.keys())),
            unflattener_fn=self.get_ast_paths_unflattener(
                paths_types=list(combined_ast_paths_encodings_by_type.keys())))

    def keep_only_relevant_fields_according_to_preprocess_params(
            self, preprocess_params: ASTPreprocessParams):
        if preprocess_params is None:
            # Possible in case of inheritor `MethodASTInputTensors` is kept (for info about AST nodes types) without
            # keeping the structural data about the AST of the entire method.
            return dataclasses.replace(
                self,
                ast_leaf_to_leaf_paths_node_indices=None,
                ast_leaf_to_leaf_paths_child_place=None,
                ast_leaf_to_leaf_paths_vertical_direction=None,
                ast_leaf_to_leaf_paths_shuffler=None,
                ast_leaf_to_root_paths_node_indices=None,
                ast_leaf_to_root_paths_child_place=None,
                ast_leaf_to_root_paths_shuffler=None,
                ast_leaves_sequence_node_indices=None,
                ast_leaves_sequence_shuffler=None,
                siblings_sequences_node_indices=None,
                siblings_w_parent_sequences_node_indices=None,
                dgl_tree=None,
                pyg_graph=None)
        return dataclasses.replace(
            self,
            ast_leaf_to_leaf_paths_node_indices=self.ast_leaf_to_leaf_paths_node_indices
            if preprocess_params.paths and preprocess_params.paths.leaf_to_leaf else None,
            ast_leaf_to_leaf_paths_child_place=self.ast_leaf_to_leaf_paths_child_place
            if preprocess_params.paths and preprocess_params.paths.leaf_to_leaf and
               preprocess_params.paths.traversal else None,
            ast_leaf_to_leaf_paths_vertical_direction=self.ast_leaf_to_leaf_paths_vertical_direction
            if preprocess_params.paths and preprocess_params.paths.leaf_to_leaf and
               preprocess_params.paths.traversal else None,
            ast_leaf_to_leaf_paths_shuffler=self.ast_leaf_to_leaf_paths_shuffler
            if preprocess_params.paths and preprocess_params.paths.leaf_to_leaf and
               preprocess_params.paths.leaf_to_leaf_shuffler else None,
            ast_leaf_to_root_paths_node_indices=self.ast_leaf_to_root_paths_node_indices
            if preprocess_params.paths and preprocess_params.paths.leaf_to_root else None,
            ast_leaf_to_root_paths_child_place=self.ast_leaf_to_root_paths_child_place
            if preprocess_params.paths and preprocess_params.paths.leaf_to_root and
               preprocess_params.paths.traversal else None,
            ast_leaf_to_root_paths_shuffler=self.ast_leaf_to_root_paths_shuffler
            if preprocess_params.paths and preprocess_params.paths.leaf_to_root and
               preprocess_params.paths.leaf_to_root_shuffler else None,
            ast_leaves_sequence_node_indices=self.ast_leaves_sequence_node_indices
            if preprocess_params.paths and preprocess_params.paths.leaves_sequence else None,
            ast_leaves_sequence_shuffler=self.ast_leaves_sequence_shuffler
            if preprocess_params.paths and preprocess_params.paths.leaves_sequence and
               preprocess_params.paths.leaves_sequence_shuffler else None,
            siblings_sequences_node_indices=self.siblings_sequences_node_indices
            if preprocess_params.paths and preprocess_params.paths.siblings_sequences else None,
            siblings_w_parent_sequences_node_indices=self.siblings_w_parent_sequences_node_indices
            if preprocess_params.paths and preprocess_params.paths.siblings_w_parent_sequences else None,
            dgl_tree=self.dgl_tree if preprocess_params.dgl_tree else None,
            pyg_graph=self.pyg_graph if preprocess_params.pyg_graph else None)


@dataclasses.dataclass
class PrunedUpperASTInputTensors(SubASTInputTensors):
    @classmethod
    def get_ast_leaf_to_root_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        return collate_data.model_hps.method_code_encoder.upper_pruned_ast_leaf_to_root_paths_dataloading_sampling_params

    @classmethod
    def get_ast_leaf_to_leaf_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        return collate_data.model_hps.method_code_encoder.upper_pruned_ast_leaf_to_leaf_paths_dataloading_sampling_params


@dataclasses.dataclass
class MethodASTInputTensors(SubASTInputTensors):
    ast_node_types: BatchFlattenedTensor = batch_flattened_tensor_field(self_indexing_group='ast_nodes')
    ast_node_major_types: BatchFlattenedTensor = batch_flattened_tensor_field()
    ast_node_minor_types: BatchFlattenedTensor = batch_flattened_tensor_field()
    ast_node_child_ltr_position: BatchFlattenedTensor = batch_flattened_tensor_field()
    ast_node_child_rtl_position: BatchFlattenedTensor = batch_flattened_tensor_field()
    ast_node_nr_children: BatchFlattenedTensor = batch_flattened_tensor_field()

    ast_nodes_with_identifier_leaf_nodes_indices: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='ast_nodes')
    ast_nodes_with_identifier_leaf_identifier_idx: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='identifiers')

    ast_nodes_with_symbol_leaf_nodes_indices: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='ast_nodes')
    ast_nodes_with_symbol_leaf_symbol_idx: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='symbols')
    ast_nodes_symbol_idx: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='symbols')
    ast_nodes_has_symbol_mask: BatchFlattenedTensor = \
        batch_flattened_tensor_field()

    ast_nodes_with_primitive_type_leaf_nodes_indices: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='ast_nodes')
    ast_nodes_with_primitive_type_leaf_primitive_type: BatchFlattenedTensor = \
        batch_flattened_tensor_field()

    ast_nodes_with_modifier_leaf_nodes_indices: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='ast_nodes')
    ast_nodes_with_modifier_leaf_modifier: BatchFlattenedTensor = \
        batch_flattened_tensor_field()

    def get_ast_nodes_unflattener(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.ast_node_types.unflatten

    def unflatten_ast_nodes_encodings(self, ast_nodes_encodings: torch.Tensor) -> torch.Tensor:
        return self.ast_node_types.unflatten(ast_nodes_encodings)

    def get_ast_nodes_unflattener_mask(self) -> torch.Tensor:
        return self.ast_node_types.unflattener_mask

    def batch_flattened_ast_nodes_as_unflattenable(
            self, ast_nodes_encodings: torch.Tensor) -> FlattenedTensor:
        return FlattenedTensor(
            flattened=ast_nodes_encodings,
            unflattener_mask_getter=self.get_ast_nodes_unflattener_mask,
            unflattener_fn=self.get_ast_nodes_unflattener())

    @classmethod
    def get_ast_leaf_to_root_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        return collate_data.model_hps.method_code_encoder.method_ast_leaf_to_root_paths_dataloading_sampling_params

    @classmethod
    def get_ast_leaf_to_leaf_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        return collate_data.model_hps.method_code_encoder.method_ast_leaf_to_leaf_paths_dataloading_sampling_params


@dataclasses.dataclass
class PDGExpressionsSubASTInputTensors(SubASTInputTensors):
    ast_leaf_to_leaf_paths_pdg_node_indices: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(
            tgt_indexing_group='cfg_nodes',
            sampling_initial_seed_salt='astpth',
            per_example_sampling=lambda collate_data:
            collate_data.model_hps.method_code_encoder.sub_asts_leaf_to_leaf_paths_dataloading_sampling_params)
    ast_leaf_to_root_paths_pdg_node_indices: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(
            tgt_indexing_group='cfg_nodes',
            sampling_initial_seed_salt='astpth',
            per_example_sampling=lambda collate_data:
            collate_data.model_hps.method_code_encoder.sub_asts_leaf_to_root_paths_dataloading_sampling_params)
    siblings_sequences_pdg_node_indices: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(
            tgt_indexing_group='cfg_nodes')

    pdg_node_idx_to_sub_ast_root_idx_mapping_key: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='cfg_nodes')
    pdg_node_idx_to_sub_ast_root_idx_mapping_value: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='ast_nodes')
    ast_node_idx_to_pdg_node_idx_mapping_key: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='ast_nodes')
    ast_node_idx_to_pdg_node_idx_mapping_value: BatchedFlattenedIndicesFlattenedTensor = \
        batched_flattened_indices_flattened_tensor_field(tgt_indexing_group='cfg_nodes')

    def get_ast_paths_pdg_node_indices(self, path_type: str) -> Optional[BatchedFlattenedIndicesFlattenedTensor]:
        if path_type == 'leaf_to_leaf':
            return self.ast_leaf_to_leaf_paths_pdg_node_indices
        elif path_type == 'leaf_to_root':
            return self.ast_leaf_to_root_paths_pdg_node_indices
        elif path_type == 'leaves_sequence':
            return None
        elif path_type in {'siblings_sequences', 'siblings_w_parent_sequences'}:
            return self.siblings_sequences_pdg_node_indices
        else:
            raise ValueError(f'Unsupported path type `{path_type}`.')

    @classmethod
    def get_ast_leaf_to_root_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        return collate_data.model_hps.method_code_encoder.sub_asts_leaf_to_root_paths_dataloading_sampling_params

    @classmethod
    def get_ast_leaf_to_leaf_paths_dataloading_sampling_params(cls, collate_data) -> Optional[SamplingParams]:
        return collate_data.model_hps.method_code_encoder.sub_asts_leaf_to_leaf_paths_dataloading_sampling_params


@dataclasses.dataclass
class MethodCodeInputTensors(TensorsDataClass):
    method_hash: str

    identifiers: IdentifiersInputTensors
    symbols: SymbolsInputTensors

    method_tokenized_code: Optional[MethodCodeTokensSequenceInputTensors] = None
    pdg: Optional[PDGInputTensors] = None
    ast: Optional[MethodASTInputTensors] = None

    def keep_only_relevant_fields_according_to_preprocess_params(self, preprocess_params: MethodCodePreprocessParams):
        return dataclasses.replace(
            self,
            method_tokenized_code=self.method_tokenized_code
            if preprocess_params.whole_method_tokens_seq else None,
            pdg=self.pdg.keep_only_relevant_fields_according_to_preprocess_params(
                preprocess_params=preprocess_params.hierarchic)
            if preprocess_params.hierarchic else None,
            ast=self.ast.keep_only_relevant_fields_according_to_preprocess_params(
                preprocess_params=preprocess_params.whole_method_ast)
            if preprocess_params.general_ast else None)
