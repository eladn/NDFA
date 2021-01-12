import os
import io
import torch
import itertools
import functools
import dataclasses
import multiprocessing as mp
from collections import defaultdict, namedtuple
from warnings import warn
from typing import Iterable, Collection, Any, Set, Optional, Dict, List, Union
from typing_extensions import Protocol
from sklearn.feature_extraction.text import HashingVectorizer
from torch_geometric.data import Data as TGData
import dgl

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams
from ndfa.nn_utils.model_wrapper.dataset_properties import DataFold
from ndfa.misc.code_data_structure_api import SerMethod, SerMethodPDG, SerMethodAST, SerToken, SerTokenKind, \
    SerASTNodeType, SerPDGNodeControlKind
from ndfa.misc.code_data_structure_utils import get_pdg_node_tokenized_expression, get_all_pdg_simple_paths, \
    get_all_ast_paths, ASTPaths, traverse_ast
from ndfa.nn_utils.model_wrapper.chunked_random_access_dataset import ChunkedRandomAccessDatasetWriter
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs, kos_token_to_kos_token_vocab_word
from ndfa.code_nn_modules.code_task_input import MethodCodeInputPaddedTensors, MethodCodeInputTensors, \
    CodeExpressionTokensSequenceInputTensors, SymbolsInputTensors, PDGInputTensors, CFGPathsInputTensors, \
    CFGPathsNGramsInputTensors, IdentifiersInputTensors, MethodASTInputTensors, PDGExpressionsSubASTInputTensors
from ndfa.misc.tensors_data_class import BatchFlattenedTensor, BatchFlattenedSeq, \
    TensorWithCollateMask, BatchedFlattenedIndicesFlattenedTensor, BatchedFlattenedIndicesFlattenedSeq, \
    BatchedFlattenedIndicesPseudoRandomPermutation, BatchFlattenedPseudoRandomSamplerFromRange, \
    BatchFlattenedSeqShuffler, TensorsDataDict

__all__ = [
    'preprocess_code_task_dataset', 'preprocess_code_task_example', 'truncate_and_pad', 'PreprocessLimitExceedError',
    'PreprocessLimitation']


def token_to_input_vector(token: SerToken, vocabs: CodeTaskVocabs):
    assert token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR,
                          SerTokenKind.IDENTIFIER, SerTokenKind.LITERAL}
    if token.kind == SerTokenKind.IDENTIFIER:
        return [vocabs.tokens_kinds.get_word_idx(token.kind.value),
                token.identifier_idx,
                (-1 if token.symbol_idx is None else token.symbol_idx)]
    if token.kind == SerTokenKind.LITERAL:
        return [vocabs.tokens_kinds.get_word_idx(token.kind.value),
                vocabs.kos_tokens.get_word_idx('<PAD>'), -1]  # TODO: add some '<NON-RELEVANT>' special word
    return [vocabs.tokens_kinds.get_word_idx(token.kind.value),
            vocabs.kos_tokens.get_word_idx_or_unk(kos_token_to_kos_token_vocab_word(token)), -1]


def truncate_and_pad(vector: Collection, max_length: int, pad_word: str = '<PAD>') -> Iterable:
    vector_truncated_len = min(len(vector), max_length)
    padding_len = max_length - vector_truncated_len
    return itertools.chain(itertools.islice(vector, max_length), (pad_word for _ in range(padding_len)))


@dataclasses.dataclass
class PreprocessLimitation:
    object_name: str
    value: Union[int, float]
    min_val: Union[int, float] = None
    max_val: Union[int, float] = None
    custom_msg: Optional[str] = None
    warn: bool = False

    @property
    def exceeds(self) -> bool:
        return (self.min_val is not None and self.value < self.min_val) or \
               (self.max_val is not None and self.value > self.max_val)

    def __str__(self):
        if not self.exceeds:
            return f'Limitation does not exceed.'
        if self.custom_msg is not None:
            return self.custom_msg
        msg = f'Limitation exceed: `{self.object_name}` ({self.value})'
        if self.min_val is not None and self.value < self.min_val:
            msg += f' < {self.min_val}'
        if self.max_val is not None and self.value > self.max_val:
            msg += f' > {self.max_val}'
        return msg

    @classmethod
    def enforce_limitations(cls, limitations: List['PreprocessLimitation']):
        exceeding_limitations = [limitation for limitation in limitations if limitation.exceeds]
        for exceeding_limitation in exceeding_limitations:
            if exceeding_limitation.warn:
                warn(str(exceeding_limitation))
        if len(exceeding_limitations) > 0:
            raise PreprocessLimitExceedError(exceeding_limitations=exceeding_limitations)


class PreprocessLimitExceedError(ValueError):
    def __init__(self, exceeding_limitations: List[PreprocessLimitation]):
        self.exceeding_limitations = exceeding_limitations
        assert all(limitation.exceeds for limitation in exceeding_limitations)
        msg = '; '.join(str(limitation) for limitation in exceeding_limitations)
        super(PreprocessLimitExceedError, self).__init__(msg)


def enforce_code_task_input_pp_limitations(
        model_hps: NDFAModelHyperParams, method: SerMethod, method_pdg: SerMethodPDG, method_ast: SerMethodAST):
    limitations = []
    limitations.append(PreprocessLimitation(
            object_name='#identifiers', value=len(method_pdg.sub_identifiers_by_idx),
            max_val=model_hps.method_code_encoder.max_nr_identifiers))
    min_sub_identifiers_in_identifier = min(
        (len(sub_identifiers_in_identifier) for sub_identifiers_in_identifier in method_pdg.sub_identifiers_by_idx),
        default=float('inf'))
    limitations.append(PreprocessLimitation(
        object_name='#sub_identifiers', value=min_sub_identifiers_in_identifier,
        min_val=1, custom_msg=f'Empty identifier (no sub-identifiers) in method {method.hash}.', warn=True))
    limitations.append(PreprocessLimitation(
        object_name='#pdg_nodes', value=len(method_pdg.pdg_nodes),
        min_val=model_hps.method_code_encoder.min_nr_pdg_nodes,
        max_val=model_hps.method_code_encoder.max_nr_pdg_nodes))
    nr_pdg_nodes_with_expression = sum(
        (int(pdg_node.code_sub_token_range_ref is not None) for pdg_node in method_pdg.pdg_nodes), start=0)
    limitations.append(PreprocessLimitation(
        object_name='#pdg_nodes_with_expression', value=nr_pdg_nodes_with_expression,
        min_val=4))  # TODO: plug-in HP here.
    longest_pdg_node_expression = max(
        (len(get_pdg_node_tokenized_expression(method, pdg_node))
         for pdg_node in method_pdg.pdg_nodes
         if pdg_node.code_sub_token_range_ref is not None), default=0)
    shortest_pdg_node_expression = max(
        (len(get_pdg_node_tokenized_expression(method, pdg_node))
         for pdg_node in method_pdg.pdg_nodes
         if pdg_node.code_sub_token_range_ref is not None), default=0)
    limitations.append(PreprocessLimitation(
        object_name='|longest_pdg_node_expression|', value=longest_pdg_node_expression,
        max_val=model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression))
    limitations.append(PreprocessLimitation(
        object_name='|shortest_pdg_node_expression|', value=shortest_pdg_node_expression, min_val=1, warn=True))
    limitations.append(PreprocessLimitation(
        object_name='#symbols', value=len(method_pdg.symbols),
        min_val=model_hps.method_code_encoder.min_nr_symbols,
        max_val=model_hps.method_code_encoder.max_nr_symbols))
    nr_edges = sum((len(pdg_node.control_flow_out_edges) +
                   sum(len(edge.symbols) for edge in pdg_node.data_dependency_out_edges)
                   for pdg_node in method_pdg.pdg_nodes), start=0)
    limitations.append(PreprocessLimitation(
        object_name='#edges', value=nr_edges,
        max_val=model_hps.method_code_encoder.max_nr_pdg_edges))
    PreprocessLimitation.enforce_limitations(limitations=limitations)


def preprocess_code_task_example(
        model_hps: NDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs,
        method: SerMethod, method_pdg: SerMethodPDG, method_ast: SerMethodAST,
        remove_edges_from_pdg_nodes_idxs: Optional[Set[int]] = None,
        pdg_nodes_to_mask: Optional[Dict[int, str]] = None) -> Optional[MethodCodeInputTensors]:
    enforce_code_task_input_pp_limitations(
        model_hps=model_hps, method=method, method_pdg=method_pdg, method_ast=method_ast)
    if pdg_nodes_to_mask is None:
        pdg_nodes_to_mask = {}

    identifiers_sub_parts_set = {
        sub_part for identifier_sub_parts in method_pdg.sub_identifiers_by_idx for sub_part in identifier_sub_parts}
    identifiers_sub_parts_sorted = sorted(list(identifiers_sub_parts_set))
    identifiers_sub_parts_indexer = {
        sub_part: sub_part_idx for sub_part_idx, sub_part in enumerate(identifiers_sub_parts_sorted)}

    identifiers_sub_parts_indices = BatchedFlattenedIndicesFlattenedSeq(
        sequences=[
            torch.LongTensor([
                identifiers_sub_parts_indexer[sub_part] for sub_part in identifier_sub_parts])
            for identifier_sub_parts in method_pdg.sub_identifiers_by_idx],
        self_indexing_group='identifiers', tgt_indexing_group='identifiers_sub_parts')

    identifiers_sub_parts_vocab_word_index = BatchFlattenedSeq(
        sequences=[
            torch.LongTensor([
                code_task_vocabs.sub_identifiers.get_word_idx_or_unk(sub_part)
                for sub_part in identifier_sub_parts])
            for identifier_sub_parts in method_pdg.sub_identifiers_by_idx],
        self_indexing_group='identifiers')

    # TODO: plug HP for hasher `n_features`
    sub_identifiers_hasher = HashingVectorizer(analyzer='char', n_features=256, ngram_range=(1, 3))
    identifiers_sub_parts_hashings = BatchFlattenedSeq(
        sequences=[
            torch.stack([
                torch.tensor(sub_identifiers_hasher.transform([sub_part]).toarray(), dtype=torch.float32).squeeze()
                for sub_part in identifier_sub_parts])
            for identifier_sub_parts in method_pdg.sub_identifiers_by_idx],
        self_indexing_group='identifiers')

    sub_parts_obfuscation = BatchFlattenedPseudoRandomSamplerFromRange(
        sample_size=len(identifiers_sub_parts_indexer),
        tgt_range_start=len(code_task_vocabs.sub_identifiers.special_words),
        tgt_range_end=len(code_task_vocabs.sub_identifiers),
        initial_seed_salt='idntf', replacement='wo_replacement_within_example')

    identifiers = IdentifiersInputTensors(
        sub_parts_batch=BatchFlattenedTensor(
            tensor=torch.arange(len(identifiers_sub_parts_indexer)),
            self_indexing_group='identifiers_sub_parts'),
        identifier_sub_parts_index=identifiers_sub_parts_indices,
        identifier_sub_parts_vocab_word_index=identifiers_sub_parts_vocab_word_index,
        identifier_sub_parts_hashings=identifiers_sub_parts_hashings,
        sub_parts_obfuscation=sub_parts_obfuscation)

    # Note:
    # For pdg-nodes of control kind CATCH_CLAUSE_ENTRY / METHOD_ENTRY, their sub-AST should be trimmed.
    # The last child of the root is a of type block-stmt and should not be considered as a part of this sub-AST.
    # The tokenized expression ref of these pdg-nodes is already trimmed correctly in the raw extracted data.

    assert all(not pdg_node.has_expression or pdg_node.code_sub_token_range_ref is not None
               for pdg_node in method_pdg.pdg_nodes)
    assert all(pdg_node.control_kind in {SerPDGNodeControlKind.METHOD_ENTRY,
                                         SerPDGNodeControlKind.CATCH_CLAUSE_ENTRY}
               for pdg_node in method_pdg.pdg_nodes
               if not pdg_node.has_expression and pdg_node.code_sub_token_range_ref is not None)
    _counter = itertools.count()
    pdg_node_idx_to_expression_idx_mapping = {
        pdg_node.idx: next(_counter)
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask}
    del _counter

    # Note: If we would like the absolute token idx (within the whole method) we could just add the param
    #       `start=pdg_node.code_sub_token_range_ref.begin_token_idx` to enumerate(...)
    SymbolOccurrence = namedtuple('SymbolOccurrence', ['expression_idx', 'within_expr_token_idx', 'symbol_idx'])
    symbols_occurrences = [
        SymbolOccurrence(
            expression_idx=pdg_node_idx_to_expression_idx_mapping[pdg_node.idx],
            within_expr_token_idx=within_expr_token_idx, symbol_idx=token.symbol_idx)
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask
        for within_expr_token_idx, token in enumerate(
            get_pdg_node_tokenized_expression(method=method, pdg_node=pdg_node))
        if token.symbol_idx is not None]
    symbols_occurrences.sort(key=lambda symbol_occurrence: symbol_occurrence.symbol_idx)
    symbols = SymbolsInputTensors(
        symbols_identifier_indices=BatchedFlattenedIndicesFlattenedTensor(
            torch.LongTensor([symbol.identifier_idx for symbol in method_pdg.symbols]),
            self_indexing_group='symbols', tgt_indexing_group='identifiers'),
        symbols_appearances_symbol_idx=BatchedFlattenedIndicesFlattenedTensor(
            torch.LongTensor([symbol_occurrence.symbol_idx for symbol_occurrence in symbols_occurrences]),
            tgt_indexing_group='symbols'),
        symbols_appearances_expression_token_idx=BatchFlattenedTensor(
            torch.LongTensor([symbol_occurrence.within_expr_token_idx for symbol_occurrence in symbols_occurrences])),
        symbols_appearances_cfg_expression_idx=BatchedFlattenedIndicesFlattenedTensor(
            torch.LongTensor([symbol_occurrence.expression_idx for symbol_occurrence in symbols_occurrences]),
            tgt_indexing_group='cfg_expressions'))

    cfg_nodes_tokenized_expressions_token_type = BatchFlattenedSeq(
        [torch.LongTensor(
            [code_task_vocabs.tokens_kinds.get_word_idx(token.kind.value)
             for token in get_pdg_node_tokenized_expression(method=method, pdg_node=pdg_node)])
            for pdg_node in method_pdg.pdg_nodes
            if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask],
        self_indexing_group='cfg_expressions')
    cfg_nodes_tokenized_expressions = CodeExpressionTokensSequenceInputTensors(
        token_type=cfg_nodes_tokenized_expressions_token_type,
        kos_token_index=BatchFlattenedTensor(torch.LongTensor(
            [code_task_vocabs.kos_tokens.get_word_idx_or_unk(kos_token_to_kos_token_vocab_word(token))
             for pdg_node in method_pdg.pdg_nodes
             if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask
             for token in get_pdg_node_tokenized_expression(method=method, pdg_node=pdg_node)
             if token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR}])),
        identifier_index=BatchedFlattenedIndicesFlattenedTensor(torch.LongTensor(
            [token.identifier_idx
             for pdg_node in method_pdg.pdg_nodes
             if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask
             for token in get_pdg_node_tokenized_expression(method=method, pdg_node=pdg_node)
             if token.kind == SerTokenKind.IDENTIFIER]),
            tgt_indexing_group='identifiers'),
        symbol_index=BatchedFlattenedIndicesFlattenedTensor(torch.LongTensor(
            [token.symbol_idx
             for pdg_node in method_pdg.pdg_nodes
             if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask
             for token in get_pdg_node_tokenized_expression(method=method, pdg_node=pdg_node)
             if token.symbol_idx is not None]),
            tgt_indexing_group='symbols'),
        is_symbol_mask=BatchFlattenedSeq(
            [torch.BoolTensor(
                [token.symbol_idx is not None
                 for token in get_pdg_node_tokenized_expression(method=method, pdg_node=pdg_node)])
             for pdg_node in method_pdg.pdg_nodes
             if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask]),
        sequence_shuffler=BatchFlattenedSeqShuffler(
            lengths=tuple(len(seq) for seq in cfg_nodes_tokenized_expressions_token_type.sequences),
            initial_seed_salt='cfg_nodes_tokenized_expressions_seq_shuffler'),
        token_idx_to_ast_leaf_idx_mapping_key=None,  # TODO
        token_idx_to_ast_leaf_idx_mapping_value=None)  # TODO
    assert \
        cfg_nodes_tokenized_expressions.symbol_index.indices.size(0) == \
        sum(seq.to(torch.long).sum().item() for seq in cfg_nodes_tokenized_expressions.is_symbol_mask.sequences)

    control_flow_paths = get_all_pdg_simple_paths(
        method_pdg=method_pdg,
        src_pdg_node_idx=method_pdg.entry_pdg_node_idx, tgt_pdg_node_idx=method_pdg.exit_pdg_node_idx,
        control_flow=True, data_dependency=False,
        max_nr_paths=model_hps.method_code_encoder.max_nr_control_flow_paths)
    if control_flow_paths is None:
        PreprocessLimitation.enforce_limitations(limitations=[
            PreprocessLimitation(
                object_name='#control_flow_paths', value=float('inf'),
                max_val=model_hps.method_code_encoder.max_nr_control_flow_paths)])
    assert control_flow_paths is not None
    control_flow_paths = [
        tuple((path_node.node_idx, None if path_node.edge is None else path_node.edge.type.value) for path_node in path)
        for path in control_flow_paths]
    control_flow_paths.sort()  # for determinism

    limitations = []
    limitations.append(PreprocessLimitation(
        object_name='#control_flow_paths', value=len(control_flow_paths),
        min_val=model_hps.method_code_encoder.min_nr_control_flow_paths,
        max_val=model_hps.method_code_encoder.max_nr_control_flow_paths))
    shortest_control_flow_path = min(len(path) for path in control_flow_paths)
    longest_control_flow_path = max(len(path) for path in control_flow_paths)
    limitations.append(PreprocessLimitation(
        object_name='|shortest_control_flow_paths|', value=shortest_control_flow_path,
        min_val=model_hps.method_code_encoder.min_control_flow_path_len))
    limitations.append(PreprocessLimitation(
        object_name='|longest_control_flow_path|', value=longest_control_flow_path,
        max_val=model_hps.method_code_encoder.max_control_flow_path_len))
    PreprocessLimitation.enforce_limitations(limitations=limitations)

    control_flow_paths_ngrams = {}
    for ngrams_n in range(2, 6 + 1):  # TODO: make these HPs
        control_flow_paths_ngrams[ngrams_n] = set()
        for control_flow_path in control_flow_paths:
            for path_ngram_start_idx in range(len(control_flow_path) - ngrams_n + 1):
                ngram = control_flow_path[path_ngram_start_idx:path_ngram_start_idx + ngrams_n]
                control_flow_paths_ngrams[ngrams_n].add(ngram)
    # sort for determinism
    control_flow_paths_ngrams = {
        ngrams_n: sorted(list(ngrams))
        for ngrams_n, ngrams in control_flow_paths_ngrams.items()
        if len(ngrams) > 0}

    control_flow_paths_node_idxs_set = {node_idx for path in control_flow_paths for node_idx, _ in path}
    cfg_node_idxs_not_in_any_cfg_path = (set(node.idx for node in method_pdg.pdg_nodes) - control_flow_paths_node_idxs_set) - {0}
    # TODO: replace with assert! after we fix the <try..catch..finally> control-flow edges in the JavaExtractor.
    PreprocessLimitation.enforce_limitations(limitations=[PreprocessLimitation(
        object_name='#cfg_node_idxs_not_in_any_cfg_path', value=len(cfg_node_idxs_not_in_any_cfg_path),
        max_val=0)])

    # FOR DEBUG:
    # from ndfa.misc.example_formatter import format_example, RawExtractedExample
    # node_idxs_not_in_any_cfg_path = sorted(list(node_idxs_not_in_any_cfg_path))
    # if method.hash == '195c251e':  # 'b6c08e69':
    # if len(node_idxs_not_in_any_cfg_path) > 0 and node_idxs_not_in_any_cfg_path != [0]:
    #     print(method.hash)
    #     print(format_example(example=RawExtractedExample(method=method, method_ast=method_ast, method_pdg=method_pdg)))
    #     print(f'node_idxs not in path: {node_idxs_not_in_any_cfg_path}')
    #     # print([(pdg_node.idx, edge.pgd_node_idx) for pdg_node in method_pdg.pdg_nodes for edge in
    #     #        pdg_node.control_flow_out_edges])
    #     # print('node #5 control type',
    #     #       method_pdg.control_scopes[method_pdg.pdg_nodes[5].belongs_to_control_scopes_idxs[-1]].type)
    #     # get_all_pdg_simple_paths(
    #     #     method_pdg=method_pdg,
    #     #     src_pdg_node_idx=method_pdg.entry_pdg_node_idx, tgt_pdg_node_idx=method_pdg.exit_pdg_node_idx,
    #     #     control_flow=True, data_dependency=False,
    #     #     max_nr_paths=model_hps.method_code_encoder.max_nr_control_flow_paths,
    #     #     dbg_verbose=True)
    #     print()
    #     print()
    #     # exit()

    # assert all(
    #     pdg_node.ast_node_idx is not None for pdg_node in method_pdg.pdg_nodes
    #     if pdg_node.code_sub_token_range_ref is not None)

    assert all(
        (pdg_node.control_kind == SerPDGNodeControlKind.METHOD_ENTRY) ^ (pdg_node.ast_node_idx != 0)
        for pdg_node in method_pdg.pdg_nodes)

    nr_pdg_nodes_with_code_expression_and_wo_sub_ast = sum(
        int(pdg_node.ast_node_idx is None)
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.code_sub_token_range_ref is not None)
    last_method_decl_child_is_block_stmt = \
        method_ast.nodes[method_ast.nodes[0].children_idxs[-1]].type == SerASTNodeType.BLOCK_STMT
    nr_non_last_method_decl_children_which_are_block_stmt = sum(
        int(method_ast.nodes[child_node_idx].type == SerASTNodeType.BLOCK_STMT)
        for child_node_idx in method_ast.nodes[0].children_idxs[:-1])
    last_catch_clause_entry_child_is_block_stmt = all(
        method_ast.nodes[method_ast.nodes[pdg_node.ast_node_idx].children_idxs[-1]].type == SerASTNodeType.BLOCK_STMT
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.control_kind == SerPDGNodeControlKind.CATCH_CLAUSE_ENTRY)
    nr_non_last_catch_clause_entry_children_which_are_block_stmt = sum(
        int(method_ast.nodes[child_node_idx].type == SerASTNodeType.BLOCK_STMT)
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.control_kind == SerPDGNodeControlKind.CATCH_CLAUSE_ENTRY
        for child_node_idx in method_ast.nodes[pdg_node.ast_node_idx].children_idxs[:-1])
    PreprocessLimitation.enforce_limitations(limitations=[
        PreprocessLimitation(
            object_name='nr_pdg_nodes_with_code_expression_and_wo_sub_ast',
            value=nr_pdg_nodes_with_code_expression_and_wo_sub_ast,
            min_val=0, max_val=0),
        PreprocessLimitation(
            object_name='last_method_decl_child_is_block_stmt',
            value=int(last_method_decl_child_is_block_stmt),
            min_val=1, max_val=1),
        PreprocessLimitation(
            object_name='nr_non_last_method_decl_children_which_are_block_stmt',
            value=nr_non_last_method_decl_children_which_are_block_stmt,
            min_val=0, max_val=0),
        PreprocessLimitation(
            object_name='last_catch_clause_entry_child_is_block_stmt',
            value=int(last_catch_clause_entry_child_is_block_stmt),
            min_val=1, max_val=1),
        PreprocessLimitation(
            object_name='nr_non_last_catch_clause_entry_children_which_are_block_stmt',
            value=nr_non_last_catch_clause_entry_children_which_are_block_stmt,
            min_val=0, max_val=0)])

    pp_method_ast = False  # TODO: make it a preprocess parameter
    # TODO: ignore all subtree of log ast node (make the node itself a leaf)
    ast_node_indices_to_mask = {
        ast_node.idx
        for pdg_node_idx in pdg_nodes_to_mask
        if method_pdg.pdg_nodes[pdg_node_idx].ast_node_idx is not None
        for ast_node, _, _ in traverse_ast(
            method_ast=method_ast, root_sub_ast_node_idx=method_pdg.pdg_nodes[pdg_node_idx].ast_node_idx)}
    method_ast_paths: ASTPaths = get_all_ast_paths(
        method_ast=method_ast,
        subtrees_to_ignore=ast_node_indices_to_mask) if pp_method_ast else None
    ast_paths_per_pdg_node: Dict[int, ASTPaths] = {
        pdg_node.idx: get_all_ast_paths(
            method_ast=method_ast, sub_ast_root_node_idx=pdg_node.ast_node_idx,
            subtrees_to_ignore={method_ast.nodes[pdg_node.ast_node_idx].children_idxs[-1]} | ast_node_indices_to_mask
            if pdg_node.control_kind in {SerPDGNodeControlKind.METHOD_ENTRY, SerPDGNodeControlKind.CATCH_CLAUSE_ENTRY}
            else ast_node_indices_to_mask)
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.ast_node_idx is not None and
        pdg_node.code_sub_token_range_ref is not None and
        pdg_node.idx not in pdg_nodes_to_mask and
        pdg_node.ast_node_idx not in ast_node_indices_to_mask}

    # FOR DEBUG:
    # from ndfa.misc.code_data_structure_utils import print_ast, get_ast_node_expression_str, get_pdg_node_expression_str
    # for pdg_node_idx, paths in ast_paths_per_pdg_node.items():
    #     # if len(paths.leaf_to_leaf_paths) > 0:
    #     #     continue
    #     pdg_node = method_pdg.pdg_nodes[pdg_node_idx]
    #     root_sub_ast_node_idx = pdg_node.ast_node_idx
    #     root_sub_ast_node = method_ast.nodes[root_sub_ast_node_idx]
    #     # if not any(len(method_ast.nodes[ast_node_idx].children_idxs) > 3 for ast_node_idx in range(paths.subtree_indices_range[0], paths.subtree_indices_range[1] + 1)):
    #     #     continue
    #     print('PDG node control kind:', pdg_node.control_kind.value)
    #     # print(get_ast_node_expression_str(method=method, ast_node=root_sub_ast_node).strip())
    #     print(get_pdg_node_expression_str(method=method, pdg_node=pdg_node).strip())
    #     subtrees_to_ignore = {root_sub_ast_node.children_idxs[-1]} \
    #         if pdg_node.control_kind in {SerPDGNodeControlKind.METHOD_ENTRY, SerPDGNodeControlKind.CATCH_CLAUSE_ENTRY} else set()
    #     subtrees_to_ignore |= ast_node_indices_to_mask
    #     print_ast(method=method, method_ast=method_ast,
    #               root_sub_ast_node_idx=root_sub_ast_node_idx,
    #               subtrees_to_ignore=subtrees_to_ignore)
    #     print()

    assert all(
        method_ast.nodes[method_ast.nodes[pdg_node.ast_node_idx].children_idxs[-1]].type == SerASTNodeType.BLOCK_STMT
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.control_kind in {SerPDGNodeControlKind.METHOD_ENTRY, SerPDGNodeControlKind.CATCH_CLAUSE_ENTRY})
    pdg_node_to_ast_node_indices = {
        pdg_node_idx: set(
            range(
                ast_paths.subtree_indices_range[0],
                method_ast.nodes[method_pdg.pdg_nodes[pdg_node_idx].ast_node_idx].children_idxs[-1] - 1 + 1)
            if method_pdg.pdg_nodes[pdg_node_idx].control_kind in
               {SerPDGNodeControlKind.METHOD_ENTRY, SerPDGNodeControlKind.CATCH_CLAUSE_ENTRY} else
            range(ast_paths.subtree_indices_range[0], ast_paths.subtree_indices_range[1] + 1))
        for pdg_node_idx, ast_paths in ast_paths_per_pdg_node.items()}
    assert all(
        len(ast_node_indices & ast_node_indices_to_mask) == 0
        for ast_node_indices in pdg_node_to_ast_node_indices.values())

    # No ast node that belongs to more than one pdg nodes
    assert all(len(s1 & s2) == 0 for s1, s2 in itertools.combinations(pdg_node_to_ast_node_indices.values(), r=2))
    ast_node_idx_to_pdg_node = {
        ast_node_idx: pdg_node_idx
        for pdg_node_idx, ast_node_indices in pdg_node_to_ast_node_indices.items()
        for ast_node_idx in ast_node_indices}
    assert len(ast_paths_per_pdg_node) > 0
    assert all(len(ast_paths.postorder_traversal_sequence) > 0 for ast_paths in ast_paths_per_pdg_node.values())
    assert all(
        len(set(sub_ast_paths.leaves_sequence) & ast_node_indices_to_mask) == 0
        for sub_ast_paths in ast_paths_per_pdg_node.values())
    assert all(
        len(set(sub_ast_paths.postorder_traversal_sequence) & ast_node_indices_to_mask) == 0
        for sub_ast_paths in ast_paths_per_pdg_node.values())
    assert method_ast_paths is None or len(set(method_ast_paths.leaves_sequence) & ast_node_indices_to_mask) == 0
    assert method_ast_paths is None or len(set(method_ast_paths.postorder_traversal_sequence) & ast_node_indices_to_mask) == 0

    dgl_ast_edges = torch.LongTensor(
        [[ast_node.idx, child_node_idx]
         for ast_node in method_ast.nodes
         for child_node_idx in ast_node.children_idxs
         if ast_node.idx not in ast_node_indices_to_mask and
         child_node_idx not in ast_node_indices_to_mask]).t().chunk(chunks=2, dim=0)
    dgl_ast_edges = tuple(u.view(-1) for u in dgl_ast_edges)
    dgl_ast = dgl.graph(data=dgl_ast_edges, num_nodes=len(method_ast.nodes))
    assert all(
        len(sub_ast_nodes & ast_node_indices_to_mask) == 0
        for sub_ast_nodes in pdg_node_to_ast_node_indices.values())
    pdg_nodes_expressions_dgl_ast_edges = torch.LongTensor(
        [[ast_node_idx, child_node_idx]
         for sub_ast_nodes in pdg_node_to_ast_node_indices.values()
         for ast_node_idx in sub_ast_nodes
         for child_node_idx in method_ast.nodes[ast_node_idx].children_idxs
         if child_node_idx in sub_ast_nodes]).t().chunk(chunks=2, dim=0)
    pdg_nodes_expressions_dgl_ast_edges = tuple(u.view(-1) for u in pdg_nodes_expressions_dgl_ast_edges)
    pdg_nodes_expressions_dgl_ast = \
        dgl.graph(data=pdg_nodes_expressions_dgl_ast_edges, num_nodes=len(method_ast.nodes))

    ast = MethodASTInputTensors(
        ast_node_types=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_types.get_word_idx_or_unk(
                    '<PAD>' if ast_node.idx in ast_node_indices_to_mask else ast_node.type.value)  # TODO: should it be a special '<MASK>' vocab word instead of a simple padding?
                for ast_node in method_ast.nodes]),
            self_indexing_group='ast_nodes'),
        ast_node_major_types=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_major_types.get_word_idx_or_unk(
                    '<PAD>' if ast_node.idx in ast_node_indices_to_mask else ast_node.type.value.split('_')[0])
                # TODO: should it be a special '<MASK>' vocab word instead of a simple padding?
                for ast_node in method_ast.nodes])),
        ast_node_minor_types=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_minor_types.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_node_indices_to_mask or '_' not in ast_node.type.value else
                    ast_node.type.value[ast_node.type.value.find('_') + 1:])
                # TODO: should it be a special '<MASK>' vocab word instead of a simple padding?
                for ast_node in method_ast.nodes])),
        ast_node_child_ltr_position=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_child_pos.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_node_indices_to_mask else
                    '<+ROOT>'
                    if ast_node.parent_node_idx is None or ast_node.child_place_at_parent is None else
                    f'+{ast_node.child_place_at_parent + 1}', unk_word='<+MORE>')
                for ast_node in method_ast.nodes])),
        ast_node_child_rtl_position=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_child_pos.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_node_indices_to_mask else
                    '<-ROOT>'
                    if ast_node.parent_node_idx is None or ast_node.child_place_at_parent is None else
                    f'-{len(method_ast.nodes[ast_node.parent_node_idx].children_idxs) - ast_node.child_place_at_parent}',
                    unk_word='<-MORE>')
                for ast_node in method_ast.nodes])),
        ast_node_nr_children=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_nr_children.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_node_indices_to_mask else
                    f'{len(ast_node.children_idxs)}', unk_word='<MORE>')
                for ast_node in method_ast.nodes])),

        ast_nodes_with_identifier_leaf_nodes_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                ast_node.idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                ast_node.code_sub_token_range_ref is not None and
                ast_node.code_sub_token_range_ref.begin_token_idx == ast_node.code_sub_token_range_ref.end_token_idx and
                method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].identifier_idx is not None and
                ast_node.idx not in ast_node_indices_to_mask]),
            tgt_indexing_group='ast_nodes'),
        ast_nodes_with_identifier_leaf_identifier_idx=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].identifier_idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                ast_node.code_sub_token_range_ref is not None and
                ast_node.code_sub_token_range_ref.begin_token_idx == ast_node.code_sub_token_range_ref.end_token_idx and
                method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].identifier_idx is not None and
                ast_node.idx not in ast_node_indices_to_mask]),
            tgt_indexing_group='identifiers'),

        ast_nodes_with_symbol_leaf_nodes_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                ast_node.idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                ast_node.code_sub_token_range_ref is not None and
                ast_node.code_sub_token_range_ref.begin_token_idx == ast_node.code_sub_token_range_ref.end_token_idx and
                method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].symbol_idx is not None and
                ast_node.idx not in ast_node_indices_to_mask]),
            tgt_indexing_group='ast_nodes'),
        ast_nodes_with_symbol_leaf_symbol_idx=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].symbol_idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                ast_node.code_sub_token_range_ref is not None and
                ast_node.code_sub_token_range_ref.begin_token_idx == ast_node.code_sub_token_range_ref.end_token_idx and
                method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].symbol_idx is not None and
                ast_node.idx not in ast_node_indices_to_mask]),
            tgt_indexing_group='symbols'),

        ast_nodes_with_primitive_type_leaf_nodes_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                ast_node.idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                ast_node.type == SerASTNodeType.PRIMITIVE_TYPE and
                ast_node.type_name is not None and
                ast_node.idx not in ast_node_indices_to_mask]),
            tgt_indexing_group='ast_nodes'),
        ast_nodes_with_primitive_type_leaf_primitive_type=BatchFlattenedTensor(tensor=torch.LongTensor([
            code_task_vocabs.primitive_types.get_word_idx_or_unk(ast_node.type_name)
            for ast_node in method_ast.nodes
            if len(ast_node.children_idxs) == 0 and
            ast_node.type == SerASTNodeType.PRIMITIVE_TYPE and
            ast_node.type_name is not None and
            ast_node.idx not in ast_node_indices_to_mask])),

        ast_nodes_with_modifier_leaf_nodes_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                ast_node.idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and ast_node.modifier is not None and
                ast_node.idx not in ast_node_indices_to_mask]),
            tgt_indexing_group='ast_nodes'),
        ast_nodes_with_modifier_leaf_modifier=BatchFlattenedTensor(tensor=torch.LongTensor([
            code_task_vocabs.modifiers.get_word_idx_or_unk(ast_node.modifier)
            for ast_node in method_ast.nodes
            if len(ast_node.children_idxs) == 0 and ast_node.modifier is not None and
            ast_node.idx not in ast_node_indices_to_mask])),

        ast_leaf_to_leaf_paths_node_indices=None if not pp_method_ast else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor([path_node.ast_node_idx for path_node in path])
                       for path in method_ast_paths.leaf_to_leaf_paths.values()],
            tgt_indexing_group='ast_nodes'),
        ast_leaf_to_leaf_paths_child_place=None if not pp_method_ast else BatchFlattenedSeq(
            sequences=[
                torch.LongTensor([
                    code_task_vocabs.ast_traversal_orientation.get_word_idx(
                        'child_place=UNK' if path_node.child_place_in_parent is None else
                        f'child_place={min(path_node.child_place_in_parent, 4 - 1)}')
                    for path_node in path])
                for path in method_ast_paths.leaf_to_leaf_paths.values()]),
        ast_leaf_to_leaf_paths_vertical_direction=None if not pp_method_ast else BatchFlattenedSeq(
            sequences=[
                torch.LongTensor([
                    code_task_vocabs.ast_traversal_orientation.get_word_idx(
                        f'DIR={path_node.direction.value}')
                    for path_node in path])
                for path in method_ast_paths.leaf_to_leaf_paths.values()]),
        ast_leaf_to_root_paths_node_indices=None if not pp_method_ast else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor([path_node.ast_node_idx for path_node in path])
                       for path in method_ast_paths.leaf_to_root_paths.values()],
            tgt_indexing_group='ast_nodes'),
        ast_leaf_to_root_paths_child_place=None if not pp_method_ast else BatchFlattenedSeq(
            sequences=[
                torch.LongTensor([
                    code_task_vocabs.ast_traversal_orientation.get_word_idx(
                        'child_place=UNK' if path_node.child_place_in_parent is None else
                        f'child_place={min(path_node.child_place_in_parent, 4 - 1)}')
                    for path_node in path])
                for path in method_ast_paths.leaf_to_root_paths.values()]),
        ast_leaves_sequence_node_indices=None if not pp_method_ast else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor(method_ast_paths.leaves_sequence)],
            tgt_indexing_group='ast_nodes'),
        siblings_sequences_node_indices=None if not pp_method_ast else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[
                torch.LongTensor(siblings_sequence)
                for siblings_sequence in method_ast_paths.siblings_sequences],
            tgt_indexing_group='ast_nodes'),
        dgl_tree=dgl_ast)

    assert all(len(s1 & s2) == 0 for s1, s2 in itertools.combinations([
        set(ast.ast_nodes_with_identifier_leaf_nodes_indices.indices.tolist()),
        set(ast.ast_nodes_with_primitive_type_leaf_nodes_indices.indices.tolist()),
        set(ast.ast_nodes_with_modifier_leaf_nodes_indices.indices.tolist())], r=2))

    cfg_nodes_expressions_ast = PDGExpressionsSubASTInputTensors(
        ast_leaf_to_leaf_paths_pdg_node_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                pdg_node_idx
                for pdg_node_idx, sub_ast_paths in ast_paths_per_pdg_node.items()
                for _ in sub_ast_paths.leaf_to_leaf_paths.values()]),
            tgt_indexing_group='cfg_nodes'),
        ast_leaf_to_leaf_paths_node_indices=BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor([path_node.ast_node_idx for path_node in path])
                       for sub_ast_paths in ast_paths_per_pdg_node.values()
                       for path in sub_ast_paths.leaf_to_leaf_paths.values()],
            tgt_indexing_group='ast_nodes'),
        ast_leaf_to_leaf_paths_child_place=BatchFlattenedSeq(
            sequences=[
                torch.LongTensor([
                    code_task_vocabs.ast_traversal_orientation.get_word_idx(
                        'child_place=UNK' if path_node.child_place_in_parent is None else
                        f'child_place={min(path_node.child_place_in_parent, 4 - 1)}')
                    for path_node in path])
                for sub_ast_paths in ast_paths_per_pdg_node.values()
                for path in sub_ast_paths.leaf_to_leaf_paths.values()]),
        ast_leaf_to_leaf_paths_vertical_direction=BatchFlattenedSeq(
            sequences=[
                torch.LongTensor([
                    code_task_vocabs.ast_traversal_orientation.get_word_idx(
                        f'DIR={path_node.direction.value}')
                    for path_node in path])
                for sub_ast_paths in ast_paths_per_pdg_node.values()
                for path in sub_ast_paths.leaf_to_leaf_paths.values()]),
        ast_leaf_to_root_paths_pdg_node_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                pdg_node_idx
                for pdg_node_idx, sub_ast_paths in ast_paths_per_pdg_node.items()
                for _ in sub_ast_paths.leaf_to_root_paths.values()]),
            tgt_indexing_group='cfg_nodes'),
        ast_leaf_to_root_paths_node_indices=BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor([path_node.ast_node_idx for path_node in path])
                       for sub_ast_paths in ast_paths_per_pdg_node.values()
                       for path in sub_ast_paths.leaf_to_root_paths.values()],
            tgt_indexing_group='ast_nodes'),
        ast_leaf_to_root_paths_child_place=BatchFlattenedSeq(
            sequences=[
                torch.LongTensor([
                    code_task_vocabs.ast_traversal_orientation.get_word_idx(
                        'child_place=UNK' if path_node.child_place_in_parent is None else
                        f'child_place={min(path_node.child_place_in_parent, 4 - 1)}')
                    for path_node in path])
                for sub_ast_paths in ast_paths_per_pdg_node.values()
                for path in sub_ast_paths.leaf_to_root_paths.values()]),
        ast_leaves_sequence_node_indices=BatchedFlattenedIndicesFlattenedSeq(
            sequences=[
                torch.LongTensor(sub_ast_paths.leaves_sequence)
                for sub_ast_paths in ast_paths_per_pdg_node.values()],
            tgt_indexing_group='ast_nodes'),
        siblings_sequences_node_indices=None if not pp_method_ast else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[
                torch.LongTensor(siblings_sequence)
                for sub_ast_paths in ast_paths_per_pdg_node.values()
                for siblings_sequence in sub_ast_paths.siblings_sequences],
            tgt_indexing_group='ast_nodes'),
        pdg_node_idx_to_sub_ast_root_idx_mapping_key=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                pdg_node.idx
                for pdg_node in method_pdg.pdg_nodes
                if pdg_node.code_sub_token_range_ref and pdg_node.idx not in pdg_nodes_to_mask
                and pdg_node.ast_node_idx not in ast_node_indices_to_mask]),
            tgt_indexing_group='cfg_nodes'),
        pdg_node_idx_to_sub_ast_root_idx_mapping_value=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                pdg_node.ast_node_idx
                for pdg_node in method_pdg.pdg_nodes
                if pdg_node.code_sub_token_range_ref and pdg_node.idx not in pdg_nodes_to_mask
                and pdg_node.ast_node_idx not in ast_node_indices_to_mask]),
            tgt_indexing_group='ast_nodes'),
        ast_node_idx_to_pdg_node_idx_mapping_key=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor(list(ast_node_idx_to_pdg_node.keys())),
            tgt_indexing_group='ast_nodes'),
        ast_node_idx_to_pdg_node_idx_mapping_value=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor(list(ast_node_idx_to_pdg_node.values())),
            tgt_indexing_group='cfg_nodes'),
        dgl_tree=pdg_nodes_expressions_dgl_ast)

    cfg_control_flow_graph = TGData(
        edge_index=torch.LongTensor(
            [[pdg_node.idx, cf_edge.pgd_node_idx]
             for pdg_node in method_pdg.pdg_nodes
             for cf_edge in pdg_node.control_flow_out_edges]).transpose(0, 1),
        edge_attr=torch.LongTensor(
            [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx(cf_edge.type.value)
             for pdg_node in method_pdg.pdg_nodes
             for cf_edge in pdg_node.control_flow_out_edges]),
        num_nodes=len(method_pdg.pdg_nodes))
    assert not cfg_control_flow_graph.contains_isolated_nodes()
    assert cfg_control_flow_graph.is_directed()
    assert (set(cfg_control_flow_graph.edge_index[0].tolist()) |
            set(cfg_control_flow_graph.edge_index[1].tolist())) == set(range(len(method_pdg.pdg_nodes)))

    pdg = PDGInputTensors(
        cfg_nodes_control_kind=BatchFlattenedTensor(torch.LongTensor(
            [code_task_vocabs.pdg_node_control_kinds.get_word_idx(
                pdg_node.control_kind.value
                if pdg_node.idx not in pdg_nodes_to_mask else
                pdg_nodes_to_mask[pdg_node.idx])
             for pdg_node in method_pdg.pdg_nodes]), self_indexing_group='cfg_nodes'),
        cfg_nodes_has_expression_mask=BatchFlattenedTensor(torch.BoolTensor(
            [pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask
             for pdg_node in method_pdg.pdg_nodes])),
        cfg_nodes_tokenized_expressions=cfg_nodes_tokenized_expressions,
        cfg_nodes_expressions_ast=cfg_nodes_expressions_ast,
        cfg_nodes_random_permutation=BatchedFlattenedIndicesPseudoRandomPermutation(
            tgt_indexing_group='cfg_nodes',
            batch_dependent_seed=True, example_dependent_seed=True, initial_seed_salt='cfgn'),
        cfg_control_flow_paths=CFGPathsInputTensors(
            nodes_indices=BatchedFlattenedIndicesFlattenedSeq(
                sequences=[torch.LongTensor([node_idx for node_idx, _ in path]) for path in control_flow_paths],
                tgt_indexing_group='cfg_nodes'),
            edges_types=BatchFlattenedSeq(
                sequences=[torch.LongTensor(
                    [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx(
                        '<PAD>' if edge_type is None else edge_type)
                     for _, edge_type in path])
                    for path in control_flow_paths])),
        cfg_control_flow_paths_ngrams=TensorsDataDict({
            key: CFGPathsNGramsInputTensors(
                nodes_indices=BatchedFlattenedIndicesFlattenedSeq(
                    sequences=[torch.LongTensor([node_idx for node_idx, _ in ngram]) for ngram in ngrams],
                    tgt_indexing_group='cfg_nodes'),
                edges_types=BatchFlattenedSeq(
                    sequences=[torch.LongTensor(
                        [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx(
                            '<PAD>' if edge_type is None else edge_type)
                            for _, edge_type in ngram])
                        for ngram in ngrams]))
            for key, ngrams in control_flow_paths_ngrams.items()}),
        cfg_control_flow_graph=cfg_control_flow_graph)

    token_ranges_to_mask = [
        (method_pdg.pdg_nodes[pdg_node_idx].code_sub_token_range_ref, mask_replacement)
        for pdg_node_idx, mask_replacement in pdg_nodes_to_mask.items()
        if method_pdg.pdg_nodes[pdg_node_idx].code_sub_token_range_ref is not None]
    token_indices_mask = {
        token_range_to_mask.begin_token_idx: mask_replacement
        for token_range_to_mask, mask_replacement in token_ranges_to_mask}
    token_indices_to_ignore = {
        token_idx for token_idx in range(len(method.code.tokenized))
        if any(token_range_to_mask.begin_token_idx < token_idx <= token_range_to_mask.end_token_idx
               for token_range_to_mask, _ in token_ranges_to_mask)}
    token_indices_to_mask_or_ignore = {
        token_idx for token_idx in range(len(method.code.tokenized))
        if any(token_range_to_mask.begin_token_idx <= token_idx <= token_range_to_mask.end_token_idx
               for token_range_to_mask, _ in token_ranges_to_mask)}
    method_tokenized_code_token_type = BatchFlattenedSeq([torch.LongTensor([
        code_task_vocabs.tokens_kinds.get_word_idx(token_indices_mask.get(token_idx, token.kind.value))
        for token_idx, token in enumerate(method.code.tokenized)
        if token_idx not in token_indices_to_ignore])], self_indexing_group='method_tokenized_expr_seq')
    method_tokenized_code = CodeExpressionTokensSequenceInputTensors(
        token_type=method_tokenized_code_token_type,
        kos_token_index=BatchFlattenedTensor(torch.LongTensor(
            [code_task_vocabs.kos_tokens.get_word_idx_or_unk(kos_token_to_kos_token_vocab_word(token))
             for token_idx, token in enumerate(method.code.tokenized)
             if token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR}
             if token_idx not in token_indices_to_mask_or_ignore])),
        identifier_index=BatchedFlattenedIndicesFlattenedTensor(torch.LongTensor(
            [token.identifier_idx
             for token_idx, token in enumerate(method.code.tokenized)
             if token.kind == SerTokenKind.IDENTIFIER
             if token_idx not in token_indices_to_mask_or_ignore]),
            tgt_indexing_group='identifiers'),
        symbol_index=BatchedFlattenedIndicesFlattenedTensor(torch.LongTensor(
            [token.symbol_idx
             for token_idx, token in enumerate(method.code.tokenized)
             if token.symbol_idx is not None
             if token_idx not in token_indices_to_mask_or_ignore]),
            tgt_indexing_group='symbols'),
        is_symbol_mask=BatchFlattenedSeq([torch.BoolTensor(
            [token.symbol_idx is not None and token_idx not in token_indices_mask
             for token_idx, token in enumerate(method.code.tokenized)
             if token_idx not in token_indices_to_ignore])]),
        sequence_shuffler=BatchFlattenedSeqShuffler(
            lengths=tuple(len(seq) for seq in method_tokenized_code_token_type.sequences),
            initial_seed_salt='method_tokenized_code_seq_permuter'),
        token_idx_to_ast_leaf_idx_mapping_key=None,  # TODO
        token_idx_to_ast_leaf_idx_mapping_value=None)  # TODO
    assert method_tokenized_code.symbol_index.indices.size(0) == \
           method_tokenized_code.is_symbol_mask.sequences[0].to(torch.long).sum().item()

    return MethodCodeInputTensors(
        method_hash=method.hash, identifiers=identifiers, symbols=symbols,
        method_tokenized_code=method_tokenized_code, pdg=pdg, ast=ast)


def preprocess_code_task_example_with_padding(
        model_hps: NDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs,
        method: SerMethod, method_pdg: SerMethodPDG, method_ast: SerMethodAST,
        remove_edges_from_pdg_nodes_idxs: Optional[Set[int]] = None,
        pdg_nodes_to_mask: Optional[Dict[int, str]] = None) -> Optional[MethodCodeInputPaddedTensors]:
    enforce_code_task_input_pp_limitations(
        model_hps=model_hps, method=method, method_pdg=method_pdg, method_ast=method_ast)

    nr_symbols = len(method_pdg.symbols)
    nr_edges = sum((len(pdg_node.control_flow_out_edges) +
                    sum(len(edge.symbols) for edge in pdg_node.data_dependency_out_edges)
                    for pdg_node in method_pdg.pdg_nodes), start=0)
    sub_identifiers_pad = [code_task_vocabs.sub_identifiers.get_word_idx_or_unk(
        '<PAD>')] * model_hps.method_code_encoder.max_nr_identifier_sub_parts
    identifiers = torch.tensor(
        [[code_task_vocabs.sub_identifiers.get_word_idx_or_unk(sub_identifier_str)
          for sub_identifier_str in truncate_and_pad(sub_identifiers, model_hps.method_code_encoder.max_nr_identifier_sub_parts)]
         for sub_identifiers in itertools.islice(method_pdg.sub_identifiers_by_idx, model_hps.method_code_encoder.max_nr_identifiers)] +
        [sub_identifiers_pad
         for _ in range(model_hps.method_code_encoder.max_nr_identifiers - min(len(method_pdg.sub_identifiers_by_idx), model_hps.method_code_encoder.max_nr_identifiers))],
        dtype=torch.long)
    sub_identifiers_mask = torch.tensor(
        [[1] * min(len(sub_identifiers), model_hps.method_code_encoder.max_nr_identifier_sub_parts) +
         [0] * (model_hps.method_code_encoder.max_nr_identifier_sub_parts - min(len(sub_identifiers), model_hps.method_code_encoder.max_nr_identifier_sub_parts))
         for sub_identifiers in itertools.islice(method_pdg.sub_identifiers_by_idx, model_hps.method_code_encoder.max_nr_identifiers)] +
        [[1] * model_hps.method_code_encoder.max_nr_identifier_sub_parts] *
        (model_hps.method_code_encoder.max_nr_identifiers - min(len(method_pdg.sub_identifiers_by_idx), model_hps.method_code_encoder.max_nr_identifiers)),
        dtype=torch.bool)
    symbols_identifier_idxs = torch.tensor(
        [symbol.identifier_idx for symbol in method_pdg.symbols] +
        ([0] * (model_hps.method_code_encoder.max_nr_symbols - len(method_pdg.symbols))), dtype=torch.long)
    symbols_identifier_mask = torch.cat([
        torch.ones(nr_symbols, dtype=torch.bool),
        torch.zeros(model_hps.method_code_encoder.max_nr_symbols - nr_symbols, dtype=torch.bool)])
    cfg_nodes_mask = torch.cat([
        torch.ones(len(method_pdg.pdg_nodes), dtype=torch.bool),
        torch.zeros(model_hps.method_code_encoder.max_nr_pdg_nodes - len(method_pdg.pdg_nodes), dtype=torch.bool)])
    cfg_nodes_control_kind = torch.tensor(list(truncate_and_pad([
        code_task_vocabs.pdg_node_control_kinds.get_word_idx_or_unk(
            pdg_node.control_kind.value
            if pdg_nodes_to_mask is None or pdg_node.idx not in pdg_nodes_to_mask else
            pdg_nodes_to_mask[pdg_node.idx])
        for pdg_node in method_pdg.pdg_nodes], max_length=model_hps.method_code_encoder.max_nr_pdg_nodes,
        pad_word=code_task_vocabs.pdg_node_control_kinds.get_word_idx('<PAD>'))), dtype=torch.long)
    padding_expression = [[code_task_vocabs.tokens_kinds.get_word_idx('<PAD>'),
                           code_task_vocabs.kos_tokens.get_word_idx('<PAD>'),
                           -1]] * (model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1)
    cfg_nodes_expressions = torch.tensor(list(truncate_and_pad([
        list(truncate_and_pad(
            [token_to_input_vector(token, code_task_vocabs) for token in get_pdg_node_tokenized_expression(method, pdg_node)],
            max_length=model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1, pad_word=padding_expression[0]))
        if pdg_node.code_sub_token_range_ref is not None and (pdg_nodes_to_mask is None or pdg_node.idx not in pdg_nodes_to_mask) else
        padding_expression
        for pdg_node in method_pdg.pdg_nodes],
        max_length=model_hps.method_code_encoder.max_nr_pdg_nodes, pad_word=padding_expression)), dtype=torch.long)
    cfg_nodes_expressions_mask = torch.tensor(
        [([1] * min(len(get_pdg_node_tokenized_expression(method, pdg_node)), (model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1)) +
         [0] * ((model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1) - min(len(get_pdg_node_tokenized_expression(method, pdg_node)), (model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1))))
         if pdg_node.code_sub_token_range_ref is not None and (pdg_nodes_to_mask is None or pdg_node.idx not in pdg_nodes_to_mask) else
         [1] * (model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1)
         for pdg_node in itertools.islice(method_pdg.pdg_nodes, model_hps.method_code_encoder.max_nr_pdg_nodes)] +
        [[1] * (model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1)] *
        (model_hps.method_code_encoder.max_nr_pdg_nodes - min(len(method_pdg.pdg_nodes), model_hps.method_code_encoder.max_nr_pdg_nodes)),
        dtype=torch.bool)

    indices_of_symbols_occurrences_in_cfg_nodes_expressions = torch.nonzero(cfg_nodes_expressions[:, :, 2] >= 0, as_tuple=False)
    symbols_idxs_of_symbols_occurrences_in_cfg_nodes_expressions = (cfg_nodes_expressions[:, :, 2].flatten(0, 1))[
        indices_of_symbols_occurrences_in_cfg_nodes_expressions[:, 0] * cfg_nodes_expressions.size(1) +
        indices_of_symbols_occurrences_in_cfg_nodes_expressions[:, 1]]

    cfg_edges_mask = torch.cat([
        torch.ones(nr_edges, dtype=torch.bool),
        torch.zeros(model_hps.method_code_encoder.max_nr_pdg_edges - nr_edges, dtype=torch.bool)])
    control_flow_edges = {}
    data_dependency_edges = defaultdict(set)
    for src_pdg_node in method_pdg.pdg_nodes:
        for edge in src_pdg_node.control_flow_out_edges:
            control_flow_edges[(src_pdg_node.idx, edge.pgd_node_idx)] = edge
    for src_pdg_node in method_pdg.pdg_nodes:
        for edge in src_pdg_node.data_dependency_out_edges:
            if remove_edges_from_pdg_nodes_idxs is not None and \
                    (src_pdg_node.idx in remove_edges_from_pdg_nodes_idxs or
                     edge.pgd_node_idx in remove_edges_from_pdg_nodes_idxs):
                continue
            data_dependency_edges[(src_pdg_node.idx, edge.pgd_node_idx)].update(
                symbol.identifier_idx for symbol in edge.symbols)
    edges = list(set(control_flow_edges.keys()) | set(data_dependency_edges.keys()))
    cfg_edges = torch.tensor(list(truncate_and_pad([
        [src_pdg_node_idx, dst_pdg_node_idx]
        for src_pdg_node_idx, dst_pdg_node_idx in edges],
        max_length=model_hps.method_code_encoder.max_nr_pdg_edges, pad_word=[-1, -1])), dtype=torch.long)

    pad_edge_attrs_vector = [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx('<PAD>')] + \
                            ([-1] * model_hps.method_code_encoder.max_nr_pdg_data_dependency_edges_between_two_nodes)
    def build_edge_attrs_vector(edge_vertices) -> List[int]:
        control_flow_edge_attrs = [
            code_task_vocabs.pdg_control_flow_edge_types.get_word_idx_or_unk(
                control_flow_edges[edge_vertices].type.value)] \
            if edge_vertices in control_flow_edges else \
            [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx('<UNK>')]
        data_dependency_edge_attrs = list(truncate_and_pad(
            data_dependency_edges[edge_vertices], model_hps.method_code_encoder.max_nr_pdg_data_dependency_edges_between_two_nodes, -1)) \
            if edge_vertices in data_dependency_edges else \
            ([-1] * model_hps.method_code_encoder.max_nr_pdg_data_dependency_edges_between_two_nodes)
        return control_flow_edge_attrs + data_dependency_edge_attrs

    cfg_edges_attrs = torch.tensor(list(truncate_and_pad([
        build_edge_attrs_vector(edge_vertices)
        for edge_vertices in edges], max_length=model_hps.method_code_encoder.max_nr_pdg_edges, pad_word=pad_edge_attrs_vector)), dtype=torch.long)

    return MethodCodeInputPaddedTensors(
        method_hash=method.hash,
        identifiers=identifiers,
        sub_identifiers_mask=sub_identifiers_mask,
        cfg_nodes_mask=cfg_nodes_mask,
        cfg_nodes_control_kind=cfg_nodes_control_kind,
        cfg_nodes_expressions=cfg_nodes_expressions,
        cfg_nodes_expressions_mask=cfg_nodes_expressions_mask,
        cfg_edges=cfg_edges,
        cfg_edges_mask=cfg_edges_mask,
        cfg_edges_attrs=cfg_edges_attrs,
        identifiers_idxs_of_all_symbols=symbols_identifier_idxs,
        identifiers_idxs_of_all_symbols_mask=symbols_identifier_mask,
        indices_of_symbols_occurrences_in_cfg_nodes_expressions=TensorWithCollateMask(
            tensor=indices_of_symbols_occurrences_in_cfg_nodes_expressions),
        symbols_idxs_of_symbols_occurrences_in_cfg_nodes_expressions=TensorWithCollateMask(
            tensor=symbols_idxs_of_symbols_occurrences_in_cfg_nodes_expressions)
    )


class PPExampleFnType(Protocol):
    def __call__(self, model_hps: NDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs, raw_example: Any) -> Any: ...


class RawExtractedExamplesGenerator(Protocol):
    def __call__(self, raw_extracted_data_dir: str) -> Iterable[Any]: ...


def catch_preprocess_limit_exceed_error(
        pp_example_fn: PPExampleFnType, model_hps: NDFAModelHyperParams,
        code_task_vocabs: CodeTaskVocabs, raw_example):
    try:
        pp_example = pp_example_fn(model_hps=model_hps, code_task_vocabs=code_task_vocabs, raw_example=raw_example)
        assert pp_example is not None
        with io.BytesIO() as bytes_io_stream:
            torch.save(pp_example, bytes_io_stream)
            binary_serialized_pp_example = bytes_io_stream.getvalue()
            return binary_serialized_pp_example
    except PreprocessLimitExceedError as err:
        return err.exceeding_limitations


def preprocess_code_task_dataset(
        model_hps: NDFAModelHyperParams, pp_data_path: str,
        raw_extracted_examples_generator: RawExtractedExamplesGenerator, pp_example_fn: PPExampleFnType,
        code_task_vocabs: CodeTaskVocabs, raw_train_data_path: Optional[str] = None,
        raw_validation_data_path: Optional[str] = None, raw_test_data_path: Optional[str] = None,
        nr_processes: int = 4, pp_override: bool = False):
    datafolds = (
        (DataFold.Train, raw_train_data_path),
        (DataFold.Validation, raw_validation_data_path),
        (DataFold.Test, raw_test_data_path))
    for datafold, raw_dataset_path in datafolds:
        if raw_dataset_path is None:
            continue
        print(f'Starting pre-processing data-fold: `{datafold.name}` ..')
        # TODO: add hash of task props & model HPs to perprocessed file name.
        # TODO: aggregate limit exceed statistics and print in the end.
        chunks_examples_writer = ChunkedRandomAccessDatasetWriter(
            pp_data_path_prefix=os.path.join(pp_data_path, f'pp_{datafold.value.lower()}'),
            max_chunk_size_in_bytes=ChunkedRandomAccessDatasetWriter.MB_IN_BYTES * 500,
            override=pp_override)
        with mp.Pool(processes=nr_processes) as pool:
            # TODO: `imap_unordered` output order is not well-defined. add option to use `imap` for reproducibility.
            for pp_example in pool.imap_unordered(
                    functools.partial(
                        catch_preprocess_limit_exceed_error,
                        pp_example_fn, model_hps, code_task_vocabs),
                    iterable=raw_extracted_examples_generator(raw_extracted_data_dir=raw_dataset_path)):
                assert pp_example is not None
                if isinstance(pp_example, list):
                    pass  # TODO: add to limit exceed statistics
                else:
                    with io.BytesIO(pp_example) as bytes_io_stream:
                        bytes_io_stream.seek(0)
                        pp_example = torch.load(bytes_io_stream)
                    chunks_examples_writer.write_example(pp_example)

        chunks_examples_writer.close_last_written_chunk()
        chunks_examples_writer.enforce_no_further_chunks()
        print(f'Finished pre-processing data-fold: `{datafold.name}`.')
