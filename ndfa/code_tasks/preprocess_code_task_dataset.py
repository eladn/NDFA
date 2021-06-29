import os
import io
import torch
import random
import itertools
import functools
import dataclasses
import multiprocessing as mp
from collections import defaultdict, namedtuple
from warnings import warn
from typing import Iterable, Collection, Any, Set, Optional, Dict, List, Union, Sequence, TypeVar
from typing_extensions import Protocol
from sklearn.feature_extraction.text import HashingVectorizer
from torch_geometric.data import Data as TGData
import dgl

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams
from ndfa.nn_utils.model_wrapper.dataset_properties import DataFold
from ndfa.misc.code_data_structure_api import SerMethod, SerMethodPDG, SerMethodAST, SerToken, SerTokenKind, \
    SerASTNodeType, SerPDGNodeControlKind, SerPDGControlFlowEdge, SerPDGDataDependencyEdge
from ndfa.misc.code_data_structure_utils import get_pdg_node_tokenized_expression, get_all_pdg_simple_paths, \
    get_all_ast_paths, ASTPaths, traverse_ast
from ndfa.nn_utils.model_wrapper.chunked_random_access_dataset import ChunkedRandomAccessDatasetWriter
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs, kos_token_to_kos_token_vocab_word
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors, SymbolsInputTensors, PDGInputTensors, \
    CFGPathsInputTensors, CFGPathsNGramsInputTensors, IdentifiersInputTensors, MethodASTInputTensors, \
    PDGExpressionsSubASTInputTensors, MethodCodeTokensSequenceInputTensors, CFGCodeExpressionTokensSequenceInputTensors
from ndfa.misc.tensors_data_class import TensorsDataClass, BatchFlattenedTensor, BatchFlattenedSeq, \
    BatchedFlattenedIndicesFlattenedTensor, BatchedFlattenedIndicesFlattenedSeq, BatchFlattenedSeqShuffler, \
    BatchedFlattenedIndicesPseudoRandomPermutation, BatchFlattenedPseudoRandomSamplerFromRange, TensorsDataDict

__all__ = [
    'preprocess_code_task_dataset', 'preprocess_code_task_example', 'truncate_and_pad', 'PreprocessLimitExceedError',
    'PreprocessLimitation']


# just for debugging purposes..
def _dbg_get_size_of_pickled_obj(a):
    with io.BytesIO() as out:
        torch.save(a, out)
        out.seek(0)
        return len(out.getvalue())


NGrams = namedtuple('NGrams', ['exact_ngrams', 'partial_ngrams'])
T = TypeVar('T')


def extract_ngrams_from_sequence(
        sequences: Collection[Sequence[T]], ngrams_min_n: int = 2, ngrams_max_n: int = 6) -> NGrams:
    exact_ngrams = defaultdict(set)
    partial_ngrams = defaultdict(set)
    for sequence in sequences:
        if len(sequence) < ngrams_min_n:
            continue
        if ngrams_min_n <= len(sequence) <= ngrams_max_n:
            exact_ngrams[len(sequence)].add(sequence)
        for ngrams_n in range(ngrams_min_n, min(ngrams_max_n, len(sequence) - 1) + 1):
            for path_ngram_start_idx in range(len(sequence) - ngrams_n + 1):
                ngram = sequence[path_ngram_start_idx: path_ngram_start_idx + ngrams_n]
                partial_ngrams[ngrams_n].add(ngram)
    # sort for determinism, and remove empty
    exact_ngrams = {
        ngrams_n: sorted(list(ngrams))
        for ngrams_n, ngrams in exact_ngrams.items()
        if len(ngrams) > 0}
    partial_ngrams = {
        ngrams_n: sorted(list(ngrams))
        for ngrams_n, ngrams in partial_ngrams.items()
        if len(ngrams) > 0}
    return NGrams(exact_ngrams=exact_ngrams, partial_ngrams=partial_ngrams)


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
    # Note: We filter-out examples with switch-stmts. We currently don't extract them well (it's CFG).
    # TODO: Remove this limitation after it is fixed in the JavaExtractor.
    limitations.append(PreprocessLimitation(
        object_name='is_there_switch_stmt',
        value=int(any(
            ast_node.type in {SerASTNodeType.SWITCH_STMT,
                              SerASTNodeType.SWITCH_ENTRY,
                              SerASTNodeType.SWITCH_ENTRY_STMT}
            for ast_node in method_ast.nodes)),
        max_val=0))
    PreprocessLimitation.enforce_limitations(limitations=limitations)


def preprocess_identifiers(method_pdg: SerMethodPDG, code_task_vocabs: CodeTaskVocabs) -> IdentifiersInputTensors:
    identifiers_sub_parts_set = {
        sub_part for identifier_sub_parts in method_pdg.sub_identifiers_by_idx for sub_part in identifier_sub_parts}
    identifiers_sub_parts_sorted = sorted(list(identifiers_sub_parts_set))
    identifiers_sub_parts_indexer = {
        sub_part: sub_part_idx for sub_part_idx, sub_part in enumerate(identifiers_sub_parts_sorted)}

    identifiers_sub_parts_indices = BatchedFlattenedIndicesFlattenedSeq(
        sequences=[
            torch.LongTensor([
                identifiers_sub_parts_indexer[sub_part] for sub_part in identifier_sub_parts])
            for identifier_sub_parts in method_pdg.sub_identifiers_by_idx]
        )  # self_indexing_group='identifiers', tgt_indexing_group='identifiers_sub_parts')

    identifiers_sub_parts_vocab_word_index = BatchFlattenedSeq(
        sequences=[
            torch.LongTensor([
                code_task_vocabs.sub_identifiers.get_word_idx_or_unk(sub_part)
                for sub_part in identifier_sub_parts])
            for identifier_sub_parts in method_pdg.sub_identifiers_by_idx]
        )  # self_indexing_group='identifiers')

    identifiers_vocab_word_index = BatchFlattenedTensor(
        tensor=torch.LongTensor([
            code_task_vocabs.identifiers.get_word_idx_or_unk(identifier)
            for identifier in method_pdg.identifier_by_idx])
        )  # self_indexing_group='identifiers')

    # TODO: plug HP for hasher `n_features`
    pp_identifiers_sub_parts_hashings = False  # TODO: add to HPs.
    sub_identifiers_hasher = HashingVectorizer(analyzer='char', n_features=256, ngram_range=(1, 3))
    identifiers_sub_parts_hashings = None if not pp_identifiers_sub_parts_hashings else BatchFlattenedSeq(
        sequences=[
            torch.stack([
                torch.tensor(sub_identifiers_hasher.transform([sub_part]).toarray(), dtype=torch.float32).squeeze()
                for sub_part in identifier_sub_parts])
            for identifier_sub_parts in method_pdg.sub_identifiers_by_idx]
        )  # self_indexing_group='identifiers')

    sub_parts_obfuscation = BatchFlattenedPseudoRandomSamplerFromRange(
        sample_size=len(identifiers_sub_parts_indexer),
        tgt_range_start=len(code_task_vocabs.sub_identifiers.special_words),
        tgt_range_end=len(code_task_vocabs.sub_identifiers)
        )  # initial_seed_salt='idntf', replacement='wo_replacement_within_example')

    identifiers_obfuscation = BatchFlattenedPseudoRandomSamplerFromRange(
        sample_size=len(method_pdg.identifier_by_idx),
        tgt_range_start=len(code_task_vocabs.identifiers.special_words),
        tgt_range_end=len(code_task_vocabs.identifiers)
        )  # initial_seed_salt='idntf', replacement='wo_replacement_within_example')

    identifiers = IdentifiersInputTensors(
        sub_parts_batch=BatchFlattenedTensor(
            tensor=torch.arange(len(identifiers_sub_parts_indexer))
            ),  # self_indexing_group='identifiers_sub_parts'),  # TODO: is it necessary?
        sub_parts_vocab_word_index=BatchFlattenedTensor(
            tensor=torch.LongTensor([
                code_task_vocabs.sub_identifiers.get_word_idx_or_unk(sub_part)
                for sub_part in identifiers_sub_parts_sorted])
            ),  # self_indexing_group='identifiers_sub_parts'),
        identifier_sub_parts_index=identifiers_sub_parts_indices,
        identifier_sub_parts_vocab_word_index=identifiers_sub_parts_vocab_word_index,
        identifiers_vocab_word_index=identifiers_vocab_word_index,
        identifier_sub_parts_hashings=identifiers_sub_parts_hashings,
        sub_parts_obfuscation=sub_parts_obfuscation,
        identifiers_obfuscation=identifiers_obfuscation)
    return identifiers


def preprocess_symbols(
        method: SerMethod, method_pdg: SerMethodPDG, pdg_nodes_to_mask: Dict[int, str]) \
        -> SymbolsInputTensors:
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
        ),  # self_indexing_group='symbols', tgt_indexing_group='identifiers'),
        symbols_appearances_symbol_idx=BatchedFlattenedIndicesFlattenedTensor(
            torch.LongTensor([symbol_occurrence.symbol_idx for symbol_occurrence in symbols_occurrences]),
        ),  # tgt_indexing_group='symbols'),
        symbols_appearances_expression_token_idx=BatchFlattenedTensor(
            torch.LongTensor([symbol_occurrence.within_expr_token_idx for symbol_occurrence in symbols_occurrences])),
        symbols_appearances_cfg_expression_idx=BatchedFlattenedIndicesFlattenedTensor(
            torch.LongTensor([symbol_occurrence.expression_idx for symbol_occurrence in symbols_occurrences]),
        ))  # tgt_indexing_group='code_expressions'))
    return symbols


def preprocess_cfg_nodes_tokenized_expressions(
        method: SerMethod, method_pdg: SerMethodPDG,
        code_task_vocabs: CodeTaskVocabs, pdg_nodes_to_mask: Dict[int, str]) \
        -> CFGCodeExpressionTokensSequenceInputTensors:
    cfg_nodes_tokenized_expressions_token_type = BatchFlattenedSeq(
        [torch.LongTensor(
            [code_task_vocabs.tokens_kinds.get_word_idx(token.kind.value)
             for token in get_pdg_node_tokenized_expression(method=method, pdg_node=pdg_node)])
            for pdg_node in method_pdg.pdg_nodes
            if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask],
    )  # self_indexing_group='code_expressions')
    cfg_nodes_tokenized_expressions = CFGCodeExpressionTokensSequenceInputTensors(
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
        ),  # tgt_indexing_group='identifiers'),
        symbol_index=BatchedFlattenedIndicesFlattenedTensor(torch.LongTensor(
            [token.symbol_idx
             for pdg_node in method_pdg.pdg_nodes
             if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask
             for token in get_pdg_node_tokenized_expression(method=method, pdg_node=pdg_node)
             if token.symbol_idx is not None]),
        ),  # tgt_indexing_group='symbols'),
        is_symbol_mask=BatchFlattenedSeq(
            [torch.BoolTensor(
                [token.symbol_idx is not None
                 for token in get_pdg_node_tokenized_expression(method=method, pdg_node=pdg_node)])
                for pdg_node in method_pdg.pdg_nodes
                if pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask]),
        sequence_shuffler=BatchFlattenedSeqShuffler(
            lengths=tuple(len(seq) for seq in cfg_nodes_tokenized_expressions_token_type.sequences),
        ),  # initial_seed_salt='code_expressions_seq_shuffler'),
        token_idx_to_ast_leaf_idx_mapping_key=None,  # TODO
        token_idx_to_ast_leaf_idx_mapping_value=None)  # TODO
    assert \
        cfg_nodes_tokenized_expressions.symbol_index.indices.size(0) == \
        sum(seq.to(torch.long).sum().item() for seq in cfg_nodes_tokenized_expressions.is_symbol_mask.sequences)
    return cfg_nodes_tokenized_expressions


CFGPaths = namedtuple(
    'CFGPaths',
    ['control_flow_paths', 'pdg_paths', 'control_flow_paths_ngrams'])


def preprocess_control_flow_paths(
        model_hps: NDFAModelHyperParams, method_pdg: SerMethodPDG) -> CFGPaths:
    control_flow_paths = get_all_pdg_simple_paths(
        method_pdg=method_pdg,
        src_pdg_node_idx=method_pdg.entry_pdg_node_idx, tgt_pdg_node_idx=method_pdg.exit_pdg_node_idx,
        control_flow=True, data_dependency=False,
        group_different_edges_of_single_nodes_pair_in_same_path=False,
        max_nr_paths=model_hps.method_code_encoder.max_nr_control_flow_paths)
    pdg_paths = None
    # pdg_paths = get_all_pdg_simple_paths(
    #     method_pdg=method_pdg,
    #     src_pdg_node_idx=method_pdg.entry_pdg_node_idx, tgt_pdg_node_idx=method_pdg.exit_pdg_node_idx,
    #     control_flow=True, data_dependency=True, max_nr_data_dependency_edges_in_path=1,
    #     group_different_edges_of_single_nodes_pair_in_same_path=True,
    #     max_nr_paths=model_hps.method_code_encoder.max_nr_pdg_paths,
    #     remove_data_dependency_edges_from_pdg_nodes_idxs=remove_edges_from_pdg_nodes_idxs)

    if control_flow_paths is None:
        PreprocessLimitation.enforce_limitations(limitations=[
            PreprocessLimitation(
                object_name='#control_flow_paths', value=float('inf'),
                max_val=model_hps.method_code_encoder.max_nr_control_flow_paths)])
    assert control_flow_paths is not None

    if model_hps.method_code_encoder.nr_control_flow_paths_to_sample_during_pp is not None and \
            len(control_flow_paths) > model_hps.method_code_encoder.nr_control_flow_paths_to_sample_during_pp:
        control_flow_paths = random.sample(
            control_flow_paths, model_hps.method_code_encoder.nr_control_flow_paths_to_sample_during_pp)

    # We didn't grouped edges so there is a single edge between two nodes in a path,
    # and actually even if we had been grouping, there is at most one control-flow edge between 2 cfg nodes.
    assert all(len(path_node.edges) <= 1 for path in control_flow_paths for path_node in path)
    control_flow_paths = [
        tuple(
            (path_node.node_idx, None if len(path_node.edges) < 1 else path_node.edges[0].type.value)
            for path_node in path)
        for path in control_flow_paths]
    control_flow_paths.sort()  # for determinism

    # if pdg_paths is None:
    #     PreprocessLimitation.enforce_limitations(limitations=[
    #         PreprocessLimitation(
    #             object_name='#pdg_paths', value=float('inf'),
    #             max_val=model_hps.method_code_encoder.max_nr_pdg_paths)])
    # assert pdg_paths is not None
    # def get_most_important_edge_type_from_edges_group(
    #         edges_group: Tuple[Union[SerPDGControlFlowEdge, SerPDGDataDependencyEdge]]) -> Optional[str]:
    #     if len(edges_group) < 1:
    #         return None
    #     control_flow_edge = next((edge for edge in edges_group if isinstance(edge, SerPDGControlFlowEdge)), None)
    #     if control_flow_edge is not None:
    #         return control_flow_edge.type.value
    #     assert isinstance(edges_group[0], SerPDGDataDependencyEdge)
    #     return 'DataDependency'
    # pdg_paths = [
    #     tuple((path_node.node_idx, get_most_important_edge_type_from_edges_group(path_node.edges))
    #           for path_node in path)
    #     for path in pdg_paths]
    # pdg_paths.sort()  # for determinism

    limitations = []
    limitations.append(PreprocessLimitation(
        object_name='#control_flow_paths', value=len(control_flow_paths),
        min_val=model_hps.method_code_encoder.min_nr_control_flow_paths,
        max_val=model_hps.method_code_encoder.max_nr_control_flow_paths))
    shortest_control_flow_path = min(
        (len(path) for path in control_flow_paths),
        default=model_hps.method_code_encoder.min_control_flow_path_len)
    longest_control_flow_path = max(
        (len(path) for path in control_flow_paths),
        default=model_hps.method_code_encoder.max_control_flow_path_len)
    limitations.append(PreprocessLimitation(
        object_name='|shortest_control_flow_path|', value=shortest_control_flow_path,
        min_val=model_hps.method_code_encoder.min_control_flow_path_len))
    limitations.append(PreprocessLimitation(
        object_name='|longest_control_flow_path|', value=longest_control_flow_path,
        max_val=model_hps.method_code_encoder.max_control_flow_path_len))
    # limitations.append(PreprocessLimitation(
    #     object_name='#pdg_paths', value=len(pdg_paths),
    #     min_val=model_hps.method_code_encoder.min_nr_pdg_paths,
    #     max_val=model_hps.method_code_encoder.max_nr_pdg_paths))
    # shortest_pdg_path = min(
    #     (len(path) for path in pdg_paths),
    #     default=model_hps.method_code_encoder.min_pdg_path_len)
    # longest_pdg_path = max(
    #     (len(path) for path in pdg_paths),
    #     default=model_hps.method_code_encoder.max_pdg_path_len)
    # limitations.append(PreprocessLimitation(
    #     object_name='|shortest_pdg_path|', value=shortest_pdg_path,
    #     min_val=model_hps.method_code_encoder.min_pdg_path_len))
    # limitations.append(PreprocessLimitation(
    #     object_name='|longest_pdg_path|', value=longest_pdg_path,
    #     max_val=model_hps.method_code_encoder.max_pdg_path_len))
    PreprocessLimitation.enforce_limitations(limitations=limitations)

    ngrams_min_n, ngrams_max_n = 2, 7  # TODO: make these HPs, which are also part of the pp data properties
    control_flow_paths_ngrams = extract_ngrams_from_sequence(
        sequences=control_flow_paths, ngrams_min_n=ngrams_min_n, ngrams_max_n=ngrams_max_n)

    control_flow_paths_node_idxs_set = {node_idx for path in control_flow_paths for node_idx, _ in path}
    cfg_node_idxs_not_in_any_cfg_path = (set(
        node.idx for node in method_pdg.pdg_nodes) - control_flow_paths_node_idxs_set) - {0}
    # TODO: replace with assert! after we fix the <try..catch..finally> control-flow edges in the JavaExtractor.
    PreprocessLimitation.enforce_limitations(limitations=[PreprocessLimitation(
        object_name='#cfg_node_idxs_not_in_any_cfg_path', value=len(cfg_node_idxs_not_in_any_cfg_path),
        max_val=0)])

    return CFGPaths(
        control_flow_paths=control_flow_paths,
        pdg_paths=pdg_paths,
        control_flow_paths_ngrams=control_flow_paths_ngrams)

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


def sanitize_sub_asts_to_ignore_or_to_mask(
        method_ast: SerMethodAST,
        sub_ast_root_indices_to_ignore: Optional[Set[int]] = None,
        sub_ast_root_indices_to_mask: Optional[Dict[int, str]] = None):
    if sub_ast_root_indices_to_ignore is None:
        sub_ast_root_indices_to_ignore = set()
    if sub_ast_root_indices_to_mask is None:
        sub_ast_root_indices_to_mask = {}
    masked_sub_asts_nodes_without_root: Dict[int, Set[int]] = {
        sub_ast_root_idx: {
            ast_node.idx
            for ast_node, _, _
            in traverse_ast(method_ast=method_ast, root_sub_ast_node_idx=sub_ast_root_idx)
            if ast_node.idx != sub_ast_root_idx}
        for sub_ast_root_idx in sub_ast_root_indices_to_mask.keys()}
    # The sub-ASTs must be disjoint. Because their roots should not be ignored (they are replaced with some mask).
    # If they appear in another sub-AST (not as its root) they must be ignored.
    assert all(len(sub_ast_1_nodes & sub_ast_2_nodes) == 0
               for sub_ast_1_nodes, sub_ast_2_nodes
               in itertools.combinations(masked_sub_asts_nodes_without_root.values(), 2))

    ast_nodes_indices_to_ignore: Set[int] = {
        ast_node.idx
        for sub_ast_root_idx in sub_ast_root_indices_to_ignore
        for ast_node, _, _ in traverse_ast(
            method_ast=method_ast, root_sub_ast_node_idx=sub_ast_root_idx)}
    assert ast_nodes_indices_to_ignore & sub_ast_root_indices_to_ignore == sub_ast_root_indices_to_ignore
    ast_nodes_indices_to_ignore.update((
        ast_node_idx
        for sub_ast_nodes in masked_sub_asts_nodes_without_root.values()
        for ast_node_idx in sub_ast_nodes))
    assert len(ast_nodes_indices_to_ignore & sub_ast_root_indices_to_mask.keys()) == 0

    return masked_sub_asts_nodes_without_root, ast_nodes_indices_to_ignore


def enforce_limits_and_sample_ast_paths(
        ast_paths: ASTPaths,
        max_nr_ast_leaf_to_leaf_paths: Optional[int] = None,
        max_nr_ast_leaf_to_root_paths: Optional[int] = None,
        nr_ast_leaf_to_leaf_paths_to_sample: Optional[int] = None,
        nr_ast_leaf_to_root_paths_to_sample: Optional[int] = None):
    if max_nr_ast_leaf_to_leaf_paths is not None:
        PreprocessLimitation.enforce_limitations(limitations=[
            PreprocessLimitation(
                object_name='max_nr_ast_leaf_to_leaf_paths',
                value=len(ast_paths.leaf_to_leaf_paths),
                max_val=max_nr_ast_leaf_to_leaf_paths)])
    if max_nr_ast_leaf_to_root_paths is not None:
        PreprocessLimitation.enforce_limitations(limitations=[
            PreprocessLimitation(
                object_name='max_nr_ast_leaf_to_root_paths',
                value=len(ast_paths.leaf_to_root_paths),
                max_val=max_nr_ast_leaf_to_root_paths)])
    if nr_ast_leaf_to_leaf_paths_to_sample is not None and \
            len(ast_paths.leaf_to_leaf_paths) > nr_ast_leaf_to_leaf_paths_to_sample:
        sampled_keys = random.sample(ast_paths.leaf_to_leaf_paths.keys(), nr_ast_leaf_to_leaf_paths_to_sample)
        leaf_to_leaf_paths = {key: ast_paths.leaf_to_leaf_paths[key] for key in sampled_keys}
        ast_paths = dataclasses.replace(ast_paths, leaf_to_leaf_paths=leaf_to_leaf_paths)
    if nr_ast_leaf_to_root_paths_to_sample is not None and \
            len(ast_paths.leaf_to_root_paths) > nr_ast_leaf_to_root_paths_to_sample:
        sampled_keys = random.sample(ast_paths.leaf_to_root_paths.keys(), nr_ast_leaf_to_root_paths_to_sample)
        leaf_to_root_paths = {key: ast_paths.leaf_to_root_paths[key] for key in sampled_keys}
        ast_paths = dataclasses.replace(ast_paths, leaf_to_root_paths=leaf_to_root_paths)
    return ast_paths


def preprocess_method_ast(
        model_hps: NDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs,
        method: SerMethod, method_ast: SerMethodAST,
        sub_ast_root_indices_to_ignore: Optional[Set[int]] = None,
        sub_ast_root_indices_to_mask: Optional[Dict[int, str]] = None) -> MethodASTInputTensors:
    pp_method_ast_paths = True  # TODO: make it a preprocess parameter
    method_ast_paths: Optional[ASTPaths] = None

    # Note: For some sub-AST that have to be masked-out:
    #       - Completely ignore the descendants of this sub-AST (the ast-nodes except its root):
    #         (i) override all of their data (type, identifier, symbol, keyword, modifier, #childs, place at parent, ..)
    #             to be the `<PAD>` special word.
    #         (ii) Do not include them in any ast-path or as a src/tgt of any graph-edge.
    #       - But *KEEP* the root node of this sub-AST while:
    #         (i) overriding its type (to the given mask str); and
    #         (ii) override its additional data (identifier, symbol, keyword, modifier, #childs, place at parent, ..)
    #              to be the `<PAD>` special word.
    #         (iii) *KEEP* it in ast-paths and as a src/tgt of graph-edges.
    masked_sub_asts_nodes_without_root, ast_nodes_indices_to_ignore = sanitize_sub_asts_to_ignore_or_to_mask(
        method_ast=method_ast,
        sub_ast_root_indices_to_ignore=sub_ast_root_indices_to_ignore,
        sub_ast_root_indices_to_mask=sub_ast_root_indices_to_mask)
    ast_nodes_indices_to_ignore_or_to_mask = ast_nodes_indices_to_ignore | sub_ast_root_indices_to_mask.keys()

    if pp_method_ast_paths:
        method_ast_paths = get_all_ast_paths(
            method_ast=method_ast,
            subtrees_to_ignore=ast_nodes_indices_to_ignore)
        assert len(set(method_ast_paths.leaves_sequence) & ast_nodes_indices_to_ignore) == 0
        assert set(method_ast_paths.leaves_sequence) & sub_ast_root_indices_to_mask.keys() == \
               sub_ast_root_indices_to_mask.keys()
        assert len(set(method_ast_paths.postorder_traversal_sequence) & ast_nodes_indices_to_ignore) == 0

        method_ast_paths = enforce_limits_and_sample_ast_paths(
            ast_paths=method_ast_paths,
            max_nr_ast_leaf_to_leaf_paths=model_hps.method_code_encoder.max_nr_method_ast_leaf_to_leaf_paths,
            max_nr_ast_leaf_to_root_paths=model_hps.method_code_encoder.max_nr_method_ast_leaf_to_root_paths,
            nr_ast_leaf_to_leaf_paths_to_sample=
            model_hps.method_code_encoder.nr_method_ast_leaf_to_leaf_paths_to_sample_during_pp,
            nr_ast_leaf_to_root_paths_to_sample=
            model_hps.method_code_encoder.nr_method_ast_leaf_to_root_paths_to_sample_during_pp)

    dgl_ast_edges = torch.LongTensor(
        [[ast_node.idx, child_node_idx]
         for ast_node in method_ast.nodes
         for child_node_idx in ast_node.children_idxs
         if ast_node.idx not in ast_nodes_indices_to_ignore and
         child_node_idx not in ast_nodes_indices_to_ignore]).t().chunk(chunks=2, dim=0)
    dgl_ast_edges = tuple(u.view(-1) for u in dgl_ast_edges)
    dgl_ast = dgl.graph(data=dgl_ast_edges, num_nodes=len(method_ast.nodes))

    method_ast_input_tensors = MethodASTInputTensors(
        ast_node_types=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_types.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_nodes_indices_to_ignore else
                    sub_ast_root_indices_to_mask.get(ast_node.idx, ast_node.type.value))
                # TODO: should it be a special '<MASK>' vocab word instead of a simple padding?
                for ast_node in method_ast.nodes]),
        ),  # self_indexing_group='ast_nodes'),
        ast_node_major_types=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_major_types.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_nodes_indices_to_ignore else
                    sub_ast_root_indices_to_mask.get(ast_node.idx, ast_node.type.value.split('_')[0]))
                # TODO: should it be a special '<MASK>' vocab word instead of a simple padding?
                for ast_node in method_ast.nodes])),
        ast_node_minor_types=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_minor_types.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_nodes_indices_to_ignore_or_to_mask or '_' not in ast_node.type.value else
                    ast_node.type.value[ast_node.type.value.find('_') + 1:])
                # TODO: should it be a special '<MASK>' vocab word instead of a simple padding?
                for ast_node in method_ast.nodes])),
        ast_node_child_ltr_position=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_child_pos.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_nodes_indices_to_ignore else
                    '<+ROOT>'
                    if ast_node.parent_node_idx is None or ast_node.child_place_at_parent is None else
                    f'+{ast_node.child_place_at_parent + 1}', unk_word='<+MORE>')
                for ast_node in method_ast.nodes])),
        ast_node_child_rtl_position=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_child_pos.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_nodes_indices_to_ignore else
                    '<-ROOT>'
                    if ast_node.parent_node_idx is None or ast_node.child_place_at_parent is None else
                    f'-{len(method_ast.nodes[ast_node.parent_node_idx].children_idxs) - ast_node.child_place_at_parent}',
                    unk_word='<-MORE>')
                for ast_node in method_ast.nodes])),
        ast_node_nr_children=BatchFlattenedTensor(
            torch.LongTensor([
                code_task_vocabs.ast_node_nr_children.get_word_idx_or_unk(
                    '<PAD>'
                    if ast_node.idx in ast_nodes_indices_to_ignore_or_to_mask else
                    f'{len(ast_node.children_idxs)}', unk_word='<MORE>')
                for ast_node in method_ast.nodes])),

        ast_nodes_with_identifier_leaf_nodes_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                ast_node.idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                   ast_node.code_sub_token_range_ref is not None and
                   ast_node.code_sub_token_range_ref.begin_token_idx == ast_node.code_sub_token_range_ref.end_token_idx and
                   method.code.tokenized[
                       ast_node.code_sub_token_range_ref.begin_token_idx].identifier_idx is not None and
                   ast_node.idx not in ast_nodes_indices_to_ignore_or_to_mask]),
        ),  # tgt_indexing_group='ast_nodes'),
        ast_nodes_with_identifier_leaf_identifier_idx=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].identifier_idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                   ast_node.code_sub_token_range_ref is not None and
                   ast_node.code_sub_token_range_ref.begin_token_idx == ast_node.code_sub_token_range_ref.end_token_idx and
                   method.code.tokenized[
                       ast_node.code_sub_token_range_ref.begin_token_idx].identifier_idx is not None and
                   ast_node.idx not in ast_nodes_indices_to_ignore_or_to_mask]),
        ),  # tgt_indexing_group='identifiers'),

        ast_nodes_with_symbol_leaf_nodes_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                ast_node.idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                   ast_node.code_sub_token_range_ref is not None and
                   ast_node.code_sub_token_range_ref.begin_token_idx == ast_node.code_sub_token_range_ref.end_token_idx and
                   method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].symbol_idx is not None and
                   ast_node.idx not in ast_nodes_indices_to_ignore_or_to_mask]),
        ),  # tgt_indexing_group='ast_nodes'),
        ast_nodes_with_symbol_leaf_symbol_idx=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].symbol_idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                   ast_node.code_sub_token_range_ref is not None and
                   ast_node.code_sub_token_range_ref.begin_token_idx == ast_node.code_sub_token_range_ref.end_token_idx and
                   method.code.tokenized[ast_node.code_sub_token_range_ref.begin_token_idx].symbol_idx is not None and
                   ast_node.idx not in ast_nodes_indices_to_ignore_or_to_mask]),
        ),  # tgt_indexing_group='symbols'),

        ast_nodes_with_primitive_type_leaf_nodes_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                ast_node.idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and
                   ast_node.type == SerASTNodeType.PRIMITIVE_TYPE and
                   ast_node.type_name is not None and
                   ast_node.idx not in ast_nodes_indices_to_ignore_or_to_mask]),
        ),  # tgt_indexing_group='ast_nodes'),
        ast_nodes_with_primitive_type_leaf_primitive_type=BatchFlattenedTensor(tensor=torch.LongTensor([
            code_task_vocabs.primitive_types.get_word_idx_or_unk(ast_node.type_name)
            for ast_node in method_ast.nodes
            if len(ast_node.children_idxs) == 0 and
               ast_node.type == SerASTNodeType.PRIMITIVE_TYPE and
               ast_node.type_name is not None and
               ast_node.idx not in ast_nodes_indices_to_ignore_or_to_mask])),

        ast_nodes_with_modifier_leaf_nodes_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                ast_node.idx
                for ast_node in method_ast.nodes
                if len(ast_node.children_idxs) == 0 and ast_node.modifier is not None and
                   ast_node.idx not in ast_nodes_indices_to_ignore_or_to_mask]),
        ),  # tgt_indexing_group='ast_nodes'),
        ast_nodes_with_modifier_leaf_modifier=BatchFlattenedTensor(tensor=torch.LongTensor([
            code_task_vocabs.modifiers.get_word_idx_or_unk(ast_node.modifier)
            for ast_node in method_ast.nodes
            if len(ast_node.children_idxs) == 0 and ast_node.modifier is not None and
               ast_node.idx not in ast_nodes_indices_to_ignore_or_to_mask])),

        ast_leaf_to_leaf_paths_node_indices=None if not pp_method_ast_paths else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor([path_node.ast_node_idx for path_node in path])
                       for path in method_ast_paths.leaf_to_leaf_paths.values()],
        ),  # tgt_indexing_group='ast_nodes'),
        ast_leaf_to_leaf_paths_child_place=None if not pp_method_ast_paths else BatchFlattenedSeq(
            sequences=[
                torch.LongTensor([
                    code_task_vocabs.ast_traversal_orientation.get_word_idx(
                        'child_place=UNK' if path_node.child_place_in_parent is None else
                        f'child_place={min(path_node.child_place_in_parent, 4 - 1)}')
                    for path_node in path])
                for path in method_ast_paths.leaf_to_leaf_paths.values()]),
        ast_leaf_to_leaf_paths_vertical_direction=None if not pp_method_ast_paths else BatchFlattenedSeq(
            sequences=[
                torch.LongTensor([
                    code_task_vocabs.ast_traversal_orientation.get_word_idx(
                        f'DIR={path_node.direction.value}')
                    for path_node in path])
                for path in method_ast_paths.leaf_to_leaf_paths.values()]),
        ast_leaf_to_root_paths_node_indices=None if not pp_method_ast_paths else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor([path_node.ast_node_idx for path_node in path])
                       for path in method_ast_paths.leaf_to_root_paths.values()],
        ),  # tgt_indexing_group='ast_nodes'),
        ast_leaf_to_root_paths_child_place=None if not pp_method_ast_paths else BatchFlattenedSeq(
            sequences=[
                torch.LongTensor([
                    code_task_vocabs.ast_traversal_orientation.get_word_idx(
                        'child_place=UNK' if path_node.child_place_in_parent is None else
                        f'child_place={min(path_node.child_place_in_parent, 4 - 1)}')
                    for path_node in path])
                for path in method_ast_paths.leaf_to_root_paths.values()]),
        ast_leaves_sequence_node_indices=None if not pp_method_ast_paths else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor(method_ast_paths.leaves_sequence)],
        ),  # tgt_indexing_group='ast_nodes'),
        siblings_sequences_node_indices=None if not pp_method_ast_paths else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[
                torch.LongTensor(siblings_sequence)
                for siblings_sequence in method_ast_paths.siblings_sequences.values()],
        ),  # tgt_indexing_group='ast_nodes'),
        siblings_w_parent_sequences_node_indices=
        None if not pp_method_ast_paths else BatchedFlattenedIndicesFlattenedSeq(
            sequences=[
                torch.LongTensor((parent_ast_node_idx,) + siblings_sequence)
                for parent_ast_node_idx, siblings_sequence in method_ast_paths.siblings_sequences.items()],
        ),  # tgt_indexing_group='ast_nodes'),
        dgl_tree=dgl_ast)

    assert all(len(s1 & s2) == 0 for s1, s2 in itertools.combinations([
        set(method_ast_input_tensors.ast_nodes_with_identifier_leaf_nodes_indices.indices.tolist()),
        set(method_ast_input_tensors.ast_nodes_with_primitive_type_leaf_nodes_indices.indices.tolist()),
        set(method_ast_input_tensors.ast_nodes_with_modifier_leaf_nodes_indices.indices.tolist())], r=2))

    return method_ast_input_tensors


def preprocess_sub_ast_for_cfg_expressions(
        model_hps: NDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs,
        method_pdg: SerMethodPDG, method_ast: SerMethodAST, pdg_nodes_to_mask: Dict[int, str],
        sub_ast_root_indices_to_mask: Optional[Dict[int, str]] = None,
        sub_ast_root_indices_to_ignore: Optional[Set[int]] = None) -> PDGExpressionsSubASTInputTensors:
    # Note: For some sub-AST that have to be masked-out:
    #       - Completely ignore the descendants of this sub-AST (the ast-nodes except its root):
    #         (i) override all of their data (type, identifier, symbol, keyword, modifier, #childs, place at parent, ..)
    #             to be the `<PAD>` special word.
    #         (ii) Do not include them in any ast-path or as a src/tgt of any graph-edge.
    #       - But *KEEP* the root node of this sub-AST while:
    #         (i) overriding its type (to the given mask str); and
    #         (ii) override its additional data (identifier, symbol, keyword, modifier, #childs, place at parent, ..)
    #              to be the `<PAD>` special word.
    #         (iii) *KEEP* it in ast-paths and as a src/tgt of graph-edges.
    masked_sub_asts_nodes_without_root, ast_nodes_indices_to_ignore = sanitize_sub_asts_to_ignore_or_to_mask(
        method_ast=method_ast,
        sub_ast_root_indices_to_ignore=sub_ast_root_indices_to_ignore,
        sub_ast_root_indices_to_mask=sub_ast_root_indices_to_mask)
    ast_nodes_indices_to_ignore_or_to_mask = ast_nodes_indices_to_ignore | sub_ast_root_indices_to_mask.keys()

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
        # Note: The following 2 limitations are important because we remove the method body from the
        #       sub-AST of the `MethodEntry` CFG node. We want only the method signature there.
        PreprocessLimitation(
            object_name='last_method_decl_child_is_block_stmt',
            value=int(last_method_decl_child_is_block_stmt),
            min_val=1, max_val=1),
        PreprocessLimitation(
            object_name='nr_non_last_method_decl_children_which_are_block_stmt',
            value=nr_non_last_method_decl_children_which_are_block_stmt,
            min_val=0, max_val=0),
        # Note: The following 2 limitations are important because we remove the catch entry body from the
        #       sub-AST of the `CatchClauseEntry` CFG node. We want only the caught exception params there.
        PreprocessLimitation(
            object_name='last_catch_clause_entry_child_is_block_stmt',
            value=int(last_catch_clause_entry_child_is_block_stmt),
            min_val=1, max_val=1),
        PreprocessLimitation(
            object_name='nr_non_last_catch_clause_entry_children_which_are_block_stmt',
            value=nr_non_last_catch_clause_entry_children_which_are_block_stmt,
            min_val=0, max_val=0)])

    ast_paths_per_pdg_node: Dict[int, ASTPaths] = {
        pdg_node.idx: get_all_ast_paths(
            method_ast=method_ast, sub_ast_root_node_idx=pdg_node.ast_node_idx,
            subtrees_to_ignore={method_ast.nodes[pdg_node.ast_node_idx].children_idxs[-1]} | ast_nodes_indices_to_ignore
            if pdg_node.control_kind in {SerPDGNodeControlKind.METHOD_ENTRY, SerPDGNodeControlKind.CATCH_CLAUSE_ENTRY}
            else ast_nodes_indices_to_ignore)
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.ast_node_idx is not None and
           pdg_node.code_sub_token_range_ref is not None and
           pdg_node.idx not in pdg_nodes_to_mask and
           pdg_node.ast_node_idx not in ast_nodes_indices_to_ignore_or_to_mask}  # FIXME: should it be ignored only?

    ast_paths_per_pdg_node = {
        pdg_node_idx: enforce_limits_and_sample_ast_paths(
            ast_paths=sub_ast_paths,
            max_nr_ast_leaf_to_leaf_paths=model_hps.method_code_encoder.max_nr_cfg_node_sub_ast_leaf_to_leaf_paths,
            max_nr_ast_leaf_to_root_paths=model_hps.method_code_encoder.max_nr_cfg_node_sub_ast_leaf_to_root_paths,
            nr_ast_leaf_to_leaf_paths_to_sample=
            model_hps.method_code_encoder.nr_cfg_node_sub_ast_leaf_to_leaf_paths_to_sample_during_pp,
            nr_ast_leaf_to_root_paths_to_sample=
            model_hps.method_code_encoder.nr_cfg_node_sub_ast_leaf_to_root_paths_to_sample_during_pp)
        for pdg_node_idx, sub_ast_paths in ast_paths_per_pdg_node.items()
    }

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
        len(sub_ast_nodes_indices & ast_nodes_indices_to_ignore) == 0
        for sub_ast_nodes_indices in pdg_node_to_ast_node_indices.values())

    # No ast-node that belongs to more than one pdg-node
    assert all(len(s1 & s2) == 0 for s1, s2 in itertools.combinations(pdg_node_to_ast_node_indices.values(), r=2))
    ast_node_idx_to_pdg_node = {
        ast_node_idx: pdg_node_idx
        for pdg_node_idx, ast_node_indices in pdg_node_to_ast_node_indices.items()
        for ast_node_idx in ast_node_indices}
    assert len(ast_paths_per_pdg_node) > 0
    assert all(len(ast_paths.postorder_traversal_sequence) > 0 for ast_paths in ast_paths_per_pdg_node.values())
    assert all(
        len(set(sub_ast_paths.leaves_sequence) & ast_nodes_indices_to_ignore) == 0
        for sub_ast_paths in ast_paths_per_pdg_node.values())
    assert all(
        len(set(sub_ast_paths.postorder_traversal_sequence) & ast_nodes_indices_to_ignore) == 0
        for sub_ast_paths in ast_paths_per_pdg_node.values())

    pdg_nodes_expressions_dgl_ast_edges = torch.LongTensor(
        [[ast_node_idx, child_node_idx]
         for sub_ast_nodes in pdg_node_to_ast_node_indices.values()
         for ast_node_idx in sub_ast_nodes
         for child_node_idx in method_ast.nodes[ast_node_idx].children_idxs
         if child_node_idx in sub_ast_nodes]).t().chunk(chunks=2, dim=0)
    pdg_nodes_expressions_dgl_ast_edges = tuple(u.view(-1) for u in pdg_nodes_expressions_dgl_ast_edges)
    pdg_nodes_expressions_dgl_ast = \
        dgl.graph(data=pdg_nodes_expressions_dgl_ast_edges, num_nodes=len(method_ast.nodes))

    cfg_nodes_expressions_sub_ast_input_tensors = PDGExpressionsSubASTInputTensors(
        ast_leaf_to_leaf_paths_pdg_node_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                pdg_node_idx
                for pdg_node_idx, sub_ast_paths in ast_paths_per_pdg_node.items()
                for _ in sub_ast_paths.leaf_to_leaf_paths.values()]),
        ),  # tgt_indexing_group='cfg_nodes'),
        ast_leaf_to_leaf_paths_node_indices=BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor([path_node.ast_node_idx for path_node in path])
                       for sub_ast_paths in ast_paths_per_pdg_node.values()
                       for path in sub_ast_paths.leaf_to_leaf_paths.values()],
        ),  # tgt_indexing_group='ast_nodes'),
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
        ),  # tgt_indexing_group='cfg_nodes'),
        ast_leaf_to_root_paths_node_indices=BatchedFlattenedIndicesFlattenedSeq(
            sequences=[torch.LongTensor([path_node.ast_node_idx for path_node in path])
                       for sub_ast_paths in ast_paths_per_pdg_node.values()
                       for path in sub_ast_paths.leaf_to_root_paths.values()],
        ),  # tgt_indexing_group='ast_nodes'),
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
        ),  # tgt_indexing_group='ast_nodes'),
        siblings_sequences_node_indices=BatchedFlattenedIndicesFlattenedSeq(
            sequences=[
                torch.LongTensor(siblings_sequence)
                for sub_ast_paths in ast_paths_per_pdg_node.values()
                for siblings_sequence in sub_ast_paths.siblings_sequences.values()],
        ),  # tgt_indexing_group='ast_nodes'),
        siblings_w_parent_sequences_node_indices=BatchedFlattenedIndicesFlattenedSeq(
            sequences=[
                torch.LongTensor((parent_ast_node_idx,) + siblings_sequence)
                for sub_ast_paths in ast_paths_per_pdg_node.values()
                for parent_ast_node_idx, siblings_sequence in sub_ast_paths.siblings_sequences.items()],
        ),  # tgt_indexing_group='ast_nodes'),
        siblings_sequences_pdg_node_indices=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                pdg_node_idx
                for pdg_node_idx, sub_ast_paths in ast_paths_per_pdg_node.items()
                for _ in sub_ast_paths.siblings_sequences.values()]),
        ),  # tgt_indexing_group='cfg_nodes'),
        pdg_node_idx_to_sub_ast_root_idx_mapping_key=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                pdg_node.idx
                for pdg_node in method_pdg.pdg_nodes
                if pdg_node.code_sub_token_range_ref and pdg_node.idx not in pdg_nodes_to_mask
                   and pdg_node.ast_node_idx not in ast_nodes_indices_to_ignore]),
        ),  # tgt_indexing_group='cfg_nodes'),
        pdg_node_idx_to_sub_ast_root_idx_mapping_value=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor([
                pdg_node.ast_node_idx
                for pdg_node in method_pdg.pdg_nodes
                if pdg_node.code_sub_token_range_ref and pdg_node.idx not in pdg_nodes_to_mask
                   and pdg_node.ast_node_idx not in ast_nodes_indices_to_ignore]),
        ),  # tgt_indexing_group='ast_nodes'),
        ast_node_idx_to_pdg_node_idx_mapping_key=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor(list(ast_node_idx_to_pdg_node.keys())),
        ),  # tgt_indexing_group='ast_nodes'),
        ast_node_idx_to_pdg_node_idx_mapping_value=BatchedFlattenedIndicesFlattenedTensor(
            indices=torch.LongTensor(list(ast_node_idx_to_pdg_node.values())),
        ),  # tgt_indexing_group='cfg_nodes'),
        dgl_tree=pdg_nodes_expressions_dgl_ast)
    return cfg_nodes_expressions_sub_ast_input_tensors


def preprocess_method_code_tokens_seq(
        code_task_vocabs: CodeTaskVocabs, method: SerMethod,
        method_pdg: SerMethodPDG, pdg_nodes_to_mask: Dict[int, str]) \
        -> MethodCodeTokensSequenceInputTensors:
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
        if token_idx not in token_indices_to_ignore])]
    )  # self_indexing_group='code_expressions')
    method_tokenized_code = MethodCodeTokensSequenceInputTensors(
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
        ),  # tgt_indexing_group='identifiers'),
        symbol_index=BatchedFlattenedIndicesFlattenedTensor(torch.LongTensor(
            [token.symbol_idx
             for token_idx, token in enumerate(method.code.tokenized)
             if token.symbol_idx is not None
             if token_idx not in token_indices_to_mask_or_ignore]),
        ),  # tgt_indexing_group='symbols'),
        is_symbol_mask=BatchFlattenedSeq([torch.BoolTensor(
            [token.symbol_idx is not None and token_idx not in token_indices_mask
             for token_idx, token in enumerate(method.code.tokenized)
             if token_idx not in token_indices_to_ignore])]),
        sequence_shuffler=BatchFlattenedSeqShuffler(
            lengths=tuple(len(seq) for seq in method_tokenized_code_token_type.sequences),
        ),  # initial_seed_salt='code_expressions_seq_shuffler'),
        token_idx_to_ast_leaf_idx_mapping_key=None,  # TODO
        token_idx_to_ast_leaf_idx_mapping_value=None)  # TODO
    assert method_tokenized_code.symbol_index.indices.size(0) == \
           method_tokenized_code.is_symbol_mask.sequences[0].to(torch.long).sum().item()
    return method_tokenized_code


def preprocess_pdg(
        model_hps: NDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs,
        method: SerMethod, method_ast: SerMethodAST, method_pdg: SerMethodPDG,
        pdg_nodes_to_mask: Dict[int, str], sub_ast_root_indices_to_mask: Dict[int, str]) \
        -> PDGInputTensors:
    # Note:
    # For pdg-nodes of control kind CATCH_CLAUSE_ENTRY / METHOD_ENTRY, their sub-AST should be trimmed.
    # The last child of the root is a of type block-stmt and should not be considered as a part of this sub-AST.
    # The tokenized expression ref of these pdg-nodes is already trimmed correctly in the raw extracted data.

    # has_expression ==> sub_token_range exists
    assert all(not pdg_node.has_expression or pdg_node.code_sub_token_range_ref is not None
               for pdg_node in method_pdg.pdg_nodes)
    # !has_expression & sub_token_range exists ==> MethodEntry/CatchClauseEntry
    assert all(pdg_node.control_kind in {SerPDGNodeControlKind.METHOD_ENTRY,
                                         SerPDGNodeControlKind.CATCH_CLAUSE_ENTRY}
               for pdg_node in method_pdg.pdg_nodes
               if not pdg_node.has_expression and pdg_node.code_sub_token_range_ref is not None)
    # Note: The eventual `has_expression` tensor mask is different than the `pdg_node.has_expression` flag.
    #       The tensor mask is by `code_sub_token_range_ref` (which might exist even if !pdg_node.has_expression)

    cfg_paths = preprocess_control_flow_paths(model_hps=model_hps, method_pdg=method_pdg)
    cfg_nodes_expressions_sub_ast_input_tensors = preprocess_sub_ast_for_cfg_expressions(
        model_hps=model_hps, code_task_vocabs=code_task_vocabs, method_pdg=method_pdg, method_ast=method_ast,
        pdg_nodes_to_mask=pdg_nodes_to_mask, sub_ast_root_indices_to_mask=sub_ast_root_indices_to_mask)

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

    pdg_input_tensors = PDGInputTensors(
        cfg_nodes_control_kind=BatchFlattenedTensor(torch.LongTensor(
            [code_task_vocabs.pdg_node_control_kinds.get_word_idx(
                pdg_node.control_kind.value
                if pdg_node.idx not in pdg_nodes_to_mask else
                pdg_nodes_to_mask[pdg_node.idx])
                for pdg_node in method_pdg.pdg_nodes])),  # self_indexing_group='cfg_nodes'),
        cfg_nodes_has_expression_mask=BatchFlattenedTensor(torch.BoolTensor(
            [pdg_node.code_sub_token_range_ref is not None and pdg_node.idx not in pdg_nodes_to_mask
             for pdg_node in method_pdg.pdg_nodes])),
        cfg_nodes_tokenized_expressions=preprocess_cfg_nodes_tokenized_expressions(
            method=method, method_pdg=method_pdg, code_task_vocabs=code_task_vocabs,
            pdg_nodes_to_mask=pdg_nodes_to_mask),
        cfg_nodes_expressions_ast=cfg_nodes_expressions_sub_ast_input_tensors,
        cfg_nodes_random_permutation=BatchedFlattenedIndicesPseudoRandomPermutation(
            # tgt_indexing_group='cfg_nodes',
            # batch_dependent_seed=True, example_dependent_seed=True, initial_seed_salt='cfgn'),
        ),
        cfg_control_flow_paths=CFGPathsInputTensors(
            nodes_indices=BatchedFlattenedIndicesFlattenedSeq(
                sequences=[torch.LongTensor([node_idx for node_idx, _ in path])
                           for path in cfg_paths.control_flow_paths],
            ),  # tgt_indexing_group='cfg_nodes'),
            edges_types=BatchFlattenedSeq(
                sequences=[torch.LongTensor(
                    [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx(
                        '<PAD>' if edge_type is None else edge_type)
                        for _, edge_type in path])
                    for path in cfg_paths.control_flow_paths])),
        cfg_pdg_paths=None if cfg_paths.pdg_paths is None else CFGPathsInputTensors(
            nodes_indices=BatchedFlattenedIndicesFlattenedSeq(
                sequences=[torch.LongTensor([node_idx for node_idx, _ in path]) for path in cfg_paths.pdg_paths],
            ),  # tgt_indexing_group='cfg_nodes'),
            edges_types=BatchFlattenedSeq(
                sequences=[torch.LongTensor(
                    [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx(
                        '<PAD>' if edge_type is None else edge_type)
                        for _, edge_type in path])
                    for path in cfg_paths.pdg_paths])),
        cfg_control_flow_paths_exact_ngrams=TensorsDataDict[int, CFGPathsNGramsInputTensors]({
            key: CFGPathsNGramsInputTensors(
                nodes_indices=BatchedFlattenedIndicesFlattenedSeq(
                    sequences=[torch.LongTensor([node_idx for node_idx, _ in ngram]) for ngram in ngrams],
                ),  # tgt_indexing_group='cfg_nodes'),
                edges_types=BatchFlattenedSeq(
                    sequences=[torch.LongTensor(
                        [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx(
                            '<PAD>' if edge_type is None else edge_type)
                            for _, edge_type in ngram])
                        for ngram in ngrams]))
            for key, ngrams in cfg_paths.control_flow_paths_ngrams.exact_ngrams.items()}),
        cfg_control_flow_paths_partial_ngrams=TensorsDataDict[int, CFGPathsNGramsInputTensors]({
            key: CFGPathsNGramsInputTensors(
                nodes_indices=BatchedFlattenedIndicesFlattenedSeq(
                    sequences=[torch.LongTensor([node_idx for node_idx, _ in ngram]) for ngram in ngrams],
                ),  # tgt_indexing_group='cfg_nodes'),
                edges_types=BatchFlattenedSeq(
                    sequences=[torch.LongTensor(
                        [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx(
                            '<PAD>' if edge_type is None else edge_type)
                            for _, edge_type in ngram])
                        for ngram in ngrams]))
            for key, ngrams in cfg_paths.control_flow_paths_ngrams.partial_ngrams.items()}),
        cfg_control_flow_graph=cfg_control_flow_graph)
    return pdg_input_tensors


def preprocess_code_task_example(
        model_hps: NDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs,
        method: SerMethod, method_pdg: SerMethodPDG, method_ast: SerMethodAST,
        remove_edges_from_pdg_nodes_idxs: Optional[Set[int]] = None,
        pdg_nodes_to_mask: Optional[Dict[int, str]] = None) -> Optional[MethodCodeInputTensors]:
    enforce_code_task_input_pp_limitations(
        model_hps=model_hps, method=method, method_pdg=method_pdg, method_ast=method_ast)
    if pdg_nodes_to_mask is None:
        pdg_nodes_to_mask = {}

    identifiers = preprocess_identifiers(method_pdg=method_pdg, code_task_vocabs=code_task_vocabs)
    symbols = preprocess_symbols(method=method, method_pdg=method_pdg, pdg_nodes_to_mask=pdg_nodes_to_mask)

    # We ignore the whole sub-ASTs of PDG nodes to mask.
    # TODO: Keep only the root node of this sub-AST, make the it a leaf (remove it's descendents),
    #  and set its type to be the custom string of the value from the `pdg_nodes_to_mask` dict.
    sub_ast_root_indices_to_mask: Dict[int, str] = {
        method_pdg.pdg_nodes[pdg_node_idx].ast_node_idx: mask_str
        for pdg_node_idx, mask_str in pdg_nodes_to_mask.items()
        if method_pdg.pdg_nodes[pdg_node_idx].ast_node_idx is not None}
    method_ast_input_tensors = preprocess_method_ast(
        model_hps=model_hps, code_task_vocabs=code_task_vocabs,
        method=method, method_ast=method_ast,
        sub_ast_root_indices_to_mask=sub_ast_root_indices_to_mask)

    pdg_input_tensors = preprocess_pdg(
        model_hps=model_hps, code_task_vocabs=code_task_vocabs, method=method,
        method_ast=method_ast, method_pdg=method_pdg, pdg_nodes_to_mask=pdg_nodes_to_mask,
        sub_ast_root_indices_to_mask=sub_ast_root_indices_to_mask)

    method_tokenized_code_input_tensors = preprocess_method_code_tokens_seq(
        code_task_vocabs=code_task_vocabs, method=method, method_pdg=method_pdg,
        pdg_nodes_to_mask=pdg_nodes_to_mask)

    return MethodCodeInputTensors(
        method_hash=method.hash, identifiers=identifiers, symbols=symbols,
        method_tokenized_code=method_tokenized_code_input_tensors,
        pdg=pdg_input_tensors, ast=method_ast_input_tensors)


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
            for pp_example_as_bytes in pool.imap_unordered(
                    functools.partial(
                        catch_preprocess_limit_exceed_error,
                        pp_example_fn, model_hps, code_task_vocabs),
                    iterable=raw_extracted_examples_generator(raw_extracted_data_dir=raw_dataset_path)):
                assert pp_example_as_bytes is not None
                if isinstance(pp_example_as_bytes, list):
                    assert all(isinstance(pp_limitation, PreprocessLimitation) for pp_limitation in pp_example_as_bytes)
                    pass  # TODO: add to limit exceed statistics
                else:
                    with io.BytesIO(pp_example_as_bytes) as pp_example_as_bytes_io_stream:
                        pp_example_as_bytes_io_stream.seek(0)
                        pp_example = torch.load(pp_example_as_bytes_io_stream)
                        assert isinstance(pp_example, TensorsDataClass)
                        chunks_examples_writer.write_example(pp_example)

        chunks_examples_writer.close_last_written_chunk()
        chunks_examples_writer.enforce_no_further_chunks()
        print(f'Finished pre-processing data-fold: `{datafold.name}`.')
