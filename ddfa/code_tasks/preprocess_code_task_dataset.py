import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from warnings import warn
from typing import Iterable, Collection, Any, Set, Optional, Dict, List
from typing_extensions import Protocol

from ddfa.ddfa_model_hyper_parameters import DDFAModelHyperParams
from ddfa.dataset_properties import DataFold
from ddfa.misc.code_data_structure_api import SerMethod, SerMethodPDG, SerMethodAST, SerToken, SerTokenKind, SerPDGNode
from ddfa.misc.chunks_kvstore_dataset import ChunksKVStoreDatasetWriter
from ddfa.code_nn_modules.code_task_vocabs import CodeTaskVocabs, non_identifier_token_to_token_vocab_word
from ddfa.code_nn_modules.code_task_input import MethodCodeInputToEncoder


__all__ = [
    'preprocess_code_task_dataset', 'preprocess_code_task_example', 'truncate_and_pad', 'PreprocessLimitExceedError']


def token_to_input_vector(token: SerToken, vocabs: CodeTaskVocabs):
    assert token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR,
                          SerTokenKind.IDENTIFIER, SerTokenKind.LITERAL}
    if token.kind == SerTokenKind.IDENTIFIER:
        return [vocabs.tokens_kinds.get_word_idx_or_unk(token.kind.value),
                token.identifier_idx]
    if token.kind == SerTokenKind.LITERAL:
        return [vocabs.tokens_kinds.get_word_idx_or_unk(token.kind.value),
            vocabs.tokens.get_word_idx_or_unk('<PAD>')]  # TODO: add some '<NON-RELEVANT>' special word
    return [vocabs.tokens_kinds.get_word_idx_or_unk(token.kind.value),
            vocabs.tokens.get_word_idx_or_unk(non_identifier_token_to_token_vocab_word(token))]


def truncate_and_pad(vector: Collection, max_length: int, pad_word: str = '<PAD>') -> Iterable:
    vector_truncated_len = min(len(vector), max_length)
    padding_len = max_length - vector_truncated_len
    return itertools.chain(itertools.islice(vector, max_length), (pad_word for _ in range(padding_len)))


class PreprocessLimitExceedError(ValueError):
    pass


def get_pdg_node_tokenized_expression(method: SerMethod, pdg_node: SerPDGNode):
    return method.code.tokenized[
        pdg_node.code_sub_token_range_ref.begin_token_idx:
        pdg_node.code_sub_token_range_ref.end_token_idx+1]


def preprocess_code_task_example(
        model_hps: DDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs,
        method: SerMethod, method_pdg: SerMethodPDG, method_ast: SerMethodAST,
        remove_edges_from_pdg_nodes_idxs: Optional[Set[int]] = None,
        pdg_nodes_to_mask: Optional[Dict[int, str]] = None) -> Optional[MethodCodeInputToEncoder]:
    # todo raise exception with failure reason
    nr_identifiers = len(method_pdg.sub_identifiers_by_idx)
    if nr_identifiers > model_hps.method_code_encoder.max_nr_identifiers:
        raise PreprocessLimitExceedError(f'#identifiers ({nr_identifiers}) > MAX_NR_IDENTIFIERS ({model_hps.method_code_encoder.max_nr_identifiers})')
    if any(len(sub_identifiers_in_identifier) < 1 for sub_identifiers_in_identifier in method_pdg.sub_identifiers_by_idx):
        warn(f'Found method {method.hash} with an empty identifier (no sub-identifiers). ignoring.')
        raise PreprocessLimitExceedError(f'Empty identifier (no sub-identifiers)')
    if len(method_pdg.pdg_nodes) < model_hps.method_code_encoder.min_nr_pdg_nodes:
        raise PreprocessLimitExceedError(f'#pdg_nodes ({len(method_pdg.pdg_nodes)}) < MIN_NR_PDG_NODES ({model_hps.method_code_encoder.min_nr_pdg_nodes})')
    if len(method_pdg.pdg_nodes) > model_hps.method_code_encoder.max_nr_pdg_nodes:
        raise PreprocessLimitExceedError(f'#pdg_nodes ({len(method_pdg.pdg_nodes)}) > MAX_NR_PDG_NODES ({model_hps.method_code_encoder.max_nr_pdg_nodes})')
    if any(len(get_pdg_node_tokenized_expression(method, pdg_node)) > model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression
           for pdg_node in method_pdg.pdg_nodes if pdg_node.code_sub_token_range_ref is not None):
        raise PreprocessLimitExceedError(f'Too long tokenized expression for one of the PDG nodes.')
    all_symbols = [symbol for symbols_scope in method_pdg.symbols_scopes
                   for symbol in symbols_scope.symbols]
    nr_symbols = len(all_symbols)
    if nr_symbols < model_hps.method_code_encoder.min_nr_symbols:
        raise PreprocessLimitExceedError(f'#symbols ({nr_symbols}) < MIN_NR_SYMBOLS ({model_hps.method_code_encoder.min_nr_symbols})')
    if nr_symbols > model_hps.method_code_encoder.max_nr_symbols:
        raise PreprocessLimitExceedError(f'#symbols ({nr_symbols}) > MAX_NR_SYMBOLS ({model_hps.method_code_encoder.max_nr_symbols})')
    nr_edges = sum(len(pdg_node.control_flow_out_edges) +
                   sum(len(edge.symbols) for edge in pdg_node.data_dependency_out_edges)
                   for pdg_node in method_pdg.pdg_nodes)
    if nr_edges > model_hps.method_code_encoder.max_nr_pdg_edges:
        raise PreprocessLimitExceedError(f'#edges ({nr_edges}) > MAX_NR_PDG_EDGES ({model_hps.method_code_encoder.max_nr_pdg_edges})')

    sub_identifiers_pad = [code_task_vocabs.sub_identifiers.get_word_idx_or_unk(
        '<PAD>')] * model_hps.method_code_encoder.max_sub_identifier_vocab_size
    identifiers = torch.tensor(
        [[code_task_vocabs.sub_identifiers.get_word_idx_or_unk(sub_identifier_str)
          for sub_identifier_str in truncate_and_pad(sub_identifiers, model_hps.method_code_encoder.max_sub_identifier_vocab_size)]
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
        [symbol.identifier_idx for symbol in all_symbols] +
        ([0] * (model_hps.method_code_encoder.max_nr_symbols - len(all_symbols))), dtype=torch.long)
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
        pad_word=code_task_vocabs.pdg_node_control_kinds.get_word_idx_or_unk('<PAD>'))), dtype=torch.long)
    padding_expression = [[code_task_vocabs.tokens_kinds.get_word_idx_or_unk('<PAD>'),
                           code_task_vocabs.tokens.get_word_idx_or_unk('<PAD>')]] * (model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1)
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
         if pdg_node.code_sub_token_range_ref is not None else [1] * (model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1)
         for pdg_node in itertools.islice(method_pdg.pdg_nodes, model_hps.method_code_encoder.max_nr_pdg_nodes)] +
        [[1] * (model_hps.method_code_encoder.max_nr_tokens_in_pdg_node_expression + 1)] *
        (model_hps.method_code_encoder.max_nr_pdg_nodes - min(len(method_pdg.pdg_nodes), model_hps.method_code_encoder.max_nr_pdg_nodes)),
        dtype=torch.bool)
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

    pad_edge_attrs_vector = [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx_or_unk('<PAD>')] + \
                            ([-1] * model_hps.method_code_encoder.max_nr_pdg_data_dependency_edges_between_two_nodes)
    def build_edge_attrs_vector(edge_vertices) -> List[int]:
        control_flow_edge_attrs = [
            code_task_vocabs.pdg_control_flow_edge_types.get_word_idx_or_unk(
                control_flow_edges[edge_vertices].type.value)] \
            if edge_vertices in control_flow_edges else \
            [code_task_vocabs.pdg_control_flow_edge_types.get_word_idx_or_unk('<UNK>')]
        data_dependency_edge_attrs = list(truncate_and_pad(
            data_dependency_edges[edge_vertices], model_hps.method_code_encoder.max_nr_pdg_data_dependency_edges_between_two_nodes, -1)) \
            if edge_vertices in data_dependency_edges else \
            ([-1] * model_hps.method_code_encoder.max_nr_pdg_data_dependency_edges_between_two_nodes)
        return control_flow_edge_attrs + data_dependency_edge_attrs

    cfg_edges_attrs = torch.tensor(list(truncate_and_pad([
        build_edge_attrs_vector(edge_vertices)
        for edge_vertices in edges], max_length=model_hps.method_code_encoder.max_nr_pdg_edges, pad_word=pad_edge_attrs_vector)), dtype=torch.long)

    return MethodCodeInputToEncoder(
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
        identifiers_idxs_of_all_symbols_mask=symbols_identifier_mask)


class PPExampleFnType(Protocol):
    def __call__(self, model_hps: DDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs, example: Any) -> Any: ...


class RawExtractedExamplesGenerator(Protocol):
    def __call__(self, raw_extracted_data_dir: str) -> Iterable[Any]: ...


def preprocess_code_task_dataset(
        model_hps: DDFAModelHyperParams, pp_data_path: str,
        raw_extracted_examples_generator: RawExtractedExamplesGenerator, pp_example_fn: PPExampleFnType,
        raw_train_data_path: Optional[str] = None, raw_eval_data_path: Optional[str] = None,
        raw_test_data_path: Optional[str] = None):
    vocabs = CodeTaskVocabs.load_or_create(
        model_hps=model_hps, pp_data_path=pp_data_path, raw_train_data_path=raw_train_data_path)
    datafolds = (
        (DataFold.Train, raw_train_data_path),
        (DataFold.Validation, raw_eval_data_path),
        (DataFold.Test, raw_test_data_path))
    for datafold, raw_dataset_path in datafolds:
        if raw_dataset_path is None:
            continue
        print(f'Starting pre-processing data-fold: `{datafold.name}` ..')
        # TODO: add hash of task props & model HPs to perprocessed file name.
        # TODO: aggregate limit exceed statistics and print in the end.
        chunks_examples_writer = ChunksKVStoreDatasetWriter(
            pp_data_path=pp_data_path, datafold=datafold,
            max_chunk_size_in_bytes=ChunksKVStoreDatasetWriter.MB_IN_BYTES * 500)
        for example in raw_extracted_examples_generator(raw_extracted_data_dir=raw_dataset_path):
            try:
                pp_example = pp_example_fn(model_hps=model_hps, code_task_vocabs=vocabs, example=example)
                assert pp_example is not None
                chunks_examples_writer.write_example(pp_example)
            except PreprocessLimitExceedError as err:
                pass  # TODO: add to limit exceed statistics

        chunks_examples_writer.close_last_written_chunk()
        chunks_examples_writer.enforce_no_further_chunks()
        print(f'Finished pre-processing data-fold: `{datafold.name}`.')
