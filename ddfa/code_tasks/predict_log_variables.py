import os
import json
import torch
import typing
import itertools
import numpy as np
import pickle as pkl
import torch.nn as nn
from collections import defaultdict
from warnings import warn
from torch.utils.data.dataset import Dataset, IterableDataset
from typing import NamedTuple, Iterable, Collection

from ddfa.ddfa_model_hyper_parameters import DDFAModelHyperParams
from ddfa.dataset_properties import DatasetProperties, DataFold
from ddfa.code_tasks.code_task_base import CodeTaskBase, CodeTaskProperties
from ddfa.code_data_structure_api import *
from ddfa.code_nn_modules.vocabulary import Vocabulary
from ddfa.code_nn_modules.expression_encoder import ExpressionEncoder
from ddfa.code_nn_modules.identifier_encoder import IdentifierEncoder
from ddfa.code_nn_modules.cfg_node_encoder import CFGNodeEncoder
from ddfa.code_nn_modules.symbols_decoder import SymbolsDecoder


__all__ = ['PredictLogVariablesTask', 'TaggedExample', 'ModelInput', 'LoggingCallsDataset']


class PredictLogVariablesTask(CodeTaskBase):
    def __init__(self, task_props: CodeTaskProperties):
        super(PredictLogVariablesTask, self).__init__(task_props)
        # TODO: extract relevant fields from `task_props` into some `PredictLogVariablesTaskProps`!

    def preprocess(self, model_hps: DDFAModelHyperParams, pp_data_path: str, raw_train_data_path: str,
                   raw_eval_data_path: Optional[str] = None, raw_test_data_path: Optional[str] = None):
        preprocess(model_hps=model_hps, pp_data_path=pp_data_path, raw_train_data_path=raw_train_data_path,
                   raw_eval_data_path=raw_eval_data_path, raw_test_data_path=raw_test_data_path)

    def build_model(self, model_hps: DDFAModelHyperParams, pp_data_path: str) -> nn.Module:
        vocabs = load_or_create_vocabs(model_hps=model_hps, pp_data_path=pp_data_path)
        return Model(model_hps=model_hps, vocabs=vocabs)

    def create_dataset(self, model_hps: DDFAModelHyperParams, dataset_props: DatasetProperties,
            datafold: DataFold, pp_data_path: str) -> Dataset:
        return LoggingCallsDataset(datafold=datafold, pp_data_path=pp_data_path)

    def build_loss_criterion(self, model_hps: DDFAModelHyperParams) -> nn.Module:
        return ModelLoss(model_hps=model_hps)


class ModelInput(NamedTuple):
    identifiers: torch.Tensor
    cfg_nodes_control_kind: torch.Tensor
    cfg_nodes_expressions: torch.Tensor
    cfg_edges: torch.Tensor
    cfg_edges_attrs: torch.Tensor
    identifiers_idxs_of_all_symbols: torch.Tensor
    logging_call_cfg_node_idx: torch.Tensor


class ModelOutput(NamedTuple):
    decoder_outputs: torch.Tensor
    all_symbols_encoding: torch.Tensor


class TaggedExample(NamedTuple):
    model_input: ModelInput
    target_symbols_idxs_used_in_logging_call: torch.Tensor


MAX_NR_DATA_DEPENDENCY_EDGES_BETWEEN_PDG_NODES = 6  # TODO: move to model hyper-parameters!
MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER = 6  # TODO: move to model hyper-parameters!
MAX_NR_TOKENS_IN_EXPRESSION = 200  # TODO: move to model hyper-parameters!


class Model(nn.Module):
    def __init__(self, model_hps: DDFAModelHyperParams, vocabs: 'Vocabs'):
        super(Model, self).__init__()
        self.model_hps = model_hps
        self.vocabs = vocabs
        self.identifier_encoder = IdentifierEncoder(sub_identifiers_vocab=vocabs.sub_identifiers)
        expression_encoder = ExpressionEncoder(
            tokens_vocab=vocabs.tokens, tokens_kinds_vocab=vocabs.tokens_kinds,
            identifier_encoder=self.identifier_encoder)
        self.cfg_node_encoder = CFGNodeEncoder(
            expression_encoder=expression_encoder, pdg_node_control_kinds_vocab=vocabs.pdg_node_control_kinds)
        self.encoder_decoder_inbetween_dense = nn.Linear(in_features=256, out_features=256)  # TODO: plug-in model hps
        self.symbols_decoder = SymbolsDecoder()

    def forward(self, x: ModelInput):
        encoded_identifiers = self.identifier_encoder(x.identifiers)
        encoded_cfg_nodes = self.cfg_node_encoder(
            encoded_identifiers=encoded_identifiers, cfg_nodes_expressions=x.cfg_nodes_expressions,
            cfg_nodes_control_kind=x.cfg_nodes_control_kind)
        all_symbols_encoding = encoded_identifiers[x.identifiers_idxs_of_all_symbols]

        final_cfg_node_vectors = encoded_cfg_nodes  # TODO: graph NN

        logging_call_cfg_node_vector = final_cfg_node_vectors[x.logging_call_cfg_node_idx]
        input_to_decoder = self.encoder_decoder_inbetween_dense(logging_call_cfg_node_vector)

        decoder_outputs = self.symbols_decoder(
            input_state=input_to_decoder, symbols_encodings=all_symbols_encoding)

        return ModelOutput(decoder_outputs=decoder_outputs, all_symbols_encoding=all_symbols_encoding)


class ModelLoss(nn.Module):
    def __init__(self, model_hps: DDFAModelHyperParams):
        super(ModelLoss, self).__init__()
        self.model_hps = model_hps
        self.criterion = nn.NLLLoss()  # TODO: decide what criterion to use based on model-hps.

    def forward(self, model_output: ModelOutput, target_symbols_idxs_used_in_logging_call: torch.Tensor):
        target_symbols_encodings = model_output.all_symbols_encoding[target_symbols_idxs_used_in_logging_call]
        return self.criterion(model_output.decoder_outputs, target_symbols_encodings)


class LoggingCallsDataset(IterableDataset):
    def __init__(self, datafold: DataFold, pp_data_path: str):
        self.datafold = datafold
        self.pp_data_path = pp_data_path

    def __iter__(self):
        # TODO: read correctly files splitted into chunks
        pp_data_filepath = os.path.join(self.pp_data_path, f'pp-{self.datafold.value.lower()}.pkl')
        with open(pp_data_filepath, 'br') as pp_data_file:
            while True:
                try:
                    example: TaggedExample = pkl.load(pp_data_file)
                    assert all(hasattr(example, field) for field in TaggedExample._fields)
                    assert all(hasattr(example.model_input, field) for field in ModelInput._fields)
                    assert all(isinstance(elem, torch.Tensor) for elem in example.model_input)
                    yield TaggedExample(
                        model_input=ModelInput(**example.model_input._asdict()),
                        target_symbols_idxs_used_in_logging_call=example.target_symbols_idxs_used_in_logging_call)
                except (EOFError, pkl.UnpicklingError):
                    break


def non_identifier_token_to_token_vocab_word(token: SerToken):
    assert token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR}
    if token.kind == SerTokenKind.KEYWORD:
        return f'kwrd_{token.text}'
    elif token.kind == SerTokenKind.OPERATOR:
        return f'op_{token.operator.value}'
    elif token.kind == SerTokenKind.SEPARATOR:
        return f'sep_{token.separator.value}'


def token_to_input_vector(token: SerToken, vocabs: 'Vocabs'):
    assert token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR, SerTokenKind.IDENTIFIER}
    if token.kind == SerTokenKind.IDENTIFIER:
        return [vocabs.tokens_kinds.get_word_idx_or_unk(token.kind.value),
                vocabs.tokens.get_word_idx_or_unk(non_identifier_token_to_token_vocab_word(token))]
    return [vocabs.tokens_kinds.get_word_idx_or_unk(token.kind.value),
            token.identifier_idx]


def truncate_and_pad(vector: Collection, max_length: int, pad_word: str = '<PAD>') -> Iterable:
    vector_truncated_len = min(len(vector), max_length)
    padding_len = max_length - vector_truncated_len
    return itertools.chain(itertools.islice(vector, max_length), (pad_word for _ in range(padding_len)))


def nullable_lists_concat(*args) -> list:
    ret_list = []
    for lst in args:
        if lst is not None:
            ret_list.extend(lst)
    return ret_list


def preprocess(model_hps: DDFAModelHyperParams, pp_data_path: str, raw_train_data_path: str,
               raw_eval_data_path: Optional[str] = None, raw_test_data_path: Optional[str] = None):
    vocabs = load_or_create_vocabs(
        model_hps=model_hps, pp_data_path=pp_data_path, raw_train_data_path=raw_train_data_path)
    datafolds = (
        (SerDataFoldType.TRAIN, raw_train_data_path),
        (SerDataFoldType.VALIDATION, raw_eval_data_path),
        (SerDataFoldType.TEST, raw_test_data_path))
    for datafold, raw_dataset_path in datafolds:
        # TODO: split file into chunks after it is too big
        pp_data_filepath = os.path.join(pp_data_path, f'pp_{datafold.value.lower()}.pkl')
        with open(pp_data_filepath, 'bw') as pp_data_file:
            for logging_call, _, method_pdg in \
                    _iterate_raw_logging_calls_examples(dataset_path=raw_dataset_path):
                pp_example = preprocess_example(
                    model_hps=model_hps, vocabs=vocabs, logging_call=logging_call, method_pdg=method_pdg)
                pkl.dump(pp_data_file, pp_example)


def preprocess_example(
        model_hps: DDFAModelHyperParams, vocabs: 'Vocabs',
        logging_call: SerLoggingCall, method_pdg: SerMethodPDG) -> TaggedExample:
    logging_call_pdg_node = method_pdg.pdg_nodes[logging_call.pdg_node_idx]
    identifiers = torch.tensor([
        [vocabs.sub_identifiers.get_word_idx_or_unk(sub_identifier_str)
         for sub_identifier_str in truncate_and_pad(sub_identifiers, MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER)]
        for sub_identifiers in method_pdg.sub_identifiers_by_idx])
    symbols_identifier_idxs = torch.tensor([
        symbol.identifier_idx
        for symbols_scope in method_pdg.symbols_scopes
        for symbol in symbols_scope.symbols])
    symbols_used_in_logging_call = logging_call_pdg_node.symbols_use_def_mut.use.must + \
                                   logging_call_pdg_node.symbols_use_def_mut.use.may
    target_symbols_idxs_used_in_logging_call = torch.tensor([
        symbol_ref.symbol_idx for symbol_ref in symbols_used_in_logging_call])
    cfg_nodes_control_kind = torch.tensor([
        vocabs.pdg_node_control_kinds.get_word_idx_or_unk(
            pdg_node.control_kind.value if pdg_node.idx != logging_call.pdg_node_idx else '<LOG_PRED>')
        for pdg_node in method_pdg.pdg_nodes])
    padding_expression = [[vocabs.tokens_kinds.get_word_idx_or_unk('<PAD>'),
                vocabs.tokens.get_word_idx_or_unk('<PAD>')]] * MAX_NR_TOKENS_IN_EXPRESSION
    cfg_nodes_expressions = torch.tensor([
        [token_to_input_vector(token, vocabs) for token in pdg_node.code.tokenized]
        if pdg_node.code is not None and pdg_node.idx != logging_call.pdg_node_idx else padding_expression
        for pdg_node in method_pdg.pdg_nodes])
    control_flow_edges = {}
    data_dependency_edges = defaultdict(set)
    for src_pdg_node in method_pdg.pdg_nodes:
        for edge in src_pdg_node.control_flow_out_edges:
            control_flow_edges[(src_pdg_node.idx, edge.pgd_node_idx)] = edge
    for src_pdg_node in method_pdg.pdg_nodes:
        for edge in src_pdg_node.data_dependency_out_edges:
            if src_pdg_node.idx == logging_call.pdg_node_idx or edge.pgd_node_idx == logging_call.pdg_node_idx:
                continue
            data_dependency_edges[(src_pdg_node.idx, edge.pgd_node_idx)].update(
                symbol.identifier_idx for symbol in edge.symbols)
    edges = list(set(control_flow_edges.keys()) | set(data_dependency_edges.keys()))
    cfg_edges = torch.tensor([
        [src_pdg_node_idx, dst_pdg_node_idx]
        for src_pdg_node_idx, dst_pdg_node_idx in edges])

    def build_edge_attrs_vector(edge_vertices) -> List[int]:
        control_flow_edge_attrs = [
            vocabs.pdg_node_control_kinds.get_word_idx_or_unk(
                control_flow_edges[edge_vertices].type.value)] \
            if edge_vertices in control_flow_edges else \
            [vocabs.pdg_node_control_kinds.get_word_idx_or_unk('<UNK>')]
        data_dependency_edge_attrs = list(truncate_and_pad(
            data_dependency_edges[edge_vertices], MAX_NR_DATA_DEPENDENCY_EDGES_BETWEEN_PDG_NODES, -1)) \
            if edge_vertices in data_dependency_edges else \
            ([-1] * MAX_NR_DATA_DEPENDENCY_EDGES_BETWEEN_PDG_NODES)
        return control_flow_edge_attrs + data_dependency_edge_attrs

    cfg_edges_attrs = torch.tensor([
        build_edge_attrs_vector(edge_vertices)
        for edge_vertices in edges])
    return TaggedExample(
        model_input=ModelInput(
            identifiers=identifiers,
            cfg_nodes_control_kind=cfg_nodes_control_kind,
            cfg_nodes_expressions=cfg_nodes_expressions,
            cfg_edges=cfg_edges,
            cfg_edges_attrs=cfg_edges_attrs,
            identifiers_idxs_of_all_symbols=symbols_identifier_idxs,
            logging_call_cfg_node_idx=torch.tensor(logging_call.pdg_node_idx)),
        target_symbols_idxs_used_in_logging_call=target_symbols_idxs_used_in_logging_call)


class Vocabs(NamedTuple):
    sub_identifiers: Vocabulary
    tokens: Vocabulary
    pdg_node_control_kinds: Vocabulary
    tokens_kinds: Vocabulary
    pdg_control_flow_edge_types: Vocabulary


def load_or_create_vocabs(
        model_hps: DDFAModelHyperParams, pp_data_path: str, raw_train_data_path: Optional[str] = None) -> Vocabs:
    vocabs_pad_unk_special_words = ('<PAD>', '<UNK>')

    sub_identifiers_carpus_generator = lambda: (
        sub_identifier
        for logging_call, method_ast, method_pdg
        in _iterate_raw_logging_calls_examples(dataset_path=raw_train_data_path)
        for identifier_as_sub_identifiers in method_pdg.sub_identifiers_by_idx
        for sub_identifier in identifier_as_sub_identifiers)
    sub_identifiers_vocab = Vocabulary.load_or_create(
        preprocessed_data_dir_path=pp_data_path, vocab_name='sub_identifiers',
        special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<EOI>',), min_word_freq=40,
        max_vocab_size_wo_specials=1000, carpus_generator=sub_identifiers_carpus_generator)

    tokens_carpus_generator = lambda: (
        non_identifier_token_to_token_vocab_word(token)
        for _, _, method_pdg
        in _iterate_raw_logging_calls_examples(dataset_path=raw_train_data_path)
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.code is not None
        for token in pdg_node.code.tokenized
        if token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR})
    tokens_vocab = Vocabulary.load_or_create(
        preprocessed_data_dir_path=pp_data_path, vocab_name='tokens',
        special_words_sorted_by_idx=vocabs_pad_unk_special_words, min_word_freq=200,
        carpus_generator=tokens_carpus_generator)

    pdg_node_control_kinds_carpus_generator = lambda: (
        pdg_node.control_kind.value
        for _, _, method_pdg
        in _iterate_raw_logging_calls_examples(dataset_path=raw_train_data_path)
        for pdg_node in method_pdg.pdg_nodes)
    pdg_node_control_kinds_vocab = Vocabulary.load_or_create(
        preprocessed_data_dir_path=pp_data_path, vocab_name='pdg_node_control_kinds',
        special_words_sorted_by_idx=vocabs_pad_unk_special_words + ('<LOG_PRED>',), min_word_freq=200,
        carpus_generator=pdg_node_control_kinds_carpus_generator)

    tokens_kinds_carpus_generator = lambda: (
        token.kind.value
        for _, _, method_pdg
        in _iterate_raw_logging_calls_examples(dataset_path=raw_train_data_path)
        for pdg_node in method_pdg.pdg_nodes
        if pdg_node.code is not None
        for token in pdg_node.code.tokenized)
    tokens_kinds_vocab = Vocabulary.load_or_create(
        preprocessed_data_dir_path=pp_data_path, vocab_name='tokens_kinds',
        special_words_sorted_by_idx=vocabs_pad_unk_special_words, min_word_freq=200,
        carpus_generator=tokens_kinds_carpus_generator)

    pdg_control_flow_edge_types_carpus_generator = lambda: (
        edge.type.value
        for _, _, method_pdg
        in _iterate_raw_logging_calls_examples(dataset_path=raw_train_data_path)
        for pdg_node in method_pdg.pdg_nodes
        for edge in pdg_node.control_flow_out_edges)
    pdg_control_flow_edge_types_vocab = Vocabulary.load_or_create(
        preprocessed_data_dir_path=pp_data_path, vocab_name='pdg_control_flow_edge_types',
        special_words_sorted_by_idx=vocabs_pad_unk_special_words, min_word_freq=200,
        carpus_generator=pdg_control_flow_edge_types_carpus_generator)

    return Vocabs(
        sub_identifiers=sub_identifiers_vocab, tokens=tokens_vocab,
        pdg_node_control_kinds=pdg_node_control_kinds_vocab, tokens_kinds=tokens_kinds_vocab,
        pdg_control_flow_edge_types=pdg_control_flow_edge_types_vocab)


def _iterate_raw_logging_calls_examples(dataset_path: str) \
        -> typing.Iterable[typing.Tuple[SerLoggingCall, SerMethodAST, SerMethodPDG]]:
    with open(os.path.join(dataset_path, 'logging_calls_json.txt')) as logging_call_file, \
            open(os.path.join(dataset_path, 'method_ast.txt')) as method_ast_file, \
            open(os.path.join(dataset_path, 'method_pdg.txt')) as method_pdg_file:

        for logging_call_json, method_ast_json, method_pdg_json in \
                zip(logging_call_file, method_ast_file, method_pdg_file):
            logging_call_dict = json.loads(logging_call_json.strip())
            logging_call = SerLoggingCall.from_dict(logging_call_dict)
            method_ast_dict = json.loads(method_ast_json.strip())
            method_ast = SerMethodAST.from_dict(method_ast_dict)
            method_pdg_dict = json.loads(method_pdg_json.strip())
            method_pdg = SerMethodPDG.from_dict(method_pdg_dict)

            if logging_call.pdg_node_idx is None:
                warn(f'LoggingCall [{logging_call.hash}] has no PDG node.')
                continue
            assert logging_call.pdg_node_idx < len(method_pdg.pdg_nodes)
            if logging_call.ast_node_idx is None:
                warn(f'LoggingCall [{logging_call.hash}] has no AST node.')
                continue
            assert logging_call.ast_node_idx < len(method_ast.nodes)

            yield logging_call, method_ast, method_pdg
