import os
import json
import torch
import pickle
import typing
import shelve
import itertools
import functools
import numpy as np
import torch.nn as nn
from collections import defaultdict
from warnings import warn
from torch.utils.data.dataset import Dataset
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
from ddfa.nn_utils import apply_batched_embeddings


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

    def collate_examples(self, examples: List['TaggedExample']):
        assert all(isinstance(example, TaggedExample) for example in examples)
        assert all(not example.model_input.is_batched for example in examples)
        for field_name in ModelInput._fields:
            if field_name == 'is_batched':
                continue
            if any(getattr(example.model_input, field_name).size() != getattr(examples[0].model_input, field_name).size()
                   for example in examples):
                raise ValueError(f'Not all examples have the same tensor size for `{field_name}`. '
                                 f'sizes: {[getattr(example.model_input, field_name).size() for example in examples]}')
        return TaggedExample(
            model_input=ModelInput(
                **{field_name: torch.cat(
                    tuple(getattr(example.model_input, field_name).unsqueeze(0) for example in examples), dim=0)
                   for field_name in ModelInput._fields if field_name != 'is_batched'},
                is_batched=True),
            target_symbols_idxs_used_in_logging_call=torch.cat(
                tuple(example.target_symbols_idxs_used_in_logging_call.unsqueeze(0) for example in examples), dim=0))


class ModelInput(NamedTuple):
    identifiers: torch.LongTensor
    sub_identifiers_mask: torch.BoolTensor
    cfg_nodes_mask: torch.BoolTensor
    cfg_nodes_control_kind: torch.LongTensor
    cfg_nodes_expressions: torch.LongTensor
    cfg_nodes_expressions_mask: torch.BoolTensor
    cfg_edges: torch.LongTensor
    cfg_edges_mask: torch.BoolTensor
    cfg_edges_attrs: torch.LongTensor
    identifiers_idxs_of_all_symbols: torch.LongTensor
    identifiers_idxs_of_all_symbols_mask: torch.BoolTensor
    logging_call_cfg_node_idx: torch.LongTensor
    is_batched: bool = False

    @property
    def batch_size(self) -> int:
        assert self.is_batched and len(self.identifiers.size()) == 3
        return self.identifiers.size()[0]

    def to(self, device):
        return ModelInput(
            is_batched=self.is_batched,
            **{field_name: getattr(self, field_name).to(device)
               for field_name in self._fields if field_name != 'is_batched'})


class ModelOutput(NamedTuple):
    decoder_outputs: torch.Tensor
    all_symbols_encoding: torch.Tensor


class TaggedExample(NamedTuple):
    model_input: ModelInput
    target_symbols_idxs_used_in_logging_call: torch.Tensor

    def to(self, device):
        return TaggedExample(
            model_input=self.model_input.to(device),
            target_symbols_idxs_used_in_logging_call=self.target_symbols_idxs_used_in_logging_call.to(device))


MAX_NR_PDG_EDGES = 300  # TODO: move to model hyper-parameters!
MAX_NR_DATA_DEPENDENCY_EDGES_BETWEEN_PDG_NODES = 6  # TODO: move to model hyper-parameters!
MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER = 5  # TODO: move to model hyper-parameters!
MIN_NR_TARGET_SYMBOLS = 1  # TODO: move to model hyper-parameters!
MAX_NR_TARGET_SYMBOLS = 4  # TODO: move to model hyper-parameters!
MAX_NR_IDENTIFIERS = 110  # TODO: move to model hyper-parameters!
MAX_NR_SYMBOLS = 55  # TODO: move to model hyper-parameters!
MIN_NR_SYMBOLS = 2  # TODO: move to model hyper-parameters!
MAX_NR_TOKENS_IN_EXPRESSION = 30  # TODO: move to model hyper-parameters!
MAX_NR_PDG_NODES = 80  # TODO: move to model hyper-parameters!


class Model(nn.Module):
    def __init__(self, model_hps: DDFAModelHyperParams, vocabs: 'Vocabs'):
        super(Model, self).__init__()
        self.model_hps = model_hps
        self.vocabs = vocabs
        self.identifier_embedding_dim = 256  # TODO: plug-in model hps
        self.expr_encoding_dim = 1028  # TODO: plug-in model hps
        self.identifier_encoder = IdentifierEncoder(
            sub_identifiers_vocab=vocabs.sub_identifiers, embedding_dim=self.identifier_embedding_dim)
        expression_encoder = ExpressionEncoder(
            tokens_vocab=vocabs.tokens, tokens_kinds_vocab=vocabs.tokens_kinds,
            tokens_embedding_dim=self.identifier_embedding_dim, expr_encoding_dim=self.expr_encoding_dim)
        self.cfg_node_encoder = CFGNodeEncoder(
            expression_encoder=expression_encoder, pdg_node_control_kinds_vocab=vocabs.pdg_node_control_kinds)
        self.encoder_decoder_inbetween_dense_layers = [
            nn.Linear(in_features=self.cfg_node_encoder.output_dim, out_features=self.cfg_node_encoder.output_dim)]
        self.symbols_special_words_embedding = nn.Embedding(
            num_embeddings=len(self.vocabs.symbols_special_words), embedding_dim=self.identifier_embedding_dim)
        self.symbols_decoder = SymbolsDecoder(
            symbols_special_words_vocab=self.vocabs.symbols_special_words,
            symbols_special_words_embedding=self.symbols_special_words_embedding, input_len=MAX_NR_PDG_NODES,
            input_dim=self.cfg_node_encoder.output_dim, symbols_encoding_dim=self.identifier_embedding_dim)

    def forward(self, x: ModelInput, target_symbols_idxs_used_in_logging_call: Optional[torch.IntTensor]):
        # x = tagged_example.model_input
        # target_symbols_idxs_used_in_logging_call: Optional[torch.IntTensor] = tagged_example.target_symbols_idxs_used_in_logging_call
        encoded_identifiers = self.identifier_encoder(
            sub_identifiers_indices=x.identifiers,
            sub_identifiers_mask=x.sub_identifiers_mask)  # (batch_size, nr_identifiers, identifier_encoding_dim)
        encoded_cfg_nodes = self.cfg_node_encoder(
            encoded_identifiers=encoded_identifiers, cfg_nodes_expressions=x.cfg_nodes_expressions,
            cfg_nodes_expressions_mask=x.cfg_nodes_expressions_mask,
            cfg_nodes_control_kind=x.cfg_nodes_control_kind)  # (batch_size, nr_cfg_nodes, cfg_node_encoding_dim)

        symbol_pad_embed = self.symbols_special_words_embedding(
                self.vocabs.symbols_special_words.get_word_idx_or_unk('<PAD>', as_tensor=True)).view(-1)
        all_symbols_encodings = apply_batched_embeddings(
            batched_embeddings=encoded_identifiers, indices=x.identifiers_idxs_of_all_symbols,
            mask=x.identifiers_idxs_of_all_symbols_mask,
            padding_embedding_vector=symbol_pad_embed)  # (batch_size, nr_symbols, identifier_encoding_dim)
        assert all_symbols_encodings.size() == (x.batch_size, MAX_NR_SYMBOLS, self.identifier_embedding_dim)

        encoded_cfg_nodes_after_dense = encoded_cfg_nodes
        if self.encoder_decoder_inbetween_dense_layers:
            encoded_cfg_nodes_after_dense = functools.reduce(
                lambda last_res, cur_layer: cur_layer(last_res),
                self.encoder_decoder_inbetween_dense_layers,
                encoded_cfg_nodes.flatten(0, 1)).view(encoded_cfg_nodes.size()[:-1] + (-1,))

        # final_cfg_node_vectors = encoded_cfg_nodes  # TODO: graph NN
        # logging_call_cfg_node_vector = final_cfg_node_vectors[x.logging_call_cfg_node_idx]
        # input_to_decoder = self.encoder_decoder_inbetween_dense(logging_call_cfg_node_vector)

        decoder_outputs = self.symbols_decoder(
            encoder_outputs=encoded_cfg_nodes_after_dense,
            encoder_outputs_mask=x.cfg_nodes_mask,
            symbols_encodings=all_symbols_encodings,
            symbols_encodings_mask=x.identifiers_idxs_of_all_symbols_mask,
            target_symbols_idxs=target_symbols_idxs_used_in_logging_call)

        return ModelOutput(decoder_outputs=decoder_outputs, all_symbols_encoding=all_symbols_encodings)


class ModelLoss(nn.Module):
    def __init__(self, model_hps: DDFAModelHyperParams):
        super(ModelLoss, self).__init__()
        self.model_hps = model_hps
        self.criterion = nn.NLLLoss()  # TODO: decide what criterion to use based on model-hps.

    def forward(self, model_output: ModelOutput, target_symbols_idxs: torch.LongTensor):
        assert len(target_symbols_idxs.size()) == 2  # (batch_size, nr_target_symbols)
        assert target_symbols_idxs.dtype == torch.long
        return self.criterion(model_output.decoder_outputs.flatten(0, 1), target_symbols_idxs[:, 1:].flatten(0, 1))


class LoggingCallsDataset(Dataset):
    def __init__(self, datafold: DataFold, pp_data_path: str):
        self.datafold = datafold
        self.pp_data_path = pp_data_path
        self._pp_data_chunks_filepaths = []
        self._kvstore_chunks = []
        self._kvstore_chunks_lengths = []
        for chunk_idx in itertools.count():
            filepath = os.path.join(self.pp_data_path, f'pp_{self.datafold.value.lower()}.{chunk_idx}.pt')
            if not os.path.isfile(filepath):
                break
            self._pp_data_chunks_filepaths.append(filepath)
            kvstore = shelve.open(filepath, 'r')
            self._kvstore_chunks.append(kvstore)
            self._kvstore_chunks_lengths.append(kvstore['len'])
        self._len = sum(self._kvstore_chunks_lengths)
        self._kvstore_chunks_lengths = np.array(self._kvstore_chunks_lengths)
        self._kvstore_chunks_stop_indices = np.cumsum(self._kvstore_chunks_lengths)
        self._kvstore_chunks_start_indices = self._kvstore_chunks_stop_indices - self._kvstore_chunks_lengths
        # TODO: add hash of task props & model HPs to perprocessed file name.

    def _get_chunk_idx_contains_item(self, item_idx: int) -> int:
        assert item_idx < self._len
        cond = self._kvstore_chunks_start_indices <= item_idx & item_idx < self._kvstore_chunks_stop_indices
        assert np.sum(cond) == 1
        found_idx = np.where(cond)
        assert len(found_idx) == 1
        return int(found_idx[0])

    def __len__(self):
        return self._len

    def __del__(self):
        for chunk_kvstore in self._kvstore_chunks:
            chunk_kvstore.close()

    def __getitem__(self, idx):
        assert isinstance(idx, int) and idx <= self._len
        chunk_kvstore = self._kvstore_chunks[self._get_chunk_idx_contains_item(idx)]
        example = chunk_kvstore[str(idx)]
        assert all(hasattr(example, field) for field in TaggedExample._fields)
        assert all(hasattr(example.model_input, field) for field in ModelInput._fields)
        assert all(isinstance(getattr(example.model_input, field), torch.Tensor)
                   for field in ModelInput._fields if field != 'is_batched')
        return TaggedExample(
            model_input=ModelInput(**example.model_input._asdict()),
            target_symbols_idxs_used_in_logging_call=example.target_symbols_idxs_used_in_logging_call)


def non_identifier_token_to_token_vocab_word(token: SerToken):
    assert token.kind in {SerTokenKind.KEYWORD, SerTokenKind.OPERATOR, SerTokenKind.SEPARATOR}
    if token.kind == SerTokenKind.KEYWORD:
        return f'kwrd_{token.text}'
    elif token.kind == SerTokenKind.OPERATOR:
        return f'op_{token.operator.value}'
    elif token.kind == SerTokenKind.SEPARATOR:
        return f'sep_{token.separator.value}'


def token_to_input_vector(token: SerToken, vocabs: 'Vocabs'):
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


def nullable_lists_concat(*args) -> list:
    ret_list = []
    for lst in args:
        if lst is not None:
            ret_list.extend(lst)
    return ret_list


class ChunksExamplesWriter:
    KB_IN_BYTES = 1024
    MB_IN_BYTES = 1024 * 1024
    GB_IN_BYTES = 1024 * 1024 * 1024

    def __init__(self, pp_data_path: str, datafold: DataFold, max_chunk_size_in_bytes: int = GB_IN_BYTES):
        self.pp_data_path: str = pp_data_path
        self.datafold: DataFold = datafold
        self.max_chunk_size_in_bytes: int = max_chunk_size_in_bytes
        self.next_example_idx: int = 0
        self.cur_chunk_idx: Optional[int] = None
        self.cur_chunk_size_in_bytes: Optional[int] = None
        self.cur_chunk_nr_examples: Optional[int] = None
        self.cur_chunk_file: Optional[shelve.Shelf] = None
        self.cur_chunk_filepath: Optional[str] = None

    @property
    def total_nr_examples(self) -> int:
        return self.next_example_idx

    def write_example(self, example):
        example_size_in_bytes = len(pickle.dumps(example))
        chunk_file = self.get_cur_chunk_to_write_example_into(example_size_in_bytes)
        chunk_file[str(self.next_example_idx)] = example
        self.next_example_idx += 1
        self.cur_chunk_nr_examples += 1
        self.cur_chunk_size_in_bytes += example_size_in_bytes
        assert self.cur_chunk_size_in_bytes <= self.max_chunk_size_in_bytes

    def get_cur_chunk_to_write_example_into(self, example_size_in_bytes: int) -> shelve.Shelf:
        assert example_size_in_bytes < self.max_chunk_size_in_bytes
        if self.cur_chunk_file is None or self.cur_chunk_size_in_bytes + example_size_in_bytes >= self.max_chunk_size_in_bytes:
            if self.cur_chunk_idx is None:
                self.cur_chunk_idx = 0
            else:
                self.cur_chunk_idx += 1
                self.close_last_written_chunk()
            self.cur_chunk_filepath = self._get_chunk_filepath(self.cur_chunk_idx)
            if os.path.isfile(self.cur_chunk_filepath):
                if self.cur_chunk_idx == 0:
                    raise ValueError(f'Preprocessed file `{self.cur_chunk_filepath}` already exists. '
                                     f'Please choose another `--pp-data` path or manually delete it.')
                else:
                    warn(f'Overwriting existing preprocessed file `{self.cur_chunk_filepath}`.')
                    os.remove(self.cur_chunk_filepath)
            self.cur_chunk_file = shelve.open(self.cur_chunk_filepath, 'c')
            self.cur_chunk_size_in_bytes = 0
            self.cur_chunk_nr_examples = 0
        return self.cur_chunk_file

    def close_last_written_chunk(self):
        assert self.cur_chunk_nr_examples > 0
        self.cur_chunk_file['len'] = self.cur_chunk_nr_examples
        self.cur_chunk_file.close()
        self.cur_chunk_file = None

    def _get_chunk_filepath(self, chunk_idx: int) -> str:
        return os.path.join(self.pp_data_path, f'pp_{self.datafold.value.lower()}.{chunk_idx}.pt')

    def enforce_no_further_chunks(self):
        # Remove old extra file chunks
        for chunk_idx_to_remove in itertools.count(start=self.cur_chunk_idx + 1):
            chunk_filepath = self._get_chunk_filepath(chunk_idx_to_remove)
            if not os.path.isfile(chunk_filepath):
                break
            warn(f'Removing existing preprocessed file `{chunk_filepath}`.')
            os.remove(chunk_filepath)


def preprocess(model_hps: DDFAModelHyperParams, pp_data_path: str, raw_train_data_path: str,
               raw_eval_data_path: Optional[str] = None, raw_test_data_path: Optional[str] = None):
    vocabs = load_or_create_vocabs(
        model_hps=model_hps, pp_data_path=pp_data_path, raw_train_data_path=raw_train_data_path)
    datafolds = (
        (SerDataFoldType.TRAIN, raw_train_data_path),
        (SerDataFoldType.VALIDATION, raw_eval_data_path),
        (SerDataFoldType.TEST, raw_test_data_path))
    for datafold, raw_dataset_path in datafolds:
        if raw_dataset_path is None:
            continue
        # TODO: add hash of task props & model HPs to perprocessed file name.
        chunks_examples_writer = ChunksExamplesWriter(
            pp_data_path=pp_data_path, datafold=datafold,
            max_chunk_size_in_bytes=ChunksExamplesWriter.MB_IN_BYTES * 500)
        for logging_call, _, method_pdg in \
                _iterate_raw_logging_calls_examples(dataset_path=raw_dataset_path):
            pp_example = preprocess_example(
                model_hps=model_hps, vocabs=vocabs, logging_call=logging_call, method_pdg=method_pdg)
            if pp_example is None:
                continue
            chunks_examples_writer.write_example(pp_example)

        chunks_examples_writer.close_last_written_chunk()
        chunks_examples_writer.enforce_no_further_chunks()


def preprocess_example(
        model_hps: DDFAModelHyperParams, vocabs: 'Vocabs',
        logging_call: SerLoggingCall, method_pdg: SerMethodPDG) -> typing.Optional[TaggedExample]:
    logging_call_pdg_node = method_pdg.pdg_nodes[logging_call.pdg_node_idx]
    sub_identifiers_pad = [vocabs.sub_identifiers.get_word_idx_or_unk('<PAD>')] * MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER
    if len(method_pdg.sub_identifiers_by_idx) > MAX_NR_IDENTIFIERS:
        return None
    if len(method_pdg.pdg_nodes) > MAX_NR_PDG_NODES:
        return None
    if any(len(pdg_node.code.tokenized) > MAX_NR_TOKENS_IN_EXPRESSION
           for pdg_node in method_pdg.pdg_nodes if pdg_node.code is not None):
        return None
    all_symbols = [symbol for symbols_scope in method_pdg.symbols_scopes
                   for symbol in symbols_scope.symbols]
    nr_symbols = len(all_symbols)
    if nr_symbols > MAX_NR_SYMBOLS or nr_symbols < MIN_NR_SYMBOLS:
        return None
    nr_edges = sum(len(pdg_node.control_flow_out_edges) +
                   sum(len(edge.symbols) for edge in pdg_node.data_dependency_out_edges)
                   for pdg_node in method_pdg.pdg_nodes)
    if nr_edges > MAX_NR_PDG_EDGES:
        return None

    symbols_idxs_used_in_logging_call = list(
        set(symbol_ref.symbol_idx for symbol_ref in logging_call_pdg_node.symbols_use_def_mut.use.must) |
        set(symbol_ref.symbol_idx for symbol_ref in logging_call_pdg_node.symbols_use_def_mut.use.may))
    if len(symbols_idxs_used_in_logging_call) > MAX_NR_TARGET_SYMBOLS or \
            len(symbols_idxs_used_in_logging_call) < MIN_NR_TARGET_SYMBOLS:
        return None
    target_symbols_idxs_used_in_logging_call = torch.tensor(list(truncate_and_pad(
        [vocabs.symbols_special_words.get_word_idx_or_unk('<SOS>')] +
        [symbol_idx_wo_specials + len(vocabs.symbols_special_words)
         for symbol_idx_wo_specials in symbols_idxs_used_in_logging_call] +
        [vocabs.symbols_special_words.get_word_idx_or_unk('<EOS>')],
        max_length=MAX_NR_TARGET_SYMBOLS + 2,
        pad_word=vocabs.symbols_special_words.get_word_idx_or_unk('<PAD>'))), dtype=torch.long)

    identifiers = torch.tensor(
        [[vocabs.sub_identifiers.get_word_idx_or_unk(sub_identifier_str)
          for sub_identifier_str in truncate_and_pad(sub_identifiers, MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER)]
         for sub_identifiers in itertools.islice(method_pdg.sub_identifiers_by_idx, MAX_NR_IDENTIFIERS)] +
        [sub_identifiers_pad
         for _ in range(MAX_NR_IDENTIFIERS - min(len(method_pdg.sub_identifiers_by_idx), MAX_NR_IDENTIFIERS))],
        dtype=torch.long)
    sub_identifiers_mask = torch.tensor(
        [[1] * min(len(sub_identifiers), MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER) +
         [0] * (MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER - min(len(sub_identifiers), MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER))
         for sub_identifiers in itertools.islice(method_pdg.sub_identifiers_by_idx, MAX_NR_IDENTIFIERS)] +
        [[1] * MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER] *
        (MAX_NR_IDENTIFIERS - min(len(method_pdg.sub_identifiers_by_idx), MAX_NR_IDENTIFIERS)),
        dtype=torch.bool)
    symbols_identifier_idxs = torch.tensor(
        [symbol.identifier_idx for symbol in all_symbols] +
        ([0] * (MAX_NR_SYMBOLS - len(all_symbols))), dtype=torch.long)
    symbols_identifier_mask = torch.cat([
        torch.ones(nr_symbols, dtype=torch.bool),
        torch.zeros(MAX_NR_SYMBOLS - nr_symbols, dtype=torch.bool)])
    cfg_nodes_mask = torch.cat([
        torch.ones(len(method_pdg.pdg_nodes), dtype=torch.bool),
        torch.zeros(MAX_NR_PDG_NODES - len(method_pdg.pdg_nodes), dtype=torch.bool)])
    cfg_nodes_control_kind = torch.tensor(list(truncate_and_pad([
        vocabs.pdg_node_control_kinds.get_word_idx_or_unk(
            pdg_node.control_kind.value if pdg_node.idx != logging_call.pdg_node_idx else '<LOG_PRED>')
        for pdg_node in method_pdg.pdg_nodes], max_length=MAX_NR_PDG_NODES,
        pad_word=vocabs.pdg_node_control_kinds.get_word_idx_or_unk('<PAD>'))), dtype=torch.long)
    padding_expression = [[vocabs.tokens_kinds.get_word_idx_or_unk('<PAD>'),
                vocabs.tokens.get_word_idx_or_unk('<PAD>')]] * (MAX_NR_TOKENS_IN_EXPRESSION + 1)
    # TODO: add <EOS>
    cfg_nodes_expressions = torch.tensor(list(truncate_and_pad([
        list(truncate_and_pad(
            [token_to_input_vector(token, vocabs) for token in pdg_node.code.tokenized],
            max_length=MAX_NR_TOKENS_IN_EXPRESSION + 1, pad_word=padding_expression[0]))
        if pdg_node.code is not None and pdg_node.idx != logging_call.pdg_node_idx else padding_expression
        for pdg_node in method_pdg.pdg_nodes],
        max_length=MAX_NR_PDG_NODES, pad_word=padding_expression)), dtype=torch.long)
    cfg_nodes_expressions_mask = torch.tensor(
        [([1] * min(len(pdg_node.code.tokenized), (MAX_NR_TOKENS_IN_EXPRESSION + 1)) +
         [0] * ((MAX_NR_TOKENS_IN_EXPRESSION + 1) - min(len(pdg_node.code.tokenized), (MAX_NR_TOKENS_IN_EXPRESSION + 1))))
         if pdg_node.code is not None else [1] * (MAX_NR_TOKENS_IN_EXPRESSION + 1)
         for pdg_node in itertools.islice(method_pdg.pdg_nodes, MAX_NR_PDG_NODES)] +
        [[1] * (MAX_NR_TOKENS_IN_EXPRESSION + 1)] *
        (MAX_NR_PDG_NODES - min(len(method_pdg.pdg_nodes), MAX_NR_PDG_NODES)),
        dtype=torch.bool)
    cfg_edges_mask = torch.cat([
        torch.ones(nr_edges, dtype=torch.bool),
        torch.zeros(MAX_NR_PDG_EDGES - nr_edges, dtype=torch.bool)])
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
    cfg_edges = torch.tensor(list(truncate_and_pad([
        [src_pdg_node_idx, dst_pdg_node_idx]
        for src_pdg_node_idx, dst_pdg_node_idx in edges],
        max_length=MAX_NR_PDG_EDGES, pad_word=[-1, -1])), dtype=torch.long)

    pad_edge_attrs_vector = [vocabs.pdg_control_flow_edge_types.get_word_idx_or_unk('<PAD>')] + \
                            ([-1] * MAX_NR_DATA_DEPENDENCY_EDGES_BETWEEN_PDG_NODES)
    def build_edge_attrs_vector(edge_vertices) -> List[int]:
        control_flow_edge_attrs = [
            vocabs.pdg_control_flow_edge_types.get_word_idx_or_unk(
                control_flow_edges[edge_vertices].type.value)] \
            if edge_vertices in control_flow_edges else \
            [vocabs.pdg_control_flow_edge_types.get_word_idx_or_unk('<UNK>')]
        data_dependency_edge_attrs = list(truncate_and_pad(
            data_dependency_edges[edge_vertices], MAX_NR_DATA_DEPENDENCY_EDGES_BETWEEN_PDG_NODES, -1)) \
            if edge_vertices in data_dependency_edges else \
            ([-1] * MAX_NR_DATA_DEPENDENCY_EDGES_BETWEEN_PDG_NODES)
        return control_flow_edge_attrs + data_dependency_edge_attrs

    cfg_edges_attrs = torch.tensor(list(truncate_and_pad([
        build_edge_attrs_vector(edge_vertices)
        for edge_vertices in edges], max_length=MAX_NR_PDG_EDGES, pad_word=pad_edge_attrs_vector)), dtype=torch.long)

    return TaggedExample(
        model_input=ModelInput(
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
            logging_call_cfg_node_idx=torch.tensor(logging_call.pdg_node_idx)),
        target_symbols_idxs_used_in_logging_call=target_symbols_idxs_used_in_logging_call)


class Vocabs(NamedTuple):
    sub_identifiers: Vocabulary
    tokens: Vocabulary
    pdg_node_control_kinds: Vocabulary
    tokens_kinds: Vocabulary
    pdg_control_flow_edge_types: Vocabulary
    symbols_special_words: Vocabulary


def load_or_create_vocabs(
        model_hps: DDFAModelHyperParams, pp_data_path: str, raw_train_data_path: Optional[str] = None) -> Vocabs:
    print('Loading / creating vocabularies ..')
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

    symbols_special_words_vocab = Vocabulary(
        name='symbols-specials', all_words_sorted_by_idx=[], params=(),
        special_words_sorted_by_idx=('<PAD>', '<SOS>', '<EOS>'))

    print('Done loading / creating vocabularies.')

    return Vocabs(
        sub_identifiers=sub_identifiers_vocab,
        tokens=tokens_vocab,
        pdg_node_control_kinds=pdg_node_control_kinds_vocab,
        tokens_kinds=tokens_kinds_vocab,
        pdg_control_flow_edge_types=pdg_control_flow_edge_types_vocab,
        symbols_special_words=symbols_special_words_vocab)


def _iterate_raw_logging_calls_examples(dataset_path: str) \
        -> typing.Iterable[typing.Tuple[SerLoggingCall, SerMethodAST, SerMethodPDG]]:
    with open(os.path.join(dataset_path, 'logging_calls_json.txt')) as logging_call_file, \
            open(os.path.join(dataset_path, 'method_ast.txt')) as method_ast_file, \
            open(os.path.join(dataset_path, 'method_pdg.txt')) as method_pdg_file:

        for example_idx, (logging_call_json, method_ast_json, method_pdg_json) in \
                enumerate(zip(logging_call_file, method_ast_file, method_pdg_file)):
            logging_call_dict = json.loads(logging_call_json.strip())
            logging_call = SerLoggingCall.from_dict(logging_call_dict)
            method_ast_dict = json.loads(method_ast_json.strip())
            method_ast = SerMethodAST.from_dict(method_ast_dict)
            method_pdg_dict = json.loads(method_pdg_json.strip())
            method_pdg = SerMethodPDG.from_dict(method_pdg_dict)
            if method_ast.method_hash != logging_call.method_ref.hash:
                raise ValueError(f'Error while reading raw data @ line #{example_idx + 1}:'
                                 f'logging_call.method_ref.hash={logging_call.method_ref.hash},'
                                 f' while method_pdg.method_hash={method_ast.method_hash}')
            if method_pdg.method_hash != logging_call.method_ref.hash:
                raise ValueError(f'Error while reading raw data @ line #{example_idx + 1}:'
                                 f'logging_call.method_ref.hash={logging_call.method_ref.hash},'
                                 f' while method_pdg.method_hash={method_pdg.method_hash}')

            if logging_call.pdg_node_idx is None:
                # warn(f'LoggingCall [{logging_call.hash}] has no PDG node.')
                continue
            assert logging_call.pdg_node_idx < len(method_pdg.pdg_nodes)
            if logging_call.ast_node_idx is None:
                warn(f'LoggingCall [{logging_call.hash}] has no AST node.')
                continue
            assert logging_call.ast_node_idx < len(method_ast.nodes)

            yield logging_call, method_ast, method_pdg
