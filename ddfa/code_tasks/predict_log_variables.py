import torch
import typing
import itertools
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from warnings import warn
from torch.utils.data.dataset import Dataset
from typing import NamedTuple, Iterable, Collection

from ddfa.ddfa_model_hyper_parameters import DDFAModelHyperParams
from ddfa.dataset_properties import DatasetProperties, DataFold
from ddfa.code_tasks.code_task_base import CodeTaskBase, CodeTaskProperties, EvaluationMetric
from ddfa.code_tasks.symbols_set_evaluation_metric import SymbolsSetEvaluationMetric
from ddfa.misc.code_data_structure_api import *
from ddfa.misc.iter_raw_extracted_data_files import iter_raw_extracted_examples_and_verify
from ddfa.misc.chunks_kvstore_dataset import ChunksKVStoreDatasetWriter, ChunksKVStoresDataset
from ddfa.code_nn_modules.code_task_vocabs import CodeTaskVocabs, non_identifier_token_to_token_vocab_word
from ddfa.code_nn_modules.code_task_encoder import CodeTaskEncoder, EncodedCode
from ddfa.code_nn_modules.symbols_decoder import SymbolsDecoder
from ddfa.code_nn_modules.code_task_input import CodeTaskInput


__all__ = ['PredictLogVariablesTask', 'TaggedExample', 'LoggingCallsTaskDataset', 'ModelInput']


class PredictLogVariablesTask(CodeTaskBase):
    def __init__(self, task_props: CodeTaskProperties):
        super(PredictLogVariablesTask, self).__init__(task_props)
        # TODO: extract relevant fields from `task_props` into some `PredictLogVariablesTaskProps`!

    def preprocess(self, model_hps: DDFAModelHyperParams, pp_data_path: str, raw_train_data_path: str,
                   raw_eval_data_path: Optional[str] = None, raw_test_data_path: Optional[str] = None):
        preprocess(model_hps=model_hps, pp_data_path=pp_data_path, raw_train_data_path=raw_train_data_path,
                   raw_eval_data_path=raw_eval_data_path, raw_test_data_path=raw_test_data_path)

    def build_model(self, model_hps: DDFAModelHyperParams, pp_data_path: str) -> nn.Module:
        vocabs = CodeTaskVocabs.load_or_create(model_hps=model_hps, pp_data_path=pp_data_path)
        return PredictLoggingCallVarsTaskModel(model_hps=model_hps, code_task_vocabs=vocabs)

    def create_dataset(self, model_hps: DDFAModelHyperParams, dataset_props: DatasetProperties,
            datafold: DataFold, pp_data_path: str) -> Dataset:
        return LoggingCallsTaskDataset(datafold=datafold, pp_data_path=pp_data_path)

    def build_loss_criterion(self, model_hps: DDFAModelHyperParams) -> nn.Module:
        return ModelLoss(model_hps=model_hps)

    def collate_examples(self, examples: List['TaggedExample']):
        assert all(isinstance(example, TaggedExample) for example in examples)
        return TaggedExample(
            code_task_input=CodeTaskInput.collate(list(example.code_task_input for example in examples)),
            target_symbols_idxs_used_in_logging_call=torch.cat(
                tuple(example.target_symbols_idxs_used_in_logging_call.unsqueeze(0) for example in examples), dim=0))

    def evaluation_metrics(self) -> List[Type[EvaluationMetric]]:
        return [LoggingCallTaskEvaluationMetric]


class LoggingCallTaskEvaluationMetric(SymbolsSetEvaluationMetric):
    def update(self, y_hat: 'ModelOutput', target: torch.Tensor):
        batch_target_symbols_indices = target
        _, batch_pred_symbols_indices = y_hat.decoder_outputs.topk(k=1, dim=-1)
        batch_pred_symbols_indices = batch_pred_symbols_indices.squeeze(dim=-1)
        assert len(batch_pred_symbols_indices.size()) == len(batch_target_symbols_indices.size()) == 2
        assert batch_pred_symbols_indices.size()[0] == batch_target_symbols_indices.size()[0]  # bsz
        assert batch_pred_symbols_indices.size()[1] + 1 == batch_target_symbols_indices.size()[
            1]  # seq_len; prefix `<SOS>` is not predicted.
        batch_pred_symbols_indices = batch_pred_symbols_indices.numpy()
        batch_target_symbols_indices = batch_target_symbols_indices.numpy()
        nr_symbols_special_words = 3  # TODO: pass it to `__init__()` cleanly
        batch_size = batch_target_symbols_indices.shape[0]
        for example_idx_in_batch in range(batch_size):
            example_pred_symbols_indices = [
                symbol_idx for symbol_idx in batch_pred_symbols_indices[example_idx_in_batch, :]
                if symbol_idx >= nr_symbols_special_words]
            example_target_symbols_indices = [
                symbol_idx for symbol_idx in batch_target_symbols_indices[example_idx_in_batch, :]
                if symbol_idx >= nr_symbols_special_words]
            super(LoggingCallTaskEvaluationMetric, self).update(
                example_pred_symbols_indices=example_pred_symbols_indices,
                example_target_symbols_indices=example_target_symbols_indices)


# TODO: remove! after next pp.. (it is here only to support old preprocessed files)
class ModelInput(CodeTaskInput):
    pass


class ModelOutput(NamedTuple):
    decoder_outputs: torch.Tensor
    all_symbols_encodings: torch.Tensor

    def numpy(self):
        return ModelOutput(**{field: getattr(self, field).cpu().numpy() for field in ModelOutput._fields})

    def cpu(self):
        return ModelOutput(**{field: getattr(self, field).cpu() for field in ModelOutput._fields})


class TaggedExample(NamedTuple):
    code_task_input: CodeTaskInput
    target_symbols_idxs_used_in_logging_call: torch.Tensor

    def to(self, device):
        return TaggedExample(
            code_task_input=self.code_task_input.to(device),
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


class PredictLoggingCallVarsTaskModel(nn.Module):
    def __init__(self, model_hps: DDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs):
        super(PredictLoggingCallVarsTaskModel, self).__init__()
        self.model_hps = model_hps
        self.code_task_vocabs = code_task_vocabs
        self.identifier_embedding_dim = 256  # TODO: plug-in model hps
        self.expr_encoding_dim = 1028  # TODO: plug-in model hps

        self.code_task_encoder = CodeTaskEncoder(
            code_task_vocabs=code_task_vocabs,
            identifier_embedding_dim=self.identifier_embedding_dim,  # TODO: plug-in model hps
            expr_encoding_dim=self.expr_encoding_dim)  # TODO: plug-in model hps

        self.symbols_special_words_embedding = nn.Embedding(
            num_embeddings=len(self.code_task_vocabs.symbols_special_words),
            embedding_dim=self.identifier_embedding_dim,
            padding_idx=self.code_task_vocabs.symbols_special_words.get_word_idx_or_unk('<PAD>'))
        self.symbols_decoder = SymbolsDecoder(
            symbols_special_words_embedding=self.symbols_special_words_embedding,
            symbols_special_words_vocab=self.code_task_vocabs.symbols_special_words,
            max_nr_taget_symbols=MAX_NR_TARGET_SYMBOLS + 2, encoder_output_len=MAX_NR_PDG_NODES,
            encoder_output_dim=self.code_task_encoder.cfg_node_encoder.output_dim,
            symbols_encoding_dim=self.identifier_embedding_dim)
        self.dbg__tensors_to_check_grads = {}

    def forward(self, code_task_input: CodeTaskInput, target_symbols_idxs_used_in_logging_call: Optional[torch.IntTensor] = None):
        self.dbg__tensors_to_check_grads = {}

        encoded_code: EncodedCode = self.code_task_encoder(code_task_input=code_task_input)
        self.dbg__tensors_to_check_grads['encoded_identifiers'] = encoded_code.encoded_identifiers
        self.dbg__tensors_to_check_grads['encoded_cfg_nodes'] = encoded_code.encoded_cfg_nodes
        self.dbg__tensors_to_check_grads['all_symbols_encodings'] = encoded_code.all_symbols_encodings
        self.dbg__tensors_to_check_grads['encoded_cfg_nodes_after_bridge'] = encoded_code.encoded_cfg_nodes_after_bridge

        decoder_outputs = self.symbols_decoder(
            encoder_outputs=encoded_code.encoded_cfg_nodes_after_bridge,
            encoder_outputs_mask=code_task_input.cfg_nodes_mask,
            symbols_encodings=encoded_code.all_symbols_encodings,
            symbols_encodings_mask=code_task_input.identifiers_idxs_of_all_symbols_mask,
            target_symbols_idxs=target_symbols_idxs_used_in_logging_call)
        self.dbg__tensors_to_check_grads['decoder_outputs'] = decoder_outputs

        return ModelOutput(decoder_outputs=decoder_outputs, all_symbols_encodings=encoded_code.all_symbols_encodings)

    def dbg_test_grads(self, grad_test_example_idx: int = 3, isclose_atol=1e-07):
        assert all(tensor.grad.size() == tensor.size() for tensor in self.dbg__tensors_to_check_grads.values())
        assert all(tensor.grad.size()[0] == next(iter(self.dbg__tensors_to_check_grads.values())).grad.size()[0]
                   for tensor in self.dbg__tensors_to_check_grads.values())
        for name, tensor in self.dbg__tensors_to_check_grads.items():
            print(f'Checking tensor `{name}` of shape {tensor.size()}:')
            # print(tensor.grad.cpu())
            grad = tensor.grad.cpu()
            batch_size = grad.size()[0]
            if not all(grad[example_idx].allclose(torch.tensor(0.0), atol=isclose_atol) for example_idx in range(batch_size) if example_idx != grad_test_example_idx):
                print(f'>>>>>>>> FAIL: Not all examples != #{grad_test_example_idx} has zero grad for tensor `{name}` of shape {tensor.size()}:')
                print(grad)
                for example_idx in range(batch_size):
                    print(f'non-zero places in grad for example #{example_idx}:')
                    print(grad[example_idx].isclose(torch.tensor(0.0), atol=isclose_atol))
                    print(grad[example_idx][~grad[example_idx].isclose(torch.tensor(0.0), atol=isclose_atol)])
            else:
                print(f'Success: All examples != #{grad_test_example_idx} has zero grad for tensor `{name}` of shape {tensor.size()}:')
            if grad[grad_test_example_idx].allclose(torch.tensor(0.0), atol=isclose_atol):
                print(f'>>>>>>>> FAIL: Tensor `{name}` of shape {tensor.size()} for example #{grad_test_example_idx} is all zero:')
                print(grad)
            else:
                print(f'Success: Tensor `{name}` of shape {tensor.size()} for example #{grad_test_example_idx} is not all zero.')
                if len(tensor.size()) > 1:
                    sub_objects_iszero = torch.tensor(
                        [grad[grad_test_example_idx, sub_object_idx].allclose(torch.tensor(0.0), atol=isclose_atol)
                         for sub_object_idx in range(tensor.size()[1])])
                    print(f'sub_objects_nonzero: {~sub_objects_iszero} (sum: {(~sub_objects_iszero).sum()})')
            print()

    def dbg_retain_grads(self):
        for tensor in self.dbg__tensors_to_check_grads.values():
            tensor.retain_grad()


class ModelLoss(nn.Module):
    def __init__(self, model_hps: DDFAModelHyperParams):
        super(ModelLoss, self).__init__()
        self.model_hps = model_hps
        self.criterion = nn.NLLLoss()  # TODO: decide what criterion to use based on model-hps.
        self.dbg__example_idx_to_test_grads = 3

    def dbg_forward_test_grads(self, model_output: ModelOutput, target_symbols_idxs: torch.LongTensor):
        # Change this to be the forward() when debugging gradients.
        assert len(model_output.decoder_outputs.size()) == 3  # (bsz, nr_target_symbols-1, max_nr_possible_symbols)
        assert len(target_symbols_idxs.size()) == 2  # (bsz, nr_target_symbols)
        assert model_output.decoder_outputs.size()[0] == target_symbols_idxs.size()[0]  # bsz
        assert model_output.decoder_outputs.size()[1] + 1 == target_symbols_idxs.size()[1]  # nr_target_symbols
        assert target_symbols_idxs.dtype == torch.long
        return model_output.decoder_outputs[self.dbg__example_idx_to_test_grads, :, :].sum()

    def forward(self, model_output: ModelOutput, target_symbols_idxs: torch.LongTensor):
        assert len(model_output.decoder_outputs.size()) == 3  # (bsz, nr_target_symbols-1, max_nr_possible_symbols)
        assert len(target_symbols_idxs.size()) == 2  # (bsz, nr_target_symbols)
        assert model_output.decoder_outputs.size()[0] == target_symbols_idxs.size()[0]  # bsz
        assert model_output.decoder_outputs.size()[1] + 1 == target_symbols_idxs.size()[1]  # nr_target_symbols
        assert target_symbols_idxs.dtype == torch.long
        return self.criterion(model_output.decoder_outputs.flatten(0, 1), target_symbols_idxs[:, 1:].flatten(0, 1))


class LoggingCallsTaskDataset(ChunksKVStoresDataset):
    def __init__(self, datafold: DataFold, pp_data_path: str):
        super(LoggingCallsTaskDataset, self).__init__(datafold=datafold, pp_data_path=pp_data_path)
        # TODO: add hash of task props & model HPs to perprocessed file name.

    def __getitem__(self, idx):
        example = super(LoggingCallsTaskDataset, self).__getitem__(idx)
        assert all(hasattr(example, field) for field in TaggedExample._fields)
        assert all(hasattr(example.code_task_input, field) for field in CodeTaskInput._fields)
        assert all(isinstance(getattr(example.code_task_input, field), torch.Tensor)
                   for field in CodeTaskInput._fields if field != 'is_batched')
        return TaggedExample(
            code_task_input=CodeTaskInput(**example.code_task_input._asdict()),
            target_symbols_idxs_used_in_logging_call=example.target_symbols_idxs_used_in_logging_call)


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


def preprocess(model_hps: DDFAModelHyperParams, pp_data_path: str, raw_train_data_path: Optional[str] = None,
               raw_eval_data_path: Optional[str] = None, raw_test_data_path: Optional[str] = None):
    vocabs = CodeTaskVocabs.load_or_create(
        model_hps=model_hps, pp_data_path=pp_data_path, raw_train_data_path=raw_train_data_path)
    datafolds = (
        (DataFold.Train, raw_train_data_path),
        (DataFold.Validation, raw_eval_data_path),
        (DataFold.Test, raw_test_data_path))
    for datafold, raw_dataset_path in datafolds:
        if raw_dataset_path is None:
            continue
        # TODO: add hash of task props & model HPs to perprocessed file name.
        chunks_examples_writer = ChunksKVStoreDatasetWriter(
            pp_data_path=pp_data_path, datafold=datafold,
            max_chunk_size_in_bytes=ChunksKVStoreDatasetWriter.MB_IN_BYTES * 500)
        for example in iter_raw_extracted_examples_and_verify(raw_extracted_data_dir=raw_dataset_path):
            pp_example = preprocess_example(
                model_hps=model_hps, vocabs=vocabs, logging_call=example.logging_call, method_pdg=example.method_pdg)
            if pp_example is None:
                continue
            chunks_examples_writer.write_example(pp_example)

        chunks_examples_writer.close_last_written_chunk()
        chunks_examples_writer.enforce_no_further_chunks()


def preprocess_example(
        model_hps: DDFAModelHyperParams, vocabs: CodeTaskVocabs,
        logging_call: SerLoggingCall, method_pdg: SerMethodPDG) -> typing.Optional[TaggedExample]:
    logging_call_pdg_node = method_pdg.pdg_nodes[logging_call.pdg_node_idx]
    sub_identifiers_pad = [vocabs.sub_identifiers.get_word_idx_or_unk('<PAD>')] * MAX_NR_SUB_IDENTIFIERS_IN_IDENTIFIER
    if len(method_pdg.sub_identifiers_by_idx) > MAX_NR_IDENTIFIERS:
        return None
    if any(len(sub_identifiers_in_identifier) < 1 for sub_identifiers_in_identifier in method_pdg.sub_identifiers_by_idx):
        warn(f'Found logging call {logging_call.hash} with an empty identifier (no sub-identifiers). ignoring.')
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
        code_task_input=CodeTaskInput(
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
