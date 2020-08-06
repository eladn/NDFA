import torch
import typing
import dataclasses
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams
from ndfa.dataset_properties import DatasetProperties, DataFold
from ndfa.code_tasks.code_task_base import CodeTaskBase
from ndfa.code_tasks.code_task_properties import CodeTaskProperties
from ndfa.code_tasks.evaluation_metric_base import EvaluationMetric
from ndfa.code_tasks.symbols_set_evaluation_metric import SymbolsSetEvaluationMetric
from ndfa.misc.code_data_structure_api import *
from ndfa.misc.iter_raw_extracted_data_files import iter_raw_extracted_examples_and_verify, RawExtractedExample
from ndfa.misc.chunks_kvstore_dataset import ChunkedRandomAccessDataset
from ndfa.misc.tensors_data_class import TensorsDataClass
from ndfa.code_nn_modules.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.method_code_encoder import MethodCodeEncoder, EncodedMethodCode
from ndfa.code_nn_modules.symbols_decoder import SymbolsDecoder
from ndfa.code_nn_modules.code_task_input import MethodCodeInputToEncoder
from ndfa.code_tasks.preprocess_code_task_dataset import preprocess_code_task_example, truncate_and_pad, \
    PreprocessLimitExceedError
from ndfa.nn_utils.dbg_test_grads import ModuleWithDbgTestGrads
from ndfa.misc.code_data_structure_utils import get_symbol_idxs_used_in_logging_call


__all__ = ['PredictLogVarsTask', 'PredictLogVarsTaggedExample', 'PredictLogVarsTaskDataset']


class PredictLogVarsTask(CodeTaskBase):
    def __init__(self, task_props: CodeTaskProperties):
        super(PredictLogVarsTask, self).__init__(task_props)
        # TODO: extract relevant fields from `task_props` into some `PredictLogVariablesTaskProps`!

    def iterate_raw_examples(self, model_hps: NDFAModelHyperParams, raw_extracted_data_dir: str) \
            -> typing.Iterable[RawExtractedExample]:
        return iter_raw_extracted_examples_and_verify(raw_extracted_data_dir=raw_extracted_data_dir)

    def preprocess_raw_example(
            self, model_hps: NDFAModelHyperParams,
            code_task_vocabs: CodeTaskVocabs,
            raw_example: Any,
            add_tag: bool = True) \
            -> 'PredictLogVarsTaggedExample':
        return preprocess_logging_call_example(
            model_hps=model_hps, code_task_vocabs=code_task_vocabs, raw_example=raw_example, add_tag=add_tag)

    def build_model(self, model_hps: NDFAModelHyperParams, pp_data_path: str) -> 'PredictLogVarsModel':
        vocabs = self.create_or_load_code_task_vocabs(model_hps=model_hps, pp_data_path=pp_data_path)
        return PredictLogVarsModel(model_hps=model_hps, code_task_vocabs=vocabs)

    def predict(
            self, model: 'PredictLogVarsModel',
            device: torch.device,
            raw_example: RawExtractedExample,
            pp_example: 'PredictLogVarsTaggedExample') -> Any:
        code_task_input = pp_example.code_task_input
        model.to(device)
        model.eval()
        code_task_input = MethodCodeInputToEncoder.collate([code_task_input])
        code_task_input = code_task_input.to(device)
        output: PredictLoggingCallVarsModelOutput = model(code_task_input=code_task_input)
        decoder_outputs = output.decoder_outputs.squeeze(dim=0)
        symbol_indices = decoder_outputs.argmax(dim=-1)
        EOS_symbol_special_word_idx = 2  # '<EOS>'
        symbol_indices = symbol_indices.cpu().tolist()
        symbol_indices = [symbol_idx - 3 for symbol_idx in symbol_indices if symbol_idx > 2]
        symbol_names = [raw_example.method_pdg.symbols[symbol_idx].symbol_name for symbol_idx in symbol_indices]
        return symbol_names

    def create_dataset(self, model_hps: NDFAModelHyperParams, dataset_props: DatasetProperties,
            datafold: DataFold, pp_data_path: str) -> Dataset:
        return PredictLogVarsTaskDataset(datafold=datafold, pp_data_path=pp_data_path)

    def build_loss_criterion(self, model_hps: NDFAModelHyperParams) -> nn.Module:
        return PredictLogVarsModelLoss(model_hps=model_hps)

    def collate_examples(self, examples: List['PredictLogVarsTaggedExample']):
        assert all(isinstance(example, PredictLogVarsTaggedExample) for example in examples)
        return PredictLogVarsTaggedExample.collate(examples)

    def evaluation_metrics(self, model_hps: NDFAModelHyperParams) -> List[Type[EvaluationMetric]]:
        class LoggingCallTaskEvaluationMetric_(LoggingCallTaskEvaluationMetric):
            def __init__(self):
                super(LoggingCallTaskEvaluationMetric_, self).__init__(
                    nr_symbols_special_words=3)  # TODO: get `nr_symbols_special_words` from `model_hps` !
        return [LoggingCallTaskEvaluationMetric_]

    def create_or_load_code_task_vocabs(
            self, model_hps: NDFAModelHyperParams,
            pp_data_path: str,
            raw_train_data_path: Optional[str] = None) -> CodeTaskVocabs:
        return CodeTaskVocabs.load_or_create(
            model_hps=model_hps, pp_data_path=pp_data_path, raw_train_data_path=raw_train_data_path)


class LoggingCallTaskEvaluationMetric(SymbolsSetEvaluationMetric):
    def __init__(self, nr_symbols_special_words: int):
        super(LoggingCallTaskEvaluationMetric, self).__init__()
        self.nr_symbols_special_words = nr_symbols_special_words

    def update(self, y_hat: 'PredictLoggingCallVarsModelOutput', target: torch.Tensor):
        batch_target_symbols_indices = target
        _, batch_pred_symbols_indices = y_hat.decoder_outputs.topk(k=1, dim=-1)
        batch_pred_symbols_indices = batch_pred_symbols_indices.squeeze(dim=-1)
        assert len(batch_pred_symbols_indices.size()) == len(batch_target_symbols_indices.size()) == 2
        assert batch_pred_symbols_indices.size()[0] == batch_target_symbols_indices.size()[0]  # bsz
        assert batch_pred_symbols_indices.size()[1] + 1 == batch_target_symbols_indices.size()[
            1]  # seq_len; prefix `<SOS>` is not predicted.
        batch_pred_symbols_indices = batch_pred_symbols_indices.numpy()
        batch_target_symbols_indices = batch_target_symbols_indices.numpy()
        batch_size = batch_target_symbols_indices.shape[0]
        for example_idx_in_batch in range(batch_size):
            example_pred_symbols_indices = [
                symbol_idx for symbol_idx in batch_pred_symbols_indices[example_idx_in_batch, :]
                if symbol_idx >= self.nr_symbols_special_words]
            example_target_symbols_indices = [
                symbol_idx for symbol_idx in batch_target_symbols_indices[example_idx_in_batch, :]
                if symbol_idx >= self.nr_symbols_special_words]
            super(LoggingCallTaskEvaluationMetric, self).update(
                example_pred_symbols_indices=example_pred_symbols_indices,
                example_target_symbols_indices=example_target_symbols_indices)


@dataclasses.dataclass
class PredictLoggingCallVarsModelOutput(TensorsDataClass):
    decoder_outputs: torch.Tensor
    all_symbols_encodings: torch.Tensor


@dataclasses.dataclass
class PredictLogVarsTaggedExample(TensorsDataClass):
    example_hash: str
    code_task_input: MethodCodeInputToEncoder
    target_symbols_idxs_used_in_logging_call: torch.Tensor

    def __iter__(self):  # To support unpacking into (x_batch, y_batch)
        yield self.code_task_input
        yield self.target_symbols_idxs_used_in_logging_call


class PredictLogVarsModel(nn.Module, ModuleWithDbgTestGrads):
    def __init__(self, model_hps: NDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs):
        super(PredictLogVarsModel, self).__init__()
        ModuleWithDbgTestGrads.__init__(self)
        self.model_hps = model_hps
        self.code_task_vocabs = code_task_vocabs
        self.identifier_embedding_dim = 256  # TODO: plug-in model hps
        self.expr_encoding_dim = 1028  # TODO: plug-in model hps

        self.code_task_encoder = MethodCodeEncoder(
            code_task_vocabs=code_task_vocabs,
            identifier_embedding_dim=self.identifier_embedding_dim,  # TODO: plug-in model hps
            expr_encoding_dim=self.expr_encoding_dim)  # TODO: plug-in model hps

        self.symbols_decoder = SymbolsDecoder(
            symbols_special_words_embedding=self.code_task_encoder.symbols_encoder.symbols_special_words_embedding,  # FIXME: might be problematic because 2 different modules hold this (both SymbolsEncoder and SymbolsDecoder).
            symbols_special_words_vocab=self.code_task_vocabs.symbols_special_words,
            max_nr_taget_symbols=model_hps.method_code_encoder.max_nr_target_symbols + 2,
            encoder_output_len=model_hps.method_code_encoder.max_nr_pdg_nodes,
            encoder_output_dim=self.code_task_encoder.cfg_node_encoder.output_dim,
            symbols_encoding_dim=self.identifier_embedding_dim)

    def forward(self, code_task_input: MethodCodeInputToEncoder, target_symbols_idxs_used_in_logging_call: Optional[torch.IntTensor] = None):
        self.dbg_log_new_fwd()

        encoded_code: EncodedMethodCode = self.code_task_encoder(code_task_input=code_task_input)
        self.dbg_log_tensor_during_fwd('encoded_identifiers', encoded_code.encoded_identifiers)
        self.dbg_log_tensor_during_fwd('encoded_cfg_nodes', encoded_code.encoded_cfg_nodes)
        self.dbg_log_tensor_during_fwd('all_symbols_encodings', encoded_code.encoded_symbols)
        self.dbg_log_tensor_during_fwd('encoded_cfg_nodes_after_bridge', encoded_code.encoded_cfg_nodes_after_bridge)

        decoder_outputs = self.symbols_decoder(
            encoder_outputs=encoded_code.encoded_cfg_nodes_after_bridge,
            encoder_outputs_mask=code_task_input.cfg_nodes_mask,
            symbols_encodings=encoded_code.encoded_symbols,
            symbols_encodings_mask=code_task_input.identifiers_idxs_of_all_symbols_mask,
            encoded_symbols_occurrences=encoded_code.encoded_symbols_occurrences,
            target_symbols_idxs=target_symbols_idxs_used_in_logging_call)
        self.dbg_log_tensor_during_fwd('decoder_outputs', decoder_outputs)

        return PredictLoggingCallVarsModelOutput(
            decoder_outputs=decoder_outputs,
            all_symbols_encodings=encoded_code.encoded_symbols)


class PredictLogVarsModelLoss(nn.Module):
    def __init__(self, model_hps: NDFAModelHyperParams):
        super(PredictLogVarsModelLoss, self).__init__()
        self.model_hps = model_hps
        self.criterion = nn.NLLLoss()  # TODO: decide what criterion to use based on model-hps.
        self.dbg__example_idx_to_test_grads = 3

    def dbg_forward_test_grads(self, model_output: PredictLoggingCallVarsModelOutput, target_symbols_idxs: torch.LongTensor):
        # Change this to be the forward() when debugging gradients.
        assert len(model_output.decoder_outputs.size()) == 3  # (bsz, nr_target_symbols-1, max_nr_possible_symbols)
        assert len(target_symbols_idxs.size()) == 2  # (bsz, nr_target_symbols)
        assert model_output.decoder_outputs.size()[0] == target_symbols_idxs.size()[0]  # bsz
        assert model_output.decoder_outputs.size()[1] + 1 == target_symbols_idxs.size()[1]  # nr_target_symbols
        assert target_symbols_idxs.dtype == torch.long
        return model_output.decoder_outputs[self.dbg__example_idx_to_test_grads, :, :].sum()

    def forward(self, model_output: PredictLoggingCallVarsModelOutput, target_symbols_idxs: torch.LongTensor):
        assert len(model_output.decoder_outputs.size()) == 3  # (bsz, nr_target_symbols-1, max_nr_possible_symbols)
        assert len(target_symbols_idxs.size()) == 2  # (bsz, nr_target_symbols)
        assert model_output.decoder_outputs.size()[0] == target_symbols_idxs.size()[0]  # bsz
        assert model_output.decoder_outputs.size()[1] + 1 == target_symbols_idxs.size()[1]  # nr_target_symbols
        assert target_symbols_idxs.dtype == torch.long
        return self.criterion(model_output.decoder_outputs.flatten(0, 1), target_symbols_idxs[:, 1:].flatten(0, 1))


class PredictLogVarsTaskDataset(ChunkedRandomAccessDataset):
    def __init__(self, datafold: DataFold, pp_data_path: str):
        super(PredictLogVarsTaskDataset, self).__init__(datafold=datafold, pp_data_path=pp_data_path)
        # TODO: add hash of task props & model HPs to perprocessed file name.

    def __getitem__(self, idx):
        example = super(PredictLogVarsTaskDataset, self).__getitem__(idx)
        assert isinstance(example, PredictLogVarsTaggedExample)
        assert all(hasattr(example, field.name) for field in dataclasses.fields(PredictLogVarsTaggedExample))
        assert isinstance(example.code_task_input, MethodCodeInputToEncoder)
        assert all(hasattr(example.code_task_input, field.name) for field in dataclasses.fields(MethodCodeInputToEncoder))
        return example


def preprocess_logging_call_example(
        model_hps: NDFAModelHyperParams,
        code_task_vocabs: CodeTaskVocabs,
        raw_example: RawExtractedExample,
        add_tag: bool = True) -> typing.Optional[PredictLogVarsTaggedExample]:  # FIXME: is it really optional?
    code_task_input = preprocess_code_task_example(
        model_hps=model_hps, code_task_vocabs=code_task_vocabs,
        method=raw_example.method, method_pdg=raw_example.method_pdg, method_ast=raw_example.method_ast,
        remove_edges_from_pdg_nodes_idxs={raw_example.logging_call.pdg_node_idx},
        pdg_nodes_to_mask={raw_example.logging_call.pdg_node_idx: '<LOG_PRED>'})

    symbols_idxs_used_in_logging_call = get_symbol_idxs_used_in_logging_call(example=raw_example)
    nr_target_symbols = len(symbols_idxs_used_in_logging_call)
    if nr_target_symbols < model_hps.method_code_encoder.min_nr_target_symbols:
        raise PreprocessLimitExceedError(f'#target_symbols ({nr_target_symbols}) < MIN_NR_TARGET_SYMBOLS ({model_hps.method_code_encoder.min_nr_target_symbols})')
    if nr_target_symbols > model_hps.method_code_encoder.max_nr_target_symbols:
        raise PreprocessLimitExceedError(f'#target_symbols ({nr_target_symbols}) > MAX_NR_TARGET_SYMBOLS ({model_hps.method_code_encoder.max_nr_target_symbols})')
    target_symbols_idxs_used_in_logging_call = None if not add_tag else torch.tensor(list(truncate_and_pad(
        [code_task_vocabs.symbols_special_words.get_word_idx('<SOS>')] +
        [symbol_idx_wo_specials + len(code_task_vocabs.symbols_special_words)
         for symbol_idx_wo_specials in symbols_idxs_used_in_logging_call] +
        [code_task_vocabs.symbols_special_words.get_word_idx('<EOS>')],
        max_length=model_hps.method_code_encoder.max_nr_target_symbols + 2,
        pad_word=code_task_vocabs.symbols_special_words.get_word_idx('<PAD>'))), dtype=torch.long)

    return PredictLogVarsTaggedExample(
        example_hash=raw_example.logging_call.hash,
        code_task_input=code_task_input,  # TODO: put logging_call_cfg_node_idx=torch.tensor(logging_call.pdg_node_idx)),
        target_symbols_idxs_used_in_logging_call=target_symbols_idxs_used_in_logging_call)