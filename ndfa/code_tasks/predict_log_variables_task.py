import os
import torch
import typing
import dataclasses
from warnings import warn
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams, NDFAModelTrainingHyperParams
from ndfa.nn_utils.model_wrapper.dataset_properties import DatasetProperties, DataFold
from ndfa.code_tasks.code_task_base import CodeTaskBase
from ndfa.code_tasks.code_task_properties import CodeTaskProperties
from ndfa.code_tasks.evaluation_metric_base import EvaluationMetric
from ndfa.code_tasks.symbols_set_evaluation_metric import SymbolsSetEvaluationMetric
from ndfa.misc.code_data_structure_api import *
from ndfa.misc.iter_raw_extracted_data_files import iter_raw_extracted_examples_and_verify, RawExtractedExample
from ndfa.nn_utils.model_wrapper.chunked_random_access_dataset import ChunkedRandomAccessDataset
from ndfa.misc.tensors_data_class import TensorsDataClass, BatchedFlattenedIndicesTensor, CollateData, \
    batch_flattened_indices_tensor_field
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.method_code_encoder import MethodCodeEncoder, EncodedMethodCode
from ndfa.code_nn_modules.symbols_decoder import SymbolsDecoder
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_tasks.preprocess_code_task_dataset import preprocess_code_task_example, \
    PreprocessLimitExceedError, PreprocessLimitation
from ndfa.nn_utils.model_wrapper.dbg_test_grads import ModuleWithDbgTestGradsMixin
from ndfa.misc.code_data_structure_utils import get_symbol_idxs_used_in_logging_call
from ndfa.code_nn_modules.method_code_encoding_feeder import MethodCodeEncodingsFeeder
from ndfa.code_nn_modules.params.method_code_encoder_params import MethodCodeEncoderParams
from ndfa.code_tasks.method_code_preprocess_params import NDFAModelPreprocessParams, NDFAModelPreprocessedDataParams
from ndfa.code_tasks.create_preprocess_params_from_model_hps import create_preprocess_params_from_model_hps


__all__ = ['PredictLogVarsTask', 'PredictLogVarsTaggedExample', 'PredictLogVarsTaskDataset']


class PredictLogVarsTask(CodeTaskBase):
    def __init__(self, task_props: CodeTaskProperties):
        super(PredictLogVarsTask, self).__init__(task_props)
        # TODO: extract relevant fields from `task_props` into some `PredictLogVariablesTaskProps`!

    def iterate_raw_examples(self, model_hps: NDFAModelHyperParams, raw_extracted_data_dir: str) \
            -> typing.Iterable[RawExtractedExample]:
        return iter_raw_extracted_examples_and_verify(
            raw_extracted_data_dir=raw_extracted_data_dir, show_progress_bar=True)

    def preprocess_raw_example(
            self, model_hps: NDFAModelHyperParams,
            preprocess_params: NDFAModelPreprocessParams,
            code_task_vocabs: CodeTaskVocabs,
            raw_example: Any,
            add_tag: bool = True) \
            -> 'PredictLogVarsTaggedExample':
        return preprocess_logging_call_example(
            model_hps=model_hps, preprocess_params=preprocess_params,
            code_task_vocabs=code_task_vocabs, raw_example=raw_example, add_tag=add_tag)

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
        example_hashes = [pp_example.example_hash]
        code_task_input = MethodCodeInputTensors.collate(
            [code_task_input], collate_data=CollateData(
                example_hashes=example_hashes, model_hps=model.model_hps, is_training=False))
        code_task_input = code_task_input.to(device)
        output: PredictLoggingCallVarsModelOutput = model(code_task_input=code_task_input)
        decoder_outputs = output.decoder_outputs.squeeze(dim=0)
        symbol_indices = decoder_outputs.argmax(dim=-1)
        EOS_symbol_special_word_idx = 2  # '<EOS>'
        symbol_indices = symbol_indices.cpu().tolist()
        symbol_indices = [symbol_idx - 3 for symbol_idx in symbol_indices if symbol_idx > 2]
        symbol_names = [raw_example.method_pdg.symbols[symbol_idx].symbol_name for symbol_idx in symbol_indices]
        print(symbol_names)
        return symbol_names

    def create_dataset(
            self, model_hps: NDFAModelHyperParams, dataset_props: DatasetProperties,
            datafold: DataFold, pp_data_path: str, pp_storage_method: str = 'dbm',
            pp_compression_method: str = 'none') -> Dataset:
        preprocess_params = create_preprocess_params_from_model_hps(model_hps=model_hps)
        preprocessed_data_params = NDFAModelPreprocessedDataParams(
            preprocess_params=preprocess_params, dataset_props=dataset_props)
        return PredictLogVarsTaskDataset(
            preprocessed_data_params=preprocessed_data_params, datafold=datafold, pp_data_path=pp_data_path,
            storage_method=pp_storage_method, compression_method=pp_compression_method)

    def build_loss_criterion(self, model_hps: NDFAModelHyperParams) -> nn.Module:
        return PredictLogVarsModelLoss(model_hps=model_hps)

    def collate_examples(
            self, examples: List['PredictLogVarsTaggedExample'],
            model_hps: NDFAModelHyperParams,
            is_training: bool) \
            -> 'PredictLogVarsTaggedExample':
        assert all(isinstance(example, PredictLogVarsTaggedExample) for example in examples)
        example_hashes = [example.example_hash for example in examples]
        return PredictLogVarsTaggedExample.collate(
            examples, collate_data=CollateData(
                example_hashes=example_hashes, model_hps=model_hps, is_training=is_training))

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

    def create_optimizer(self, model: nn.Module, train_hps: NDFAModelTrainingHyperParams) -> torch.optim.Optimizer:
        # TODO: fully implement (choose optimizer and lr)!
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=train_hps.learning_rate,
            weight_decay=train_hps.weight_decay)
        # return torch.optim.Adam(model.parameters(), lr=0.0005)

    def create_lr_schedulers(
            self, model: nn.Module, train_hps: NDFAModelTrainingHyperParams, optimizer: torch.optim.Optimizer) \
            -> typing.Tuple[torch.optim.lr_scheduler._LRScheduler, ...]:
        # FIXME: should we load `last_epoch` from `loaded_checkpoint` or is it loaded on `load_state_dict()`?
        schedulers = []
        if train_hps.learning_rate_decay:
            schedulers.append(torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda epoch_nr: (1 - train_hps.learning_rate_decay) ** epoch_nr,
                last_epoch=-1))
        if train_hps.reduce_lr_on_plateau:
            # TODO: load these params from `train_hps`
            schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='min', factor=0.8, patience=4, verbose=True,
                threshold=0.1, threshold_mode='rel'))
        return tuple(schedulers)


class LoggingCallTaskEvaluationMetric(SymbolsSetEvaluationMetric):
    def __init__(self, nr_symbols_special_words: int):
        super(LoggingCallTaskEvaluationMetric, self).__init__()
        self.nr_symbols_special_words = nr_symbols_special_words

    def update(self, y_hat: 'PredictLoggingCallVarsModelOutput', target: 'LogVarTargetSymbolsIndices'):
        _, batch_pred_symbols_indices = y_hat.decoder_outputs.topk(k=1, dim=-1)
        batch_pred_symbols_indices = batch_pred_symbols_indices.squeeze(dim=-1)
        assert batch_pred_symbols_indices.ndim == target.example_based_symbol_indices.ndim == 2
        assert batch_pred_symbols_indices.size(0) == target.example_based_symbol_indices.size(0)  # bsz
        assert batch_pred_symbols_indices.size(1) + 1 == target.example_based_symbol_indices.size(1)  # seq_len; prefix `<SOS>` is not predicted.
        batch_pred_symbols_indices = batch_pred_symbols_indices.numpy()
        target_example_based_symbol_indices = target.example_based_symbol_indices.numpy()
        batch_size = target_example_based_symbol_indices.shape[0]
        for example_idx_in_batch in range(batch_size):
            example_pred_symbols_indices = [
                symbol_idx for symbol_idx in batch_pred_symbols_indices[example_idx_in_batch, :]
                if symbol_idx >= self.nr_symbols_special_words]
            example_target_symbols_indices = [
                symbol_idx for symbol_idx in target_example_based_symbol_indices[example_idx_in_batch, :]
                if symbol_idx >= self.nr_symbols_special_words]
            super(LoggingCallTaskEvaluationMetric, self).update(
                example_pred_symbols_indices=example_pred_symbols_indices,
                example_target_symbols_indices=example_target_symbols_indices)


@dataclasses.dataclass
class PredictLoggingCallVarsModelOutput(TensorsDataClass):
    decoder_outputs: torch.Tensor
    all_symbols_encodings: torch.Tensor


@dataclasses.dataclass
class LogVarTargetSymbolsIndices(TensorsDataClass):
    # Note: it is a wasteful solution for having both batch-flattened & example-based symbols indices.
    example_based_symbol_indices: torch.Tensor
    batch_flattened_symbol_indices: BatchedFlattenedIndicesTensor


@dataclasses.dataclass
class PredictLogVarsTaggedExample(TensorsDataClass):
    example_hash: str
    code_task_input: MethodCodeInputTensors
    logging_call_token_idx_in_flat_whole_method_tokenized: torch.LongTensor
    logging_call_cfg_node_idx: BatchedFlattenedIndicesTensor = \
        batch_flattened_indices_tensor_field(tgt_indexing_group='cfg_nodes')
    logging_call_ast_node_idx: BatchedFlattenedIndicesTensor = \
        batch_flattened_indices_tensor_field(tgt_indexing_group='ast_nodes')
    target_symbols_idxs_used_in_logging_call: Optional[LogVarTargetSymbolsIndices] = None

    def __iter__(self):  # To support unpacking into (x_batch, y_batch)
        yield self.code_task_input
        yield self.target_symbols_idxs_used_in_logging_call


class PredictLogVarsModel(nn.Module, ModuleWithDbgTestGradsMixin):
    def __init__(self, model_hps: NDFAModelHyperParams, code_task_vocabs: CodeTaskVocabs, dropout_rate: float = 0.3):
        super(PredictLogVarsModel, self).__init__()
        ModuleWithDbgTestGradsMixin.__init__(self)
        self.model_hps = model_hps
        self.code_task_vocabs = code_task_vocabs

        self.code_task_encoder = MethodCodeEncoder(
            encoder_params=self.model_hps.method_code_encoder,
            code_task_vocabs=code_task_vocabs,
            norm_params=self.model_hps.normalization,
            dropout_rate=dropout_rate, activation_fn=self.model_hps.activation_fn)

        # FIXME: might be problematic because 2 different modules hold this (both PredictLogVarsModel and SymbolsDecoder).
        self.symbols_special_words_embedding = nn.Embedding(
            num_embeddings=len(self.code_task_vocabs.symbols_special_words),
            embedding_dim=self.model_hps.method_code_encoder.symbol_embedding_dim,
            padding_idx=self.code_task_vocabs.symbols_special_words.get_word_idx('<PAD>'))

        if self.model_hps.method_code_encoder.method_encoder_type in \
                {MethodCodeEncoderParams.EncoderType.MethodCFG, MethodCodeEncoderParams.EncoderType.MethodCFGV2}:
            encoder_output_dim = self.model_hps.method_code_encoder.method_cfg_encoder.cfg_node_encoding_dim
        elif self.model_hps.method_code_encoder.method_encoder_type == MethodCodeEncoderParams.EncoderType.Hierarchic:
            encoder_output_dim = self.model_hps.method_code_encoder.hierarchic_micro_macro_encoder.macro_encoding_dim
        elif self.model_hps.method_code_encoder.method_encoder_type == MethodCodeEncoderParams.EncoderType.WholeMethod:
            encoder_output_dim = \
                self.model_hps.method_code_encoder.whole_method_expression_encoder.expression_encoding_dim
        else:
            assert False

        self.symbols_decoder = SymbolsDecoder(
            symbols_special_words_embedding=self.symbols_special_words_embedding,  # FIXME: might be problematic because 2 different modules hold this (both PredictLogVarsModel and SymbolsDecoder).
            symbols_special_words_vocab=self.code_task_vocabs.symbols_special_words,
            max_nr_taget_symbols=model_hps.target_symbols_decoder.max_nr_target_symbols + 2,
            encoder_output_dim=encoder_output_dim,
            symbols_encoding_dim=self.model_hps.method_code_encoder.symbol_embedding_dim,
            use_batch_flattened_target_symbols_vocab=
            self.model_hps.target_symbols_decoder.use_batch_flattened_target_symbols_vocab,
            dropout_rate=dropout_rate, activation_fn=self.model_hps.activation_fn)

        self.method_code_encodings_feeder = MethodCodeEncodingsFeeder(
            method_code_encoder_params=self.model_hps.method_code_encoder)

    def forward(
            self, code_task_input: MethodCodeInputTensors,
            target_symbols_idxs: Optional[LogVarTargetSymbolsIndices] = None):
        self.dbg_log_new_fwd()
        use_batch_flattened_target_symbols_vocab = \
            self.model_hps.target_symbols_decoder.use_batch_flattened_target_symbols_vocab and self.training
        if target_symbols_idxs is not None:
            target_symbols_idxs = \
                target_symbols_idxs.batch_flattened_symbol_indices.indices \
                if use_batch_flattened_target_symbols_vocab else \
                target_symbols_idxs.example_based_symbol_indices

        encoded_method_code: EncodedMethodCode = self.code_task_encoder(code_task_input=code_task_input)
        self.dbg_log_tensor_during_fwd('encoded_identifiers', encoded_method_code.encoded_identifiers)
        self.dbg_log_tensor_during_fwd('encoded_cfg_nodes', encoded_method_code.encoded_cfg_nodes)
        self.dbg_log_tensor_during_fwd('all_symbols_encodings', encoded_method_code.encoded_symbols)
        self.dbg_log_tensor_during_fwd(
            'encoded_cfg_nodes_after_bridge', encoded_method_code.encoded_cfg_nodes_after_bridge)

        encoder_outputs, encoder_outputs_mask = self.method_code_encodings_feeder(
            code_task_input=code_task_input, encoded_method_code=encoded_method_code)

        decoder_outputs = self.symbols_decoder(
            encoder_outputs=encoder_outputs,
            encoder_outputs_mask=encoder_outputs_mask,
            symbols=code_task_input.symbols,
            batched_flattened_symbols_encodings=encoded_method_code.encoded_symbols,
            encoded_symbols_occurrences=encoded_method_code.encoded_symbols_occurrences,
            groundtruth_target_symbols_idxs=target_symbols_idxs)
        self.dbg_log_tensor_during_fwd('decoder_outputs', decoder_outputs)

        return PredictLoggingCallVarsModelOutput(
            decoder_outputs=decoder_outputs,
            all_symbols_encodings=encoded_method_code.encoded_symbols)


class PredictLogVarsModelLoss(nn.Module):
    def __init__(self, model_hps: NDFAModelHyperParams):
        super(PredictLogVarsModelLoss, self).__init__()
        self.model_hps = model_hps
        self.criterion = nn.NLLLoss(reduction='mean')  # TODO: decide what criterion to use based on model-hps.
        self.dbg__example_idx_to_test_grads = 3

    def dbg_forward_test_grads(
            self, model_output: PredictLoggingCallVarsModelOutput,
            target_symbols_idxs: LogVarTargetSymbolsIndices):
        target_symbols_idxs = \
            target_symbols_idxs.batch_flattened_symbol_indices.indices \
            if self.model_hps.use_batch_flattened_target_symbols_vocab and self.training else \
            target_symbols_idxs.example_based_symbol_indices
        # Change this to be the forward() when debugging gradients.
        assert model_output.decoder_outputs.ndim == 3  # (bsz, nr_target_symbols-1, max_nr_possible_symbols)
        assert target_symbols_idxs.ndim == 2  # (bsz, nr_target_symbols)
        assert model_output.decoder_outputs.size(0) == target_symbols_idxs.size(0)  # bsz
        assert model_output.decoder_outputs.size(1) + 1 == target_symbols_idxs.size(1)  # nr_target_symbols
        assert target_symbols_idxs.dtype == torch.long
        return model_output.decoder_outputs[self.dbg__example_idx_to_test_grads, :, :].sum()

    def forward(self, model_output: PredictLoggingCallVarsModelOutput,
                target_symbols_idxs: LogVarTargetSymbolsIndices):
        target_symbols_idxs = \
            target_symbols_idxs.batch_flattened_symbol_indices.indices \
            if self.model_hps.target_symbols_decoder.use_batch_flattened_target_symbols_vocab and self.training else \
            target_symbols_idxs.example_based_symbol_indices

        assert model_output.decoder_outputs.ndim == 3  # (bsz, nr_target_symbols-1, max_nr_possible_symbols)
        assert target_symbols_idxs.ndim == 2  # (bsz, nr_target_symbols)
        assert model_output.decoder_outputs.size(0) == target_symbols_idxs.size(0)  # bsz
        assert model_output.decoder_outputs.size(1) + 1 == target_symbols_idxs.size(1)  # nr_target_symbols
        assert target_symbols_idxs.dtype == torch.long
        return self.criterion(model_output.decoder_outputs.flatten(0, 1), target_symbols_idxs[:, 1:].flatten(0, 1))


class PredictLogVarsTaskDataset(ChunkedRandomAccessDataset):
    def __init__(
            self,
            preprocessed_data_params: NDFAModelPreprocessedDataParams,
            datafold: DataFold,
            pp_data_path: str,
            storage_method: str = 'dbm',
            compression_method: str = 'none'):
        super(PredictLogVarsTaskDataset, self).__init__(
            pp_data_path_prefix=os.path.join(
                pp_data_path, f'pp_{datafold.value.lower()}_{preprocessed_data_params.get_sha1_base64()}'),
            storage_method=storage_method, compression_method=compression_method)

    def __getitem__(self, possibly_batched_index):
        example = super(PredictLogVarsTaskDataset, self).__getitem__(possibly_batched_index)
        if isinstance(example, PredictLogVarsTaggedExample):
            # assert all(hasattr(example, field.name) for field in dataclasses.fields(PredictLogVarsTaggedExample))
            # assert isinstance(example.code_task_input, MethodCodeInputTensors)
            # assert all(hasattr(example.code_task_input, field.name) for field in dataclasses.fields(MethodCodeInputTensors))
            return example
        elif isinstance(example, list):
            assert all(isinstance(ex, PredictLogVarsTaggedExample) for ex in example)
            return example
        else:
            assert False


def preprocess_logging_call_example(
        model_hps: NDFAModelHyperParams,
        preprocess_params: NDFAModelPreprocessParams,
        code_task_vocabs: CodeTaskVocabs,
        raw_example: RawExtractedExample,
        add_tag: bool = True) -> typing.Optional[PredictLogVarsTaggedExample]:  # FIXME: is it really optional?
    code_task_input = preprocess_code_task_example(
        model_hps=model_hps, preprocess_params=preprocess_params, code_task_vocabs=code_task_vocabs,
        method=raw_example.method, method_pdg=raw_example.method_pdg, method_ast=raw_example.method_ast,
        remove_edges_from_pdg_nodes_idxs={raw_example.logging_call.pdg_node_idx},
        pdg_nodes_to_mask={raw_example.logging_call.pdg_node_idx: '<LOG_PRED>'})

    # Note: The ast-node of the logging-call is the method-call itself, while the ast-node of the pdg-node of the
    #       logging-call is the stmt-expr. The ast-node of the pdg-node is an ancestor of the method-call, not
    #       necessarily the direct parent. We take only examples in which it is the direct parent.
    logging_call_pdg_node = raw_example.method_pdg.pdg_nodes[raw_example.logging_call.pdg_node_idx]
    logging_call_pdg_node_ast_node_idx = logging_call_pdg_node.ast_node_idx
    logging_call_pdg_node_ast_node = raw_example.method_ast.nodes[logging_call_pdg_node_ast_node_idx]

    limitations = [
        PreprocessLimitation(
            object_name='is_valid_log_stmt_pdg_node_sub_ast_structure',
            value=int(
                (raw_example.logging_call.ast_node_idx in logging_call_pdg_node_ast_node.children_idxs) and
                (logging_call_pdg_node_ast_node.type == SerASTNodeType.EXPRESSION_STMT) and
                (raw_example.method_ast.nodes[raw_example.logging_call.ast_node_idx].type ==
                 SerASTNodeType.METHOD_CALL_EXPR) and
                (code_task_input.ast is None or code_task_input.ast.ast_leaves_sequence_node_indices is None or
                 len(code_task_input.ast.ast_leaves_sequence_node_indices.sequences) == 1) and
                logging_call_pdg_node.code_sub_token_range_ref is not None and
                logging_call_pdg_node.code_sub_token_range_ref.begin_token_idx ==
                logging_call_pdg_node_ast_node.code_sub_token_range_ref.begin_token_idx),
            min_val=1)]
    PreprocessLimitation.enforce_limitations(limitations=limitations)

    if code_task_input.ast is not None and code_task_input.ast.ast_leaf_to_leaf_paths_node_indices is not None:
        is_log_stmt_included_in_any_l2l_path = any(
            ast_path[0] == logging_call_pdg_node_ast_node_idx or ast_path[-1] == logging_call_pdg_node_ast_node_idx
            for ast_path in code_task_input.ast.ast_leaf_to_leaf_paths_node_indices.sequences)
        if not is_log_stmt_included_in_any_l2l_path:
            warn('Log-stmt ast-node is not included in any sampled AST leaf-to-leaf path.')

    # DBG: Calc probability (per example) for not including the log-stmt in any sampled AST leaf2leaf path.
    # nr_ast_leaves = len(code_task_input.ast.ast_leaves_sequence_node_indices.sequences[0])
    # from math import comb
    # nr_ast_l2l_paths = comb(nr_ast_leaves, 2)
    # is_sampled = nr_ast_l2l_paths > 1000
    # nr_paths_including_log_node = (nr_ast_leaves - 1)
    # prob_for_single_choose_of_path_including_log = nr_paths_including_log_node / nr_ast_l2l_paths
    # prob_not_choosing_path_including_log = \
    #     (1 - prob_for_single_choose_of_path_including_log) ** 1000 if is_sampled else 0
    # chosen_nodes = set(
    #     int(ast_path[edge]) for ast_path in code_task_input.ast.ast_leaf_to_leaf_paths_node_indices.sequences for edge
    #     in [0, -1])
    # print(f'\n{is_log_included_in_any_l2l_path} -- '
    #       f'\tsampled: {is_sampled} -- '
    #       f'\t#chosen/#nodes: {len(chosen_nodes)}/{nr_ast_leaves}={(len(chosen_nodes)/nr_ast_leaves)*100:.2f}% -- '
    #       f'\tprob1: {prob_not_choosing_path_including_log*100:.2f}% -- '
    #       f'\tprob2: {prob_for_single_choose_of_path_including_log*100:.2f}% -- '
    #       f'\t#l2l: {nr_ast_l2l_paths} -- '
    #       f'\t#leaves: {nr_ast_leaves}')

    # Shallow sanity-check for logging-call masking (note we don't really verify here the whole expression/sub-ast is
    # masked, but it is checked by `preprocess_code_task_example()`).
    assert code_task_input.method_tokenized_code is None or code_task_vocabs.tokens_kinds.idx2word[
           code_task_input.method_tokenized_code.token_type.sequences[0][
               logging_call_pdg_node.code_sub_token_range_ref.begin_token_idx].item()] == '<LOG_PRED>'
    assert code_task_input.pdg is None or code_task_vocabs.pdg_node_control_kinds.idx2word[
               code_task_input.pdg.cfg_nodes_control_kind.tensor[logging_call_pdg_node.idx].item()] == '<LOG_PRED>'
    assert code_task_input.ast is None or code_task_vocabs.ast_node_major_types.idx2word[
               code_task_input.ast.ast_node_types.tensor[logging_call_pdg_node_ast_node_idx].item()] == '<LOG_PRED>'

    symbols_idxs_used_in_logging_call = get_symbol_idxs_used_in_logging_call(example=raw_example)
    nr_target_symbols = len(symbols_idxs_used_in_logging_call)

    limitations = [PreprocessLimitation(
        object_name='#target_symbols', value=nr_target_symbols,
        min_val=model_hps.target_symbols_decoder.min_nr_target_symbols,
        max_val=model_hps.target_symbols_decoder.max_nr_target_symbols)]
    PreprocessLimitation.enforce_limitations(limitations=limitations)

    target_symbols_idxs_used_in_logging_call = None
    if add_tag:
        target_example_based_symbol_indices = torch.LongTensor(
            [code_task_vocabs.symbols_special_words.get_word_idx('<SOS>')] +
            [symbol_idx_wo_specials + len(code_task_vocabs.symbols_special_words)
             for symbol_idx_wo_specials in symbols_idxs_used_in_logging_call] +
            [code_task_vocabs.symbols_special_words.get_word_idx('<EOS>')] +
            [code_task_vocabs.symbols_special_words.get_word_idx('<PAD>')] *
            (model_hps.target_symbols_decoder.max_nr_target_symbols - len(symbols_idxs_used_in_logging_call)))
        # Used for batched target vocab decoder
        # (using encodings of symbols of all examples in the batch as tgt vocab)
        target_batch_flattened_symbol_indices = BatchedFlattenedIndicesTensor(
                indices=target_example_based_symbol_indices,
                tgt_indexing_group='symbols',
                within_example_indexing_start=len(code_task_vocabs.symbols_special_words))
        target_symbols_idxs_used_in_logging_call = LogVarTargetSymbolsIndices(
            example_based_symbol_indices=target_example_based_symbol_indices,
            batch_flattened_symbol_indices=target_batch_flattened_symbol_indices)

    return PredictLogVarsTaggedExample(
        example_hash=raw_example.logging_call.hash,
        code_task_input=code_task_input,
        logging_call_cfg_node_idx=BatchedFlattenedIndicesTensor(
            indices=torch.LongTensor([raw_example.logging_call.pdg_node_idx]),
        ) if preprocess_params.method_code.hierarchic else None,  # tgt_indexing_group='cfg_nodes'),
        logging_call_ast_node_idx=BatchedFlattenedIndicesTensor(
            indices=torch.LongTensor([logging_call_pdg_node_ast_node_idx]),
        ) if preprocess_params.method_code.general_ast else None,  # tgt_indexing_group='ast_nodes'),
        logging_call_token_idx_in_flat_whole_method_tokenized=torch.LongTensor(
            [logging_call_pdg_node.code_sub_token_range_ref.begin_token_idx])
        if preprocess_params.method_code.whole_method_tokens_seq else None,
        target_symbols_idxs_used_in_logging_call=target_symbols_idxs_used_in_logging_call)
