import os
import torch
import typing
import dataclasses
from warnings import warn
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from ndfa.ndfa_model_hyper_parameters import NDFAModelHyperParams
from ndfa.nn_utils.model_wrapper.dataset_properties import DatasetProperties, DataFold
from ndfa.code_tasks.code_task_base import CodeTaskBase
from ndfa.code_tasks.code_task_properties import CodeTaskProperties
from ndfa.code_tasks.evaluation_metric_base import EvaluationMetric
from ndfa.code_tasks.symbols_set_evaluation_metric import SymbolsSetEvaluationMetric
from ndfa.misc.code_data_structure_api import *
from ndfa.misc.iter_raw_extracted_data_files import iter_raw_extracted_examples_and_verify, RawExtractedExample
from ndfa.nn_utils.model_wrapper.chunked_random_access_dataset import ChunkedRandomAccessDataset
from ndfa.misc.tensors_data_class import TensorsDataClass, BatchedFlattenedIndicesTensor, CollateData
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.method_code_encoder import MethodCodeEncoder, EncodedMethodCode
from ndfa.code_nn_modules.symbols_decoder import SymbolsDecoder
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_tasks.preprocess_code_task_dataset import preprocess_code_task_example, \
    PreprocessLimitExceedError, PreprocessLimitation
from ndfa.nn_utils.model_wrapper.dbg_test_grads import ModuleWithDbgTestGradsMixin
from ndfa.misc.code_data_structure_utils import get_symbol_idxs_used_in_logging_call


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
        example_hashes = [pp_example.example_hash]
        code_task_input = MethodCodeInputTensors.collate(
            [code_task_input], collate_data=CollateData(example_hashes=example_hashes, model_hps=model.model_hps))
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
        return PredictLogVarsTaskDataset(
            datafold=datafold, pp_data_path=pp_data_path,
            storage_method=pp_storage_method, compression_method=pp_compression_method)

    def build_loss_criterion(self, model_hps: NDFAModelHyperParams) -> nn.Module:
        return PredictLogVarsModelLoss(model_hps=model_hps)

    def collate_examples(
            self, examples: List['PredictLogVarsTaggedExample'],
            model_hps: NDFAModelHyperParams) \
            -> 'PredictLogVarsTaggedExample':
        assert all(isinstance(example, PredictLogVarsTaggedExample) for example in examples)
        example_hashes = [example.example_hash for example in examples]
        return PredictLogVarsTaggedExample.collate(
            examples, collate_data=CollateData(example_hashes=example_hashes, model_hps=model_hps))

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

    def create_optimizer(self, model: nn.Module, train_hps: NDFAModelHyperParams) -> torch.optim.Optimizer:
        # TODO: fully implement (choose optimizer and lr)!
        return torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0)
        # return torch.optim.Adam(model.parameters(), lr=0.0005)

    def create_lr_schedulers(
            self, model: nn.Module, train_hps: NDFAModelHyperParams, optimizer: torch.optim.Optimizer) \
            -> typing.Tuple[torch.optim.lr_scheduler._LRScheduler, ...]:
        # FIXME: should we load `last_epoch` from `loaded_checkpoint` or is it loaded on `load_state_dict()`?
        return (
            torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch, last_epoch=-1),
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode='min', factor=0.8, patience=4, verbose=True,
                threshold=0.1, threshold_mode='rel'))


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
    target_symbols_idxs_used_in_logging_call: Optional[LogVarTargetSymbolsIndices]

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
            dropout_rate=dropout_rate, activation_fn=self.model_hps.activation_fn)

        # FIXME: might be problematic because 2 different modules hold this (both PredictLogVarsModel and SymbolsDecoder).
        self.symbols_special_words_embedding = nn.Embedding(
            num_embeddings=len(self.code_task_vocabs.symbols_special_words),
            embedding_dim=self.model_hps.method_code_encoder.symbol_embedding_dim,
            padding_idx=self.code_task_vocabs.symbols_special_words.get_word_idx('<PAD>'))

        if self.model_hps.method_code_encoder.method_encoder_type in {'method-cfg', 'method-cfg-v2'}:
            encoder_output_dim = self.model_hps.method_code_encoder.method_cfg_encoder.cfg_node_encoding_dim
        elif self.model_hps.method_code_encoder.method_encoder_type == 'whole-method':
            # TODO: put in HPs
            encoder_output_dim = \
                self.model_hps.method_code_encoder.whole_method_expression_encoder.tokens_seq_encoder.token_encoding_dim \
                if self.model_hps.method_code_encoder.whole_method_expression_encoder.encoder_type == 'FlatTokensSeq' else \
                self.model_hps.method_code_encoder.whole_method_expression_encoder.ast_encoder.ast_node_embedding_dim
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

        encoded_code: EncodedMethodCode = self.code_task_encoder(code_task_input=code_task_input)
        self.dbg_log_tensor_during_fwd('encoded_identifiers', encoded_code.encoded_identifiers)
        self.dbg_log_tensor_during_fwd('encoded_cfg_nodes', encoded_code.encoded_cfg_nodes)
        self.dbg_log_tensor_during_fwd('all_symbols_encodings', encoded_code.encoded_symbols)
        self.dbg_log_tensor_during_fwd('encoded_cfg_nodes_after_bridge', encoded_code.encoded_cfg_nodes_after_bridge)

        if self.model_hps.method_code_encoder.method_encoder_type == 'method-cfg':
            encoder_outputs = encoded_code.encoded_cfg_nodes_after_bridge
            encoder_outputs_mask = code_task_input.pdg.cfg_nodes_control_kind.unflattener_mask
        elif self.model_hps.method_code_encoder.method_encoder_type == 'method-cfg-v2':
            encoder_outputs = encoded_code.encoded_cfg_nodes_after_bridge
            encoder_outputs_mask = code_task_input.pdg.cfg_nodes_control_kind.unflattener_mask
        elif self.model_hps.method_code_encoder.method_encoder_type == 'whole-method':
            if self.model_hps.method_code_encoder.whole_method_expression_encoder.encoder_type == 'FlatTokensSeq':
                encoder_outputs = encoded_code.whole_method_token_seqs_encoding
                encoder_outputs_mask = code_task_input.method_tokenized_code.token_type.sequences_mask
            elif self.model_hps.method_code_encoder.whole_method_expression_encoder.encoder_type == 'ast':
                if self.model_hps.method_code_encoder.whole_method_expression_encoder.ast_encoder.encoder_type == 'set-of-paths':

                    # TODO: use all path-types (not only 'leaf_to_leaf')
                    encoder_outputs = code_task_input.ast.get_ast_paths_node_indices('leaf_to_leaf').unflatten(
                        encoded_code.whole_method_combined_ast_paths_encoding_by_type['leaf_to_leaf'])
                    encoder_outputs_mask = code_task_input.ast.get_ast_paths_node_indices('leaf_to_leaf').unflattener_mask
                    # all_ast_paths_nodes_encodings = torch.cat([
                    #     encoded_paths
                    #     for ast_paths_type, encoded_paths
                    #     in encoded_code.whole_method_combined_ast_paths_encoding_by_type.items()], dim=0)
                    # all_ast_paths_node_indices = torch.cat([
                    #     code_task_input.ast.get_ast_paths_node_indices(ast_paths_type).example_indices
                    #     for ast_paths_type in encoded_code.whole_method_combined_ast_paths_encoding_by_type.keys()], dim=0)

                elif self.model_hps.method_code_encoder.whole_method_expression_encoder.ast_encoder.encoder_type in {'tree', 'paths-folded'}:
                    encoder_outputs = code_task_input.ast.ast_node_major_types.unflatten(
                        encoded_code.whole_method_ast_nodes_encoding)
                    encoder_outputs_mask = code_task_input.ast.ast_node_major_types.unflattener_mask
                else:
                    assert False
            else:
                assert False
            # print('encoder_outputs', encoder_outputs.shape)
            # print('encoder_outputs_mask', encoder_outputs_mask.shape)
        else:
            assert False

        decoder_outputs = self.symbols_decoder(
            encoder_outputs=encoder_outputs,
            encoder_outputs_mask=encoder_outputs_mask,
            symbols=code_task_input.symbols,
            batched_flattened_symbols_encodings=encoded_code.encoded_symbols,
            encoded_symbols_occurrences=encoded_code.encoded_symbols_occurrences,
            groundtruth_target_symbols_idxs=target_symbols_idxs)
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
            self, datafold: DataFold, pp_data_path: str,
            storage_method: str = 'dbm',
            compression_method: str = 'none'):
        super(PredictLogVarsTaskDataset, self).__init__(
            pp_data_path_prefix=os.path.join(pp_data_path, f'pp_{datafold.value.lower()}'),
            storage_method=storage_method, compression_method=compression_method)
        # TODO: add hash of task props & model HPs to perprocessed file name.

    def __getitem__(self, idx):
        example = super(PredictLogVarsTaskDataset, self).__getitem__(idx)
        assert isinstance(example, PredictLogVarsTaggedExample)
        assert all(hasattr(example, field.name) for field in dataclasses.fields(PredictLogVarsTaggedExample))
        assert isinstance(example.code_task_input, MethodCodeInputTensors)
        assert all(hasattr(example.code_task_input, field.name) for field in dataclasses.fields(MethodCodeInputTensors))
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

    # Note: The ast-node of the logging-call is the method-call itself, while the ast-node of the pdg-node of the
    #       logging-call is the stmt-expr (which is the parent of the method-call)
    logging_call_pdg_node_ast_node_idx = raw_example.method_pdg.pdg_nodes[raw_example.logging_call.pdg_node_idx].ast_node_idx
    logging_call_pdg_node_ast_node = raw_example.method_ast.nodes[logging_call_pdg_node_ast_node_idx]

    limitations = [
        PreprocessLimitation(
            object_name='is_valid_log_stmt_pdg_node_sub_ast_structure',
            value=int(
                (raw_example.logging_call.ast_node_idx in logging_call_pdg_node_ast_node.children_idxs) and
                (logging_call_pdg_node_ast_node.type == SerASTNodeType.EXPRESSION_STMT) and
                (raw_example.method_ast.nodes[raw_example.logging_call.ast_node_idx].type ==
                 SerASTNodeType.METHOD_CALL_EXPR) and
                (len(code_task_input.ast.ast_leaves_sequence_node_indices.sequences) == 1)),
            min_val=1)]
    PreprocessLimitation.enforce_limitations(limitations=limitations)

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

    symbols_idxs_used_in_logging_call = get_symbol_idxs_used_in_logging_call(example=raw_example)
    nr_target_symbols = len(symbols_idxs_used_in_logging_call)

    limitations = [PreprocessLimitation(
        object_name='#target_symbols', value=nr_target_symbols,
        min_val=model_hps.target_symbols_decoder.min_nr_target_symbols,
        max_val=model_hps.target_symbols_decoder.max_nr_target_symbols)]
    exceeding_limitations = [limitation for limitation in limitations if limitation.exceeds]
    for exceeding_limitation in exceeding_limitations:
        if exceeding_limitation.warn:
            warn(str(exceeding_limitation))
    if len(exceeding_limitations) > 0:
        raise PreprocessLimitExceedError(exceeding_limitations=exceeding_limitations)

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
        code_task_input=code_task_input,  # TODO: put logging_call_cfg_node_idx=torch.tensor(logging_call.pdg_node_idx)),
        target_symbols_idxs_used_in_logging_call=target_symbols_idxs_used_in_logging_call)
