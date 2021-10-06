import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from ndfa.code_nn_modules.params.hierarchic_micro_macro_method_code_encoder_params import \
    HierarchicMicroMacroMethodCodeEncoderParams as EncoderParams
from ndfa.code_nn_modules.code_expression_encoder_with_combiner import CodeExpressionEncoderWithCombiner
from ndfa.code_nn_modules.code_expression_encoder import CodeExpressionEncoder
from ndfa.code_nn_modules.code_expression_embedder import CodeExpressionEmbedder
from ndfa.code_nn_modules.code_expression_context_mixer import CodeExpressionContextMixer
from ndfa.code_nn_modules.method_cfg_macro_encoder import MethodCFGMacroEncoder, MethodCFGMacroEncodings
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.code_nn_modules.symbols_encoder import SymbolsEncoder
from ndfa.code_nn_modules.params.symbols_encoder_params import SymbolsEncoderParams
from ndfa.code_nn_modules.symbol_occurrences_extractor_from_encoded_method import \
    SymbolOccurrencesExtractorFromEncodedMethod
from ndfa.code_nn_modules.params.method_cfg_macro_encoder_params import MethodCFGMacroEncoderParams
from ndfa.code_nn_modules.params.cfg_paths_macro_encoder_params import CFGPathsMacroEncoderParams
from ndfa.nn_utils.model_wrapper.flattened_tensor import FlattenedTensor


__all__ = ['HierarchicMicroMacroMethodCodeEncoder', 'HierarchicMicroMacroMethodCodeEncodings']


@dataclass
class HierarchicMicroMacroMethodCodeEncodings:
    identifiers_encodings: torch.Tensor
    micro_encodings: CodeExpressionEncodingsTensors
    macro_encodings: FlattenedTensor
    global_context_aware_micro_encodings: CodeExpressionEncodingsTensors
    symbols_encodings: torch.Tensor


class HierarchicMicroMacroMethodCodeEncoder(nn.Module):
    def __init__(
            self,
            code_task_vocabs: CodeTaskVocabs,
            identifier_embedding_dim: int,
            symbol_embedding_dim: int,
            symbols_encoder_params: SymbolsEncoderParams,
            params: EncoderParams,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(HierarchicMicroMacroMethodCodeEncoder, self).__init__()
        self.params = params
        self.identifier_embedding_dim = identifier_embedding_dim
        self.symbol_embedding_dim = symbol_embedding_dim
        self.code_expression_embedder = CodeExpressionEmbedder(
            code_task_vocabs=code_task_vocabs,
            encoder_params=self.params.local_expression_encoder,
            identifier_embedding_dim=self.identifier_embedding_dim,
            nr_final_embeddings_linear_layers=1,  # TODO: plug HP here
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.code_expression_encoder_before_macro = CodeExpressionEncoderWithCombiner(
            encoder_params=self.params.local_expression_encoder,
            code_task_vocabs=code_task_vocabs,
            identifier_embedding_dim=self.identifier_embedding_dim,
            is_first_encoder_layer=True,
            norm_params=norm_params,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

        self.macro_encoder = MethodCFGMacroEncoder(
            params=self.params.global_context_encoder,
            combined_micro_expression_dim=self.params.local_expression_encoder.combined_expression_encoding_dim,
            identifier_embedding_dim=identifier_embedding_dim,
            code_task_vocabs=code_task_vocabs,
            norm_params=norm_params,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

        # TODO: maybe the case of `self.params.global_context_encoder.encoder_type == SetOfCFGNodes`
        #  should also be considered as shouldn't be mixed.
        self.should_mix_code_expression_with_global_context = not (
            self.params.global_context_encoder.encoder_type == MethodCFGMacroEncoderParams.EncoderType.CFGPaths and
            self.params.global_context_encoder.paths_encoder.output_type ==
            CFGPathsMacroEncoderParams.OutputType.SetOfPaths)

        if self.should_mix_code_expression_with_global_context:
            self.code_expression_global_context_mixer = CodeExpressionContextMixer(
                cfg_node_encoding_dim=params.global_context_encoder.cfg_node_encoding_dim,
                macro_context_to_micro_state_updater=
                self.params.global_context_encoder.macro_context_to_micro_state_updater,
                encoder_params=self.params.local_expression_encoder,
                norm_params=norm_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)

            if self.params.after_macro.requires_micro:
                self.code_expression_encoder_after_macro = CodeExpressionEncoder(
                    encoder_params=self.params.local_expression_encoder_after_macro,
                    code_task_vocabs=code_task_vocabs,
                    identifier_embedding_dim=self.identifier_embedding_dim,
                    is_first_encoder_layer=False,
                    norm_params=norm_params,
                    dropout_rate=dropout_rate,
                    activation_fn=activation_fn)
        else:
            assert not self.params.after_macro.requires_micro

        self.symbols_encoder = SymbolsEncoder(
            identifier_embedding_dim=self.identifier_embedding_dim,
            symbol_embedding_dim=self.symbol_embedding_dim,
            expression_encoding_dim=self.params.expression_encoding_dim,
            encoder_params=symbols_encoder_params,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        if symbols_encoder_params.use_symbols_occurrences:
            self.symbol_occurrences_extractor = SymbolOccurrencesExtractorFromEncodedMethod(
                code_expression_encoder_params=self.params.local_expression_encoder)

    def forward(
            self,
            code_task_input: MethodCodeInputTensors,
            encoded_identifiers: torch.Tensor) -> HierarchicMicroMacroMethodCodeEncodings:
        embedded_code_expressions: CodeExpressionEncodingsTensors = self.code_expression_embedder(
            encoded_identifiers=encoded_identifiers,
            tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
            method_ast_input=code_task_input.ast)

        micro_encoded_code_expressions: CodeExpressionEncodingsTensors = self.code_expression_encoder_before_macro(
            previous_code_expression_encodings=embedded_code_expressions,
            tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
            cfg_nodes_expressions_ast=code_task_input.pdg.cfg_nodes_expressions_ast,
            cfg_nodes_has_expression_mask=code_task_input.pdg.cfg_nodes_has_expression_mask.tensor)

        macro_encodings: MethodCFGMacroEncodings = self.macro_encoder(
            code_task_input=code_task_input,
            encoded_identifiers=encoded_identifiers,
            encoded_combined_code_expressions=micro_encoded_code_expressions.combined_expressions)

        # Note: For cases like CFG-Paths without node occurrences folding, where the encodings of the CFG nodes are
        #       NOT being updated with their global context, we should avoid here from the further steps of mixing
        #       the code expressions with the CFG nodes embedding and so on.
        global_context_aware_micro_encodings = None
        if self.should_mix_code_expression_with_global_context:
            global_context_aware_micro_encodings = self.code_expression_global_context_mixer(
                encoded_code_expressions=micro_encoded_code_expressions,
                encoded_cfg_nodes=macro_encodings.cfg_nodes_encodings,
                pdg_input=code_task_input.pdg)

            if self.params.after_macro.requires_micro:
                final_encoded_code_expressions: CodeExpressionEncodingsTensors = \
                    self.code_expression_encoder_after_macro(
                        previous_code_expression_encodings=global_context_aware_micro_encodings,
                        tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
                        cfg_nodes_expressions_ast=code_task_input.pdg.cfg_nodes_expressions_ast,
                        cfg_nodes_has_expression_mask=code_task_input.pdg.cfg_nodes_has_expression_mask.tensor)
            else:
                final_encoded_code_expressions = global_context_aware_micro_encodings
        else:
            final_encoded_code_expressions = micro_encoded_code_expressions

        encodings_of_symbols_occurrences, symbols_indices_of_symbols_occurrences = None, None
        if self.symbols_encoder.encoder_params.use_symbols_occurrences:
            encodings_of_symbols_occurrences, symbols_indices_of_symbols_occurrences = \
                self.symbol_occurrences_extractor(
                    code_expression_encodings=final_encoded_code_expressions,
                    tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
                    sub_ast_expressions_input=code_task_input.pdg.cfg_nodes_expressions_ast,
                    method_ast_input=code_task_input.ast)
        encoded_symbols = self.symbols_encoder(
            encoded_identifiers=encoded_identifiers,
            symbols=code_task_input.symbols,
            encodings_of_symbols_occurrences=encodings_of_symbols_occurrences,
            symbols_indices_of_symbols_occurrences=symbols_indices_of_symbols_occurrences)

        return HierarchicMicroMacroMethodCodeEncodings(
            identifiers_encodings=encoded_identifiers,
            micro_encodings=micro_encoded_code_expressions,
            macro_encodings=macro_encodings.macro_encodings,
            global_context_aware_micro_encodings=global_context_aware_micro_encodings,
            symbols_encodings=encoded_symbols)
