__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-05"

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from ndfa.code_nn_modules.params.hierarchic_micro_macro_method_code_encoder_params import \
    HierarchicMicroMacroMethodCodeEncoderParams as EncoderParams
from ndfa.code_nn_modules.code_expression_encoder_with_combiner import CodeExpressionEncoderWithCombiner
from ndfa.code_nn_modules.code_expression_multistage_encoder import CodeExpressionMultistageEncoder
from ndfa.code_nn_modules.code_expression_multistage_encoder_with_combiner import \
    CodeExpressionMultistageEncoderWithCombiner
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
from ndfa.code_nn_modules.micro_code_expression_encodings_unflattener import \
    micro_code_expression_encodings_as_unflattenable


__all__ = ['HierarchicMicroMacroMethodCodeEncoder', 'HierarchicMicroMacroMethodCodeEncodings']


@dataclass
class HierarchicMicroMacroMethodCodeEncodings:
    identifiers_encodings: torch.Tensor
    unflattenable_final_micro_encodings: FlattenedTensor
    macro_encodings: FlattenedTensor
    # micro_encodings: CodeExpressionEncodingsTensors
    # global_context_aware_micro_encodings: CodeExpressionEncodingsTensors
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
        self.code_expression_encoders = nn.ModuleList([
            CodeExpressionMultistageEncoderWithCombiner(
                encoder_params=self.params.local_expression_encoder,
                code_task_vocabs=code_task_vocabs,
                identifier_embedding_dim=self.identifier_embedding_dim,
                # is_first_encoder_layer=True,  # (for `CodeExpressionEncoderWithCombiner`)
                nr_layers=self.params.nr_micro_encoding_layers_before_macro,
                reuse_inner_encodings_from_previous_input_layer=False,
                reuse_inner_encodings_between_layers=self.params.reuse_inner_encodings_between_micro_layers,
                norm_params=norm_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for _ in range(self.params.nr_layers)])

        self.macro_encoders = nn.ModuleList([
            MethodCFGMacroEncoder(
                params=self.params.global_context_encoder,
                combined_micro_expression_dim=self.params.local_expression_encoder.combined_expression_encoding_dim,
                identifier_embedding_dim=identifier_embedding_dim,
                code_task_vocabs=code_task_vocabs,
                norm_params=norm_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for _ in range(self.params.nr_layers)])

        self.should_mix_code_expression_with_global_context = not self.params.force_no_local_global_mix and not (
            self.params.global_context_encoder.encoder_type == MethodCFGMacroEncoderParams.EncoderType.CFGPaths and
            self.params.global_context_encoder.paths_encoder.output_type ==
            CFGPathsMacroEncoderParams.OutputType.SetOfPaths)

        if self.should_mix_code_expression_with_global_context:
            self.code_expression_global_context_mixer = nn.ModuleList([
                CodeExpressionContextMixer(
                    cfg_node_encoding_dim=params.global_context_encoder.cfg_node_encoding_dim,
                    macro_context_to_micro_state_updater=
                    self.params.global_context_encoder.macro_context_to_micro_state_updater,
                    encoder_params=self.params.local_expression_encoder,
                    norm_params=norm_params,
                    dropout_rate=dropout_rate, activation_fn=activation_fn)
                for _ in range(self.params.nr_layers)])

            if self.params.after_macro.requires_micro:
                self.code_expression_encoder_after_macro = CodeExpressionMultistageEncoder(
                    encoder_params=self.params.local_expression_encoder_after_macro,
                    code_task_vocabs=code_task_vocabs,
                    identifier_embedding_dim=self.identifier_embedding_dim,
                    nr_layers=self.params.nr_micro_encoding_layers_after_macro,
                    # Note: When it `False`, for AST paths encoder, it keeps the previous encodings (output of the
                    #  previous encoder layer) of the paths (node occurrences and edges), and just update them
                    #  (linear project) using the new AST nodes encodings after they mixed with the global ctx, while
                    #  keeping the edges encodings as they were before.
                    # TODO: We might consider trying setting it to `True` to check whether the update of the
                    #  occurrences (in the path) of AST node encodings is actually necessary.
                    reuse_inner_encodings_from_previous_input_layer=
                    not self.params.reuse_inner_encodings_between_micro_layers,
                    reuse_inner_encodings_between_layers=not self.params.reuse_inner_encodings_between_micro_layers,
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

        micro_encoded_code_expressions = embedded_code_expressions
        macro_encodings = None
        for layer_idx in range(self.params.nr_layers):
            micro_encoded_code_expressions: CodeExpressionEncodingsTensors = self.code_expression_encoders[layer_idx](
                previous_code_expression_encodings=micro_encoded_code_expressions,
                tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
                cfg_nodes_expressions_ast=code_task_input.pdg.cfg_nodes_expressions_ast,
                cfg_nodes_has_expression_mask=code_task_input.pdg.cfg_nodes_has_expression_mask.tensor)

            macro_encodings: MethodCFGMacroEncodings = self.macro_encoders[layer_idx](
                code_task_input=code_task_input,
                encoded_identifiers=encoded_identifiers,
                encoded_combined_code_expressions=micro_encoded_code_expressions.combined_expressions)

            # Note: For cases like CFG-Paths without node occurrences folding, where the encodings of the CFG nodes are
            #         NOT being updated with their global context, we should avoid here from the further steps of
            #         mixing the code expressions with the CFG nodes embedding and so on.
            #       The case of `NoMacro` is considered as should mix, as the CFG embeddings contain the CFG node kind
            #         (apart from the combined local top-level expression encoding). Thus, it contains information that
            #         is not available otherwise.
            if self.should_mix_code_expression_with_global_context:
                micro_encoded_code_expressions = self.code_expression_global_context_mixer[layer_idx](
                    encoded_code_expressions=micro_encoded_code_expressions,
                    encoded_cfg_nodes=macro_encodings.cfg_nodes_encodings,
                    pdg_input=code_task_input.pdg)

        if self.should_mix_code_expression_with_global_context and self.params.after_macro.requires_micro:
            micro_encoded_code_expressions: CodeExpressionEncodingsTensors = \
                self.code_expression_encoder_after_macro(
                    previous_code_expression_encodings=micro_encoded_code_expressions,
                    tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
                    sub_ast_input=code_task_input.pdg.cfg_nodes_expressions_ast)

        encodings_of_symbols_occurrences, symbols_indices_of_symbols_occurrences = None, None
        if self.symbols_encoder.encoder_params.use_symbols_occurrences:
            encodings_of_symbols_occurrences, symbols_indices_of_symbols_occurrences = \
                self.symbol_occurrences_extractor(
                    code_expression_encodings=micro_encoded_code_expressions,
                    tokenized_expressions_input=code_task_input.pdg.cfg_nodes_tokenized_expressions,
                    sub_ast_expressions_input=code_task_input.pdg.cfg_nodes_expressions_ast,
                    method_ast_input=code_task_input.ast)
        encoded_symbols = self.symbols_encoder(
            encoded_identifiers=encoded_identifiers,
            symbols=code_task_input.symbols,
            encodings_of_symbols_occurrences=encodings_of_symbols_occurrences,
            symbols_indices_of_symbols_occurrences=symbols_indices_of_symbols_occurrences)

        unflattenable_final_micro_encodings = micro_code_expression_encodings_as_unflattenable(
            micro_encoder_params=self.params.last_local_expression_encoder,
            code_task_input=code_task_input,
            code_expression_encodings=micro_encoded_code_expressions)

        return HierarchicMicroMacroMethodCodeEncodings(
            identifiers_encodings=encoded_identifiers,
            unflattenable_final_micro_encodings=unflattenable_final_micro_encodings,
            macro_encodings=macro_encodings.macro_encodings,
            # micro_encodings=micro_encoded_code_expressions,
            # global_context_aware_micro_encodings=global_context_aware_micro_encodings,
            symbols_encodings=encoded_symbols)
