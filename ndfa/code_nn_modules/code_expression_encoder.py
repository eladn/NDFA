import torch
import torch.nn as nn

from .code_expression_tokens_sequence_encoder import CodeExpressionTokensSequenceEncoder
from .ast_paths_encoder import ASTPathsEncoder
from ndfa.ndfa_model_hyper_parameters import CodeExpressionEncoderParams
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.nn_utils.modules.sequence_combiner import SequenceCombiner
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors, \
    MethodASTInputTensors, SubASTInputTensors


__all__ = ['CodeExpressionEncoder']


class CodeExpressionEncoder(nn.Module):
    def __init__(
            self, encoder_params: CodeExpressionEncoderParams,
            code_task_vocabs: CodeTaskVocabs,
            identifier_embedding_dim: int,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionEncoder, self).__init__()
        self.encoder_params = encoder_params
        self.identifier_embedding_dim = identifier_embedding_dim
        if self.encoder_params.encoder_type == 'tokens-seq':
            self.code_expression_linear_seq_encoder = CodeExpressionTokensSequenceEncoder(
                kos_tokens_vocab=code_task_vocabs.kos_tokens,
                tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
                expressions_special_words_vocab=code_task_vocabs.expressions_special_words,
                identifiers_special_words_vocab=code_task_vocabs.identifiers_special_words,
                encoder_params=self.encoder_params,
                identifier_embedding_dim=self.identifier_embedding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.encoder_type == 'ast':
            # TODO: plug-in these params from HPs
            ast_node_embedding_dim = self.encoder_params.token_encoding_dim
            self.ast_node_embedding_dim = ast_node_embedding_dim
            self.primitive_type_embedding_dim = self.ast_node_embedding_dim
            self.modifier_embedding_dim = self.ast_node_embedding_dim
            self.ast_paths_encoder = ASTPathsEncoder(
                ast_node_embedding_dim=self.ast_node_embedding_dim,
                identifier_encoding_dim=self.identifier_embedding_dim,
                primitive_type_embedding_dim=self.primitive_type_embedding_dim,
                modifier_embedding_dim=self.modifier_embedding_dim,
                encoder_params=self.encoder_params.ast_encoder,
                is_first_encoder_layer=True,
                ast_node_type_vocab=code_task_vocabs.ast_node_types,
                primitive_types_vocab=code_task_vocabs.primitive_types,
                modifiers_vocab=code_task_vocabs.modifiers,
                ast_traversal_orientation_vocab=code_task_vocabs.ast_traversal_orientation,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        else:
            raise ValueError(f'Unsupported expression encoder type `{self.encoder_params.encoder_type}`.')

    def forward(
            self,
            tokenized_expressions_input: CodeExpressionTokensSequenceInputTensors,
            method_ast_input: MethodASTInputTensors,
            sub_ast_input: SubASTInputTensors,
            encoded_identifiers: torch.Tensor):
        if self.encoder_params.encoder_type == 'tokens-seq':
            return self.code_expression_linear_seq_encoder(
                expressions_input=tokenized_expressions_input,
                encoded_identifiers=encoded_identifiers)
        elif self.encoder_params.encoder_type == 'ast':
            self.ast_paths_type = 'leaf_to_leaf'  # TODO: make something about this param!
            return self.ast_paths_encoder(
                ast_paths_node_indices=sub_ast_input.ast_leaf_to_leaf_paths_node_indices if self.ast_paths_type == 'leaf_to_leaf' else sub_ast_input.ast_leaf_to_root_paths_node_indices,
                method_ast_input=method_ast_input,
                ast_nodes_representation_previous_states=None,  # TODO
                ast_paths_child_place=sub_ast_input.ast_leaf_to_leaf_paths_child_place if self.ast_paths_type == 'leaf_to_leaf' else sub_ast_input.ast_leaf_to_root_paths_child_place,
                ast_paths_vertical_direction=sub_ast_input.ast_leaf_to_leaf_paths_vertical_direction if self.ast_paths_type == 'leaf_to_leaf' else None,
                ast_paths_last_states_for_nodes=None,  # TODO
                ast_paths_last_states_for_traversal_order=None,  # TODO
                identifiers_encodings=encoded_identifiers)
        else:
            assert False
