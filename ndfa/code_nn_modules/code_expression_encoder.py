import torch
import torch.nn as nn
from typing import Optional, Tuple

from .code_expression_tokens_sequence_encoder import CodeExpressionTokensSequenceEncoder
from ndfa.code_nn_modules.ast_paths_encoder import ASTPathsEncoder
from ndfa.code_nn_modules.ast_tree_lstm_encoder import ASTTreeLSTMEncoder
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors, \
    SubASTInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors


__all__ = ['CodeExpressionEncoder']


class CodeExpressionEncoder(nn.Module):
    def __init__(
            self, encoder_params: CodeExpressionEncoderParams,
            code_task_vocabs: CodeTaskVocabs,
            identifier_embedding_dim: int,
            is_first_encoder_layer: bool = True,
            ast_paths_types: Tuple[str, ...] = ('leaf_to_leaf',),
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionEncoder, self).__init__()
        self.encoder_params = encoder_params
        self.is_first_encoder_layer = is_first_encoder_layer
        self.identifier_embedding_dim = identifier_embedding_dim
        if self.encoder_params.encoder_type == 'tokens-seq':
            self.code_expression_linear_seq_encoder = CodeExpressionTokensSequenceEncoder(
                encoder_params=self.encoder_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.encoder_type in {'ast_paths', 'ast_treelstm'}:
            # TODO: plug-in these params from HPs
            ast_node_embedding_dim = self.encoder_params.token_encoding_dim
            self.ast_node_embedding_dim = ast_node_embedding_dim
            self.primitive_type_embedding_dim = self.ast_node_embedding_dim
            self.modifier_embedding_dim = self.ast_node_embedding_dim

            if self.encoder_params.encoder_type == 'ast_paths':
                self.ast_paths_types = ast_paths_types
                self.ast_paths_encoder = ASTPathsEncoder(
                    ast_node_embedding_dim=self.ast_node_embedding_dim,
                    encoder_params=self.encoder_params.ast_encoder,
                    ast_paths_types=self.ast_paths_types,
                    is_first_encoder_layer=self.is_first_encoder_layer,
                    ast_traversal_orientation_vocab=code_task_vocabs.ast_traversal_orientation,
                    dropout_rate=dropout_rate, activation_fn=activation_fn)
            elif self.encoder_params.encoder_type == 'ast_treelstm':
                self.ast_treelstm_up = ASTTreeLSTMEncoder(
                    ast_node_embedding_dim=self.ast_node_embedding_dim,
                    direction='leaves_to_root', dropout_rate=dropout_rate)
                # self.ast_treelstm_down = ASTTreeLSTMEncoder(
                #     ast_node_embedding_dim=self.ast_node_embedding_dim,
                #     direction='root_to_leaves', dropout_rate=dropout_rate)
            else:
                assert False
        else:
            raise ValueError(f'Unsupported expression encoder type `{self.encoder_params.encoder_type}`.')

    def forward(
            self,
            previous_code_expression_encodings: CodeExpressionEncodingsTensors,
            tokenized_expressions_input: Optional[CodeExpressionTokensSequenceInputTensors] = None,
            sub_ast_input: Optional[SubASTInputTensors] = None) -> CodeExpressionEncodingsTensors:
        if self.encoder_params.encoder_type == 'tokens-seq':
            return self.code_expression_linear_seq_encoder(
                token_seqs_embeddings=previous_code_expression_encodings.token_seqs,
                expressions_input=tokenized_expressions_input)
        elif self.encoder_params.encoder_type in {'ast_paths', 'ast_treelstm'}:
            if self.encoder_params.encoder_type == 'ast_paths':
                return self.ast_paths_encoder(
                    ast_nodes_encodings=previous_code_expression_encodings.ast_nodes,
                    sub_ast_input=sub_ast_input,
                    ast_paths_last_states=previous_code_expression_encodings.ast_paths_by_type
                    if previous_code_expression_encodings.ast_paths_types == self.ast_paths_types else None)
            elif self.encoder_params.encoder_type == 'ast_treelstm':
                ast_nodes_encodings_up = self.ast_treelstm_up(
                    ast_nodes_embeddings=previous_code_expression_encodings.ast_nodes,
                    ast_batch=sub_ast_input.dgl_tree)
                # ast_nodes_encodings_down = self.ast_treelstm_down(
                #     ast_nodes_embeddings=previous_code_expression_encodings.ast_nodes,
                #     ast_batch=sub_ast_input.dgl_tree)
                return ast_nodes_encodings_up  # + ast_nodes_encodings_down
        else:
            assert False
