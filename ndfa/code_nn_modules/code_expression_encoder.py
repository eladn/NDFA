import torch
import torch.nn as nn

from .code_expression_tokens_sequence_encoder import CodeExpressionTokensSequenceEncoder
from ndfa.code_nn_modules.ast_nodes_embedder import ASTNodesEmbedder
from ndfa.code_nn_modules.ast_paths_encoder import ASTPathsEncoder
from ndfa.code_nn_modules.ast_tree_lstm_encoder import ASTTreeLSTMEncoder
from ndfa.ndfa_model_hyper_parameters import CodeExpressionEncoderParams
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
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
                encoder_params=self.encoder_params,
                identifier_embedding_dim=self.identifier_embedding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.encoder_type in {'ast_paths', 'ast_treelstm'}:
            # TODO: plug-in these params from HPs
            ast_node_embedding_dim = self.encoder_params.token_encoding_dim
            self.ast_node_embedding_dim = ast_node_embedding_dim
            self.primitive_type_embedding_dim = self.ast_node_embedding_dim
            self.modifier_embedding_dim = self.ast_node_embedding_dim
            self.ast_nodes_embedder = ASTNodesEmbedder(
                ast_node_embedding_dim=self.ast_node_embedding_dim,
                identifier_encoding_dim=self.identifier_embedding_dim,
                primitive_type_embedding_dim=self.primitive_type_embedding_dim,
                modifier_embedding_dim=self.modifier_embedding_dim,
                ast_node_type_vocab=code_task_vocabs.ast_node_types,
                ast_node_major_type_vocab=code_task_vocabs.ast_node_major_types,
                ast_node_minor_type_vocab=code_task_vocabs.ast_node_minor_types,
                ast_node_nr_children_vocab=code_task_vocabs.ast_node_nr_children,
                ast_node_child_pos_vocab=code_task_vocabs.ast_node_child_pos,
                primitive_types_vocab=code_task_vocabs.primitive_types,
                modifiers_vocab=code_task_vocabs.modifiers,
                dropout_rate=dropout_rate, activation_fn=activation_fn)

            if self.encoder_params.encoder_type == 'ast_paths':
                self.ast_paths_encoder = ASTPathsEncoder(
                    ast_node_embedding_dim=self.ast_node_embedding_dim,
                    encoder_params=self.encoder_params.ast_encoder,
                    is_first_encoder_layer=True,
                    ast_traversal_orientation_vocab=code_task_vocabs.ast_traversal_orientation,
                    dropout_rate=dropout_rate, activation_fn=activation_fn)
            elif self.encoder_params.encoder_type == 'ast_treelstm':
                self.ast_treelstm_up = ASTTreeLSTMEncoder(
                    ast_node_embedding_dim=self.ast_node_embedding_dim,
                    direction='leaves_to_root', dropout_rate=dropout_rate)
                self.ast_treelstm_down = ASTTreeLSTMEncoder(
                    ast_node_embedding_dim=self.ast_node_embedding_dim,
                    direction='root_to_leaves', dropout_rate=dropout_rate)
            else:
                assert False
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
        elif self.encoder_params.encoder_type in {'ast_paths', 'ast_treelstm'}:
            self.ast_paths_type = 'leaf_to_leaf'  # TODO: make something about this param!
            ast_nodes_embeddings = self.ast_nodes_embedder(
                method_ast_input=method_ast_input,
                identifiers_encodings=encoded_identifiers)
            if self.encoder_params.encoder_type == 'ast_paths':
                return self.ast_paths_encoder(
                    ast_nodes_encodings=ast_nodes_embeddings,
                    ast_paths_node_indices=sub_ast_input.get_ast_paths_node_indices(self.ast_paths_type),
                    ast_paths_child_place=sub_ast_input.get_ast_paths_child_place(self.ast_paths_type),
                    ast_paths_vertical_direction=sub_ast_input.get_ast_paths_vertical_direction(self.ast_paths_type),
                    ast_paths_last_states_for_nodes=None,  # TODO
                    ast_paths_last_states_for_traversal_order=None)  # TODO
            elif self.encoder_params.encoder_type == 'ast_treelstm':
                ast_nodes_encodings = self.ast_treelstm_up(
                    ast_nodes_embeddings=ast_nodes_embeddings,
                    ast_batch=sub_ast_input.dgl_tree)
                ast_nodes_encodings = self.ast_treelstm_down(
                    ast_nodes_embeddings=ast_nodes_encodings,
                    ast_batch=sub_ast_input.dgl_tree)
                return ast_nodes_encodings
        else:
            assert False
