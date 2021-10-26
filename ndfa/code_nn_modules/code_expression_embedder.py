import torch
import torch.nn as nn
from typing import Optional

from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_tasks.code_task_vocabs import CodeTaskVocabs
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors, MethodASTInputTensors
from ndfa.code_nn_modules.code_tokens_embedder import CodeTokensEmbedder
from ndfa.code_nn_modules.ast_nodes_embedder import ASTNodesEmbedder
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors


__all__ = ['CodeExpressionEmbedder']


class CodeExpressionEmbedder(nn.Module):
    def __init__(self, code_task_vocabs: CodeTaskVocabs,
                 encoder_params: CodeExpressionEncoderParams,
                 identifier_embedding_dim: int,
                 nr_final_embeddings_linear_layers: int = 1,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionEmbedder, self).__init__()
        self.encoder_params = encoder_params
        self.identifier_embedding_dim = identifier_embedding_dim
        if self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
            self.code_tokens_embedder = CodeTokensEmbedder(
                kos_tokens_vocab=code_task_vocabs.kos_tokens,
                tokens_kinds_vocab=code_task_vocabs.tokens_kinds,
                token_encoding_dim=self.encoder_params.tokens_seq_encoder.token_encoding_dim,
                kos_token_embedding_dim=self.encoder_params.tokens_seq_encoder.kos_token_embedding_dim,
                token_type_embedding_dim=self.encoder_params.tokens_seq_encoder.token_type_embedding_dim,
                identifier_embedding_dim=self.identifier_embedding_dim,
                nr_out_linear_layers=nr_final_embeddings_linear_layers,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.AST:
            # TODO: put in HPs
            self.primitive_type_embedding_dim = self.encoder_params.ast_encoder.ast_node_embedding_dim
            self.modifier_embedding_dim = self.encoder_params.ast_encoder.ast_node_embedding_dim

            self.ast_nodes_embedder = ASTNodesEmbedder(
                ast_node_embedding_dim=self.encoder_params.ast_encoder.ast_node_embedding_dim,
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
        else:
            raise ValueError(f'Unsupported expression encoder type `{self.encoder_params.encoder_type}`.')

    def forward(
            self,
            encoded_identifiers: torch.Tensor,
            tokenized_expressions_input: Optional[CodeExpressionTokensSequenceInputTensors] = None,
            method_ast_input: Optional[MethodASTInputTensors] = None) -> CodeExpressionEncodingsTensors:
        if self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
            return CodeExpressionEncodingsTensors(token_seqs=self.code_tokens_embedder(
                token_type=tokenized_expressions_input.token_type.sequences,
                kos_token_index=tokenized_expressions_input.kos_token_index.tensor,
                identifier_index=tokenized_expressions_input.identifier_index.indices,
                encoded_identifiers=encoded_identifiers))
        elif self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.AST:
            return CodeExpressionEncodingsTensors(ast_nodes=self.ast_nodes_embedder(
                method_ast_input=method_ast_input,
                identifiers_encodings=encoded_identifiers))
        else:
            assert False
