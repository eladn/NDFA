import torch
import torch.nn as nn
from typing import Optional

from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_nn_modules.code_task_input import CodeExpressionTokensSequenceInputTensors, \
    PDGExpressionsSubASTInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors
from ndfa.nn_utils.modules.sequence_combiner import SequenceCombiner
from ndfa.code_nn_modules.cfg_node_sub_ast_expression_combiner import CFGSubASTExpressionCombiner
from ndfa.nn_utils.modules.params.sequence_combiner_params import SequenceCombinerParams


__all__ = ['CodeExpressionCombiner']


class CodeExpressionCombiner(nn.Module):
    def __init__(self,
                 encoder_params: CodeExpressionEncoderParams,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionCombiner, self).__init__()
        self.encoder_params = encoder_params
        if self.encoder_params.encoder_type == 'tokens-seq':
            self.tokenized_sequence_combiner = SequenceCombiner(
                encoding_dim=self.encoder_params.tokens_seq_encoder.token_encoding_dim,
                combined_dim=self.encoder_params.combined_expression_encoding_dim,
                combiner_params=self.encoder_params.tokenized_expression_combiner,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.encoder_type == 'ast':
            self.sub_ast_expression_combiner = CFGSubASTExpressionCombiner(
                ast_node_encoding_dim=self.encoder_params.ast_encoder.ast_node_embedding_dim,
                combined_dim=self.encoder_params.combined_expression_encoding_dim,  # TODO!
                combining_params=self.encoder_params.cfg_sub_ast_expression_combiner_params,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        else:
            raise ValueError(f'Unsupported expression encoder type `{self.encoder_params.encoder_type}`.')

    def forward(
            self,
            encoded_code_expressions: CodeExpressionEncodingsTensors,
            tokenized_expressions_input: Optional[CodeExpressionTokensSequenceInputTensors] = None,
            cfg_nodes_expressions_ast: Optional[PDGExpressionsSubASTInputTensors] = None,
            cfg_nodes_has_expression_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.encoder_params.encoder_type == 'tokens-seq':
            return self.tokenized_sequence_combiner(
                sequence_encodings=encoded_code_expressions.token_seqs,
                sequence_lengths=tokenized_expressions_input.token_type.sequences_lengths)
        elif self.encoder_params.encoder_type == 'ast':
            combined_expressions = self.sub_ast_expression_combiner(
                encoded_code_expressions=encoded_code_expressions,
                cfg_nodes_expressions_ast_input=cfg_nodes_expressions_ast,
                nr_cfg_nodes=cfg_nodes_has_expression_mask.size(0))
            # This assert is heavy because it blocks the computation (requires sync copy from gpu to cpu).
            # assert torch.all(
            #     cfg_nodes_expressions_ast.pdg_node_idx_to_sub_ast_root_idx_mapping_key.indices
            #     == torch.nonzero(cfg_nodes_has_expression_mask.long(), as_tuple=False)
            #     .view(-1)).item()
            combined_expressions = combined_expressions[
                cfg_nodes_has_expression_mask]  # TODO: solve this problem in a more elegant way.
            return combined_expressions
        else:
            assert False
