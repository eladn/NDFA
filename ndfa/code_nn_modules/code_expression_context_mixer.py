__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-05"

import torch
import torch.nn as nn
import dataclasses
from typing import Optional, Dict, Tuple

from ndfa.nn_utils.modules.seq_context_adder import SeqContextAdder
from ndfa.nn_utils.modules.state_updater import StateUpdater
from ndfa.nn_utils.modules.params.state_updater_params import StateUpdaterParams
from ndfa.nn_utils.modules.norm_wrapper import NormWrapper
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.code_nn_modules.code_task_input import PDGExpressionsSubASTInputTensors
from ndfa.code_nn_modules.code_task_input import PDGInputTensors
from ndfa.code_nn_modules.code_expression_encodings_tensors import CodeExpressionEncodingsTensors, \
    ASTPathsEncodingsTensors
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams


__all__ = ['CodeExpressionContextMixer']


class CodeExpressionContextMixer(nn.Module):
    def __init__(
            self,
            cfg_node_encoding_dim: int,
            macro_context_to_micro_state_updater: StateUpdaterParams,
            encoder_params: CodeExpressionEncoderParams,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(CodeExpressionContextMixer, self).__init__()
        self.encoder_params = encoder_params
        if self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
            # TODO: make the `SeqContextAdder` accept `macro_context_to_micro_state_updater` like `StateUpdater`.
            self.tokenized_expression_context_adder = SeqContextAdder(
                main_dim=self.encoder_params.tokens_seq_encoder.token_encoding_dim,
                ctx_dim=cfg_node_encoding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
        elif self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.AST:
            if self.encoder_params.ast_encoder.encoder_type in \
                    {ASTEncoderParams.EncoderType.PathsFolded,
                     ASTEncoderParams.EncoderType.Tree,
                     ASTEncoderParams.EncoderType.GNN}:
                self.macro_context_adder_to_sub_ast_nodes = MacroContextAdderToSubASTNodes(
                    ast_node_encoding_dim=self.encoder_params.ast_encoder.ast_node_embedding_dim,
                    cfg_node_encoding_dim=cfg_node_encoding_dim,
                    state_updater_params=macro_context_to_micro_state_updater,
                    dropout_rate=dropout_rate, activation_fn=activation_fn)
            elif self.encoder_params.ast_encoder.encoder_type == ASTEncoderParams.EncoderType.SetOfPaths:
                self.macro_context_adder_to_sub_ast_paths = MacroContextAdderToSubASTPaths(
                    ast_paths_types=self.encoder_params.ast_encoder.ast_paths_types,
                    ast_node_encoding_dim=self.encoder_params.ast_encoder.ast_node_embedding_dim,
                    cfg_node_encoding_dim=cfg_node_encoding_dim,
                    state_updater_params=macro_context_to_micro_state_updater,
                    dropout_rate=dropout_rate, activation_fn=activation_fn)
            else:
                assert False
        else:
            raise ValueError(f'Unsupported expression encoder type `{self.encoder_params.encoder_type}`.')
        self.expressions_norm = None if norm_params is None else NormWrapper(
            nr_features=self.encoder_params.expression_encoding_dim, params=norm_params)

    def forward(
            self,
            encoded_code_expressions: CodeExpressionEncodingsTensors,
            encoded_cfg_nodes: torch.Tensor,
            pdg_input: PDGInputTensors) -> CodeExpressionEncodingsTensors:
        # Add CFG-node macro context to its own expression (tokens-seq / sub-AST).
        # FIXME: notice that in 'FlatTokensSeq' case, there is a skip-connection built-in in `SeqContextAdder`.
        #  Is this also the case for the 'ast' case in `StateUpdater`?
        if self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
            new_token_seqs_encodings = self.tokenized_expression_context_adder(
                sequence=encoded_code_expressions.token_seqs,
                sequence_mask=pdg_input.cfg_nodes_tokenized_expressions.token_type.sequences_mask,
                context=encoded_cfg_nodes[pdg_input.cfg_nodes_has_expression_mask.tensor])
            if self.expressions_norm is not None:
                new_token_seqs_encodings = self.expressions_norm(new_token_seqs_encodings)
            return dataclasses.replace(encoded_code_expressions, token_seqs=new_token_seqs_encodings)
        elif self.encoder_params.encoder_type == CodeExpressionEncoderParams.EncoderType.AST:
            if self.encoder_params.ast_encoder.encoder_type in \
                    {ASTEncoderParams.EncoderType.PathsFolded,
                     ASTEncoderParams.EncoderType.Tree,
                     ASTEncoderParams.EncoderType.GNN}:
                new_ast_nodes_encodings = self.macro_context_adder_to_sub_ast_nodes(
                    previous_ast_nodes_encodings=encoded_code_expressions.ast_nodes,
                    new_cfg_nodes_encodings=encoded_cfg_nodes,
                    cfg_expressions_sub_ast_input=pdg_input.cfg_nodes_expressions_ast)
                if self.expressions_norm is not None:
                    new_ast_nodes_encodings = self.expressions_norm(new_ast_nodes_encodings)
                return dataclasses.replace(encoded_code_expressions, ast_nodes=new_ast_nodes_encodings)
            elif self.encoder_params.ast_encoder.encoder_type == ASTEncoderParams.EncoderType.SetOfPaths:
                new_ast_paths_encodings = self.macro_context_adder_to_sub_ast_paths(
                    nr_ast_nodes=encoded_code_expressions.ast_nodes.size(0),
                    previous_ast_paths_encodings=encoded_code_expressions.ast_paths_by_type,
                    new_cfg_nodes_encodings=encoded_cfg_nodes,
                    cfg_expressions_sub_ast_input=pdg_input.cfg_nodes_expressions_ast)
                if self.expressions_norm is not None:
                    new_ast_paths_encodings = {
                        paths_type: dataclasses.replace(
                            paths, nodes_occurrences=self.expressions_norm(paths.nodes_occurrences))
                        for paths_type, paths in new_ast_paths_encodings.items()}
                return dataclasses.replace(encoded_code_expressions, ast_paths_by_type=new_ast_paths_encodings)
            else:
                assert False
        else:
            assert False


class MacroContextAdderToSubASTNodes(nn.Module):
    def __init__(
            self,
            ast_node_encoding_dim: int,
            cfg_node_encoding_dim: int,
            state_updater_params: StateUpdaterParams,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(MacroContextAdderToSubASTNodes, self).__init__()
        self.ast_node_encoding_dim = ast_node_encoding_dim
        self.cfg_node_encoding_dim = cfg_node_encoding_dim
        self.gate = StateUpdater(
            state_dim=self.ast_node_encoding_dim, params=state_updater_params,
            update_dim=self.cfg_node_encoding_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)

    def forward(
            self, previous_ast_nodes_encodings: torch.Tensor,
            new_cfg_nodes_encodings: torch.Tensor,
            cfg_expressions_sub_ast_input: PDGExpressionsSubASTInputTensors):
        ast_nodes_encodings_updates = self.gate(
            previous_state=previous_ast_nodes_encodings[
                cfg_expressions_sub_ast_input.ast_node_idx_to_pdg_node_idx_mapping_key.indices],
            state_update=new_cfg_nodes_encodings[
                cfg_expressions_sub_ast_input.ast_node_idx_to_pdg_node_idx_mapping_value.indices])
        # Note: We do NOT update AST nodes that haven't related CFG node.
        return previous_ast_nodes_encodings.scatter(
            dim=0,
            index=cfg_expressions_sub_ast_input.ast_node_idx_to_pdg_node_idx_mapping_key.indices
                .unsqueeze(-1).expand(ast_nodes_encodings_updates.shape),
            src=ast_nodes_encodings_updates)


class MacroContextAdderToSubASTPaths(nn.Module):
    def __init__(
            self,
            ast_paths_types: Tuple[str],
            ast_node_encoding_dim: int,
            cfg_node_encoding_dim: int,
            state_updater_params: StateUpdaterParams,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(MacroContextAdderToSubASTPaths, self).__init__()
        self.ast_node_encoding_dim = ast_node_encoding_dim
        self.cfg_node_encoding_dim = cfg_node_encoding_dim
        self.state_updater_by_paths_type = nn.ModuleDict({
            paths_type: StateUpdater(
                state_dim=self.ast_node_encoding_dim, params=state_updater_params,
                update_dim=self.cfg_node_encoding_dim,
                dropout_rate=dropout_rate, activation_fn=activation_fn)
            for paths_type in ast_paths_types})

    def forward(
            self,
            nr_ast_nodes: int,
            previous_ast_paths_encodings: Dict[str, ASTPathsEncodingsTensors],
            new_cfg_nodes_encodings: torch.Tensor,
            cfg_expressions_sub_ast_input: PDGExpressionsSubASTInputTensors):
        new_cfg_nodes_encodings_ordered_by_ast_pdg_mapping = new_cfg_nodes_encodings[
            cfg_expressions_sub_ast_input.ast_node_idx_to_pdg_node_idx_mapping_value.indices]
        cfg_nodes_encodings_scattered_by_related_ast_node_idx = \
            new_cfg_nodes_encodings.new_zeros(size=(nr_ast_nodes, self.ast_node_encoding_dim)).scatter(
                dim=0,
                index=cfg_expressions_sub_ast_input.ast_node_idx_to_pdg_node_idx_mapping_key.indices
                    .unsqueeze(-1).expand(new_cfg_nodes_encodings_ordered_by_ast_pdg_mapping.shape),
                src=new_cfg_nodes_encodings_ordered_by_ast_pdg_mapping)
        # Note: We also update encodings of AST nodes (occurrences) that haven't related CFG node.
        #       These are being updated with a zero vector.
        new_ast_paths_encodings = {
            paths_type: dataclasses.replace(paths, nodes_occurrences=self.state_updater_by_paths_type[paths_type](
                previous_state=paths.nodes_occurrences,
                state_update=cfg_nodes_encodings_scattered_by_related_ast_node_idx[
                    cfg_expressions_sub_ast_input.get_ast_paths_node_indices(path_type=paths_type).sequences]))
            for paths_type, paths in previous_ast_paths_encodings.items()
        }
        return new_ast_paths_encodings
