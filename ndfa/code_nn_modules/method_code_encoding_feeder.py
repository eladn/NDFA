__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-07-10"

import torch
import torch.nn as nn
from typing import Tuple

from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_nn_modules.params.method_code_encoder_params import MethodCodeEncoderParams
from ndfa.code_nn_modules.params.ast_encoder_params import ASTEncoderParams
from ndfa.code_nn_modules.params.hierarchic_micro_macro_method_code_encoder_params import \
    HierarchicMicroMacroMethodCodeEncoderParams
from ndfa.code_nn_modules.method_code_encoder import EncodedMethodCode
from ndfa.code_nn_modules.params.code_expression_encoder_params import CodeExpressionEncoderParams


__all__ = ['MethodCodeEncodingsFeeder']


class MethodCodeEncodingsFeeder(nn.Module):
    """
    (1) Selects the relevant method encoder output tensors to feed the decoder,
        based on the selected encoding approach.
    (2) Unflattens these tensors to have 1st dim of #example as the decoder expects.
    (3) concatenates multiple tensors if necessary.
    (4) also return the matching output mask (the created outputs now includes paddings,
        because we unflattened to example-based tensors).

    TODO: Return a sole embedding per example for the initial state of the LSTM decoder.
          We currently feed only the attentive "memory bank" of the decoder.
    TODO: Add hyper-params to control what tensors to return.
          Relevant when there multiple possibilities. For CFG-method encoder we can pass:
          (i) CFG nodes; or (ii) CFG paths; or (iii) expressions (after mixing with their global ctx).
    """
    def __init__(self, method_code_encoder_params: MethodCodeEncoderParams):
        super(MethodCodeEncodingsFeeder, self).__init__()
        self.method_code_encoder_params = method_code_encoder_params

    def forward(self, code_task_input: MethodCodeInputTensors, encoded_method_code: EncodedMethodCode) \
            -> Tuple[torch.Tensor, torch.BoolTensor]:
        if self.method_code_encoder_params.method_encoder_type == MethodCodeEncoderParams.EncoderType.MethodCFG:
            encoder_outputs = encoded_method_code.encoded_cfg_nodes_after_bridge
            encoder_outputs_mask = code_task_input.pdg.cfg_nodes_control_kind.unflattener_mask
        elif self.method_code_encoder_params.method_encoder_type == MethodCodeEncoderParams.EncoderType.MethodCFGV2:
            encoder_outputs = encoded_method_code.encoded_cfg_nodes_after_bridge
            encoder_outputs_mask = code_task_input.pdg.cfg_nodes_control_kind.unflattener_mask
        elif self.method_code_encoder_params.method_encoder_type == MethodCodeEncoderParams.EncoderType.Hierarchic:
            if self.method_code_encoder_params.hierarchic_micro_macro_encoder.decoder_feeding_policy == \
                    HierarchicMicroMacroMethodCodeEncoderParams.DecoderFeedingPolicy.MacroItems:
                unflattanable_encodings = encoded_method_code.macro_encodings
            elif self.method_code_encoder_params.hierarchic_micro_macro_encoder.decoder_feeding_policy == \
                    HierarchicMicroMacroMethodCodeEncoderParams.DecoderFeedingPolicy.MicroItems:
                unflattanable_encodings = encoded_method_code.micro_encodings
            else:
                assert False
            encoder_outputs = unflattanable_encodings.get_unflattened()
            encoder_outputs_mask = unflattanable_encodings.get_unflattener_mask()
            # assert torch.allclose(encoder_outputs[~encoder_outputs_mask], torch.tensor(.0))
        elif self.method_code_encoder_params.method_encoder_type == MethodCodeEncoderParams.EncoderType.WholeMethod:
            if self.method_code_encoder_params.whole_method_expression_encoder.encoder_type == \
                    CodeExpressionEncoderParams.EncoderType.FlatTokensSeq:
                encoder_outputs = encoded_method_code.whole_method_token_seqs_encoding
                encoder_outputs_mask = code_task_input.method_tokenized_code.token_type.sequences_mask
            elif self.method_code_encoder_params.whole_method_expression_encoder.encoder_type == \
                    CodeExpressionEncoderParams.EncoderType.AST:
                if self.method_code_encoder_params.whole_method_expression_encoder.ast_encoder.encoder_type == \
                        ASTEncoderParams.EncoderType.SetOfPaths:
                    # TODO: for 'leaves_sequence' we might want to have the whole sequence rather than the combined path
                    # TODO: is it ok that the outputs are defragmented?
                    #  (the masks might have `True` after a `False` for the same examples)
                    ast_paths_by_type = encoded_method_code.whole_method_combined_ast_paths_encoding_by_type
                    all_encoder_outputs = [
                        code_task_input.ast.get_ast_paths_node_indices(path_type).unflatten(ast_paths)
                        for path_type, ast_paths in ast_paths_by_type.items()]
                    all_encoder_outputs_mask = [
                        code_task_input.ast.get_ast_paths_node_indices(path_type).unflattener_mask
                        for path_type, ast_paths in ast_paths_by_type.items()]
                    assert all(enc.shape[:-1] == mask.shape
                               for enc, mask in zip(all_encoder_outputs, all_encoder_outputs_mask))
                    assert len(all_encoder_outputs) >= 1 and len(all_encoder_outputs_mask) >= 1
                    encoder_outputs = all_encoder_outputs[0] if len(all_encoder_outputs) == 1 else \
                        torch.cat(all_encoder_outputs, dim=1)
                    encoder_outputs_mask = all_encoder_outputs_mask[0] if len(all_encoder_outputs_mask) == 1 else \
                        torch.cat(all_encoder_outputs_mask, dim=1)
                    assert encoder_outputs.shape[:-1] == encoder_outputs_mask.shape
                elif self.method_code_encoder_params.whole_method_expression_encoder.ast_encoder.encoder_type in \
                        {ASTEncoderParams.EncoderType.Tree,
                         ASTEncoderParams.EncoderType.PathsFolded,
                         ASTEncoderParams.EncoderType.GNN}:
                    encoder_outputs = code_task_input.ast.ast_node_major_types.unflatten(
                        encoded_method_code.whole_method_ast_nodes_encoding)
                    encoder_outputs_mask = code_task_input.ast.ast_node_major_types.unflattener_mask
                else:
                    assert False
            else:
                assert False
        else:
            assert False

        assert encoder_outputs.ndim == 3
        assert encoder_outputs.shape[:-1] == encoder_outputs_mask.shape
        return encoder_outputs, encoder_outputs_mask
