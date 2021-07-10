import torch
import torch.nn as nn
from typing import Tuple

from ndfa.code_nn_modules.code_task_input import MethodCodeInputTensors
from ndfa.code_nn_modules.params.method_code_encoder_params import MethodCodeEncoderParams
from ndfa.code_nn_modules.method_code_encoder import EncodedMethodCode


__all__ = ['MethodCodeEncodingsFeeder']


class MethodCodeEncodingsFeeder(nn.Module):
    """
    (1) selects the relevant method encoder output tensors to feed the decoder based on the selected encoding approach
    (2) unflattens these tensors to have 1st dim of #example as the decoder expects
    (3) concatenates multiple tensors if necessary
    (4) also return the matching output mask (the created outputs now includes paddings because we unflattened
        to example-based tensors we created paddings)
    """
    def __init__(self, method_code_encoder_params: MethodCodeEncoderParams):
        super(MethodCodeEncodingsFeeder, self).__init__()
        self.method_code_encoder_params = method_code_encoder_params

    def forward(self, code_task_input: MethodCodeInputTensors, encoded_method_code: EncodedMethodCode) \
            -> Tuple[torch.Tensor, torch.BoolTensor]:
        if self.method_code_encoder_params.method_encoder_type == 'method-cfg':
            encoder_outputs = encoded_method_code.encoded_cfg_nodes_after_bridge
            encoder_outputs_mask = code_task_input.pdg.cfg_nodes_control_kind.unflattener_mask
        elif self.method_code_encoder_params.method_encoder_type == 'method-cfg-v2':
            encoder_outputs = encoded_method_code.encoded_cfg_nodes_after_bridge
            encoder_outputs_mask = code_task_input.pdg.cfg_nodes_control_kind.unflattener_mask
        elif self.method_code_encoder_params.method_encoder_type == 'whole-method':
            if self.method_code_encoder_params.whole_method_expression_encoder.encoder_type == 'FlatTokensSeq':
                encoder_outputs = encoded_method_code.whole_method_token_seqs_encoding
                encoder_outputs_mask = code_task_input.method_tokenized_code.token_type.sequences_mask
            elif self.method_code_encoder_params.whole_method_expression_encoder.encoder_type == 'ast':
                if self.method_code_encoder_params.whole_method_expression_encoder.ast_encoder.encoder_type == \
                        'set-of-paths':
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
                        {'tree', 'paths-folded'}:
                    encoder_outputs = code_task_input.ast.ast_node_major_types.unflatten(
                        encoded_method_code.whole_method_ast_nodes_encoding)
                    encoder_outputs_mask = code_task_input.ast.ast_node_major_types.unflattener_mask
                else:
                    assert False
            else:
                assert False
            # print('ast_nodes_with_symbol_leaf_nodes - types',
            #       code_task_input.ast.ast_node_types.tensor[code_task_input.ast.ast_nodes_with_symbol_leaf_nodes_indices.indices])
            # print('#zero(encodings_of_symbols_occurrences)',
            #       torch.sum(torch.all(torch.isclose(encoded_code.whole_method_ast_nodes_encoding[code_task_input.ast.ast_leaves_sequence_node_indices.sequences], torch.tensor(0.0)), dim=1)))
            # print('ast_nodes_with_symbol_leaf_symbol_idx (per example)', code_task_input.ast.ast_nodes_with_symbol_leaf_symbol_idx.nr_items_per_example)
            # print('nr_ast_leaves (per example)', code_task_input.ast.ast_leaves_sequence_node_indices.sequences_lengths)
            # print('ast_node_types', code_task_input.ast.ast_node_types.tensor.shape)
            # print('ast_node_types', code_task_input.ast.ast_node_types.nr_items_per_example)
            # print('encoder_outputs', encoder_outputs.shape)
            # print('encoder_outputs_mask', encoder_outputs_mask.shape)
        else:
            assert False

        return encoder_outputs, encoder_outputs_mask
