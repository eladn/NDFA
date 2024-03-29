__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-10-05"

import torch
import torch.nn as nn
from typing import Optional

from ndfa.code_nn_modules.params.cfg_single_path_macro_encoder_params import SingleFlatCFGNodesSeqMacroEncoderParams
from ndfa.nn_utils.modules.sequence_encoder import SequenceEncoder
from ndfa.code_nn_modules.code_task_input import PDGInputTensors
from ndfa.nn_utils.modules.params.norm_wrapper_params import NormWrapperParams
from ndfa.nn_utils.modules.norm_wrapper import NormWrapper


__all__ = ['SingleFlatCFGNodesSeqMacroEncoder']


class SingleFlatCFGNodesSeqMacroEncoder(nn.Module):
    def __init__(
            self,
            cfg_node_dim: int,
            params: SingleFlatCFGNodesSeqMacroEncoderParams,
            norm_params: Optional[NormWrapperParams] = None,
            dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        super(SingleFlatCFGNodesSeqMacroEncoder, self).__init__()
        self.cfg_node_dim = cfg_node_dim
        self.params = params
        self.sequence_encoder_layer = SequenceEncoder(
            encoder_params=params.path_sequence_encoder,
            input_dim=self.cfg_node_dim,
            dropout_rate=dropout_rate, activation_fn=activation_fn)
        self.norm = None if norm_params is None else NormWrapper(
            nr_features=self.cfg_node_dim, params=norm_params)

    def forward(
            self,
            pdg_input: PDGInputTensors,
            cfg_nodes_encodings: torch.Tensor):
        if self.params.cfg_nodes_order == SingleFlatCFGNodesSeqMacroEncoderParams.CFGNodesOrder.CodeTextualAppearance:
            unflattened_nodes_encodings = pdg_input.cfg_nodes_control_kind.unflatten(cfg_nodes_encodings)
        elif self.params.cfg_nodes_order == SingleFlatCFGNodesSeqMacroEncoderParams.CFGNodesOrder.Random:
            # TODO: fix to use `BatchedFlattenedIndicesPseudoRandomPermutation.permute()`!
            unflattened_nodes_encodings = cfg_nodes_encodings[pdg_input.cfg_nodes_random_permutation.permutations]
            unflattened_nodes_encodings = unflattened_nodes_encodings.masked_fill(
                ~pdg_input.cfg_nodes_control_kind.unflattener_mask.unsqueeze(-1)
                    .expand(unflattened_nodes_encodings.shape), 0)
        else:
            raise ValueError(f'Unsupported value `{self.params.cfg_nodes_order}` for `cfg_nodes_order`.')
        assert unflattened_nodes_encodings.ndim == 3
        assert unflattened_nodes_encodings.shape == \
               (pdg_input.nr_examples, pdg_input.cfg_nodes_control_kind.max_nr_items, cfg_nodes_encodings.size(1))
        path_encodings = self.sequence_encoder_layer(
            sequence_input=unflattened_nodes_encodings,
            lengths=pdg_input.cfg_nodes_control_kind.nr_items_per_example,
            batch_first=True).sequence
        assert path_encodings.shape == unflattened_nodes_encodings.shape

        if self.params.cfg_nodes_order == SingleFlatCFGNodesSeqMacroEncoderParams.CFGNodesOrder.CodeTextualAppearance:
            reflattened_nodes_encodings = pdg_input.cfg_nodes_control_kind.flatten(path_encodings)
        elif self.params.cfg_nodes_order == SingleFlatCFGNodesSeqMacroEncoderParams.CFGNodesOrder.Random:
            nr_cfg_nodes = cfg_nodes_encodings.size(0)
            # TODO: fix to use `BatchedFlattenedIndicesPseudoRandomPermutation.inverse_permute()`!
            cfg_nodes_permuted_indices = pdg_input.cfg_nodes_random_permutation.permutations
            # Note: The padded path suffix shouldn't be counted as node occurrences. Hence, we first map them
            #       to a new dummy non-exists node (id=nr_cfg_nodes), and then we remove this node.
            cfg_nodes_permuted_indices = cfg_nodes_permuted_indices.masked_fill(
                ~pdg_input.cfg_nodes_control_kind.unflattener_mask, nr_cfg_nodes)
            new_cfg_nodes_encodings = \
                cfg_nodes_encodings.new_zeros(size=(nr_cfg_nodes + 1, cfg_nodes_encodings.size(1)))
            path_encodings_flattened = path_encodings.flatten(0, 1)
            cfg_nodes_permuted_indices_flattened = cfg_nodes_permuted_indices.flatten(0, 1)
            new_cfg_nodes_encodings.scatter_(
                dim=0,
                index=cfg_nodes_permuted_indices_flattened.unsqueeze(-1).expand(path_encodings_flattened.shape),
                src=path_encodings_flattened)
            reflattened_nodes_encodings = new_cfg_nodes_encodings[:-1]
        else:
            assert False

        if self.norm:
            reflattened_nodes_encodings = self.norm(reflattened_nodes_encodings)

        assert reflattened_nodes_encodings.shape == cfg_nodes_encodings.shape
        return reflattened_nodes_encodings
