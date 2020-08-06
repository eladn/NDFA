import dataclasses
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional

from ndfa.code_nn_modules.vocabulary import Vocabulary
from ndfa.code_nn_modules.expression_encoder import ExpressionEncoder, EncodedExpression
from ndfa.nn_utils.scattered_encodings import ScatteredEncodings


@dataclasses.dataclass
class EncodedCFGNode:
    encoded_cfg_nodes: torch.Tensor
    encoded_cfg_nodes_expressions: EncodedExpression


class CFGNodeEncoder(nn.Module):
    def __init__(self, expression_encoder: ExpressionEncoder, pdg_node_control_kinds_vocab: Vocabulary,
                 pdg_node_control_kinds_embedding_dim: int = 8, rnn_type: str = 'lstm',
                 nr_rnn_layers: int = 2, rnn_bi_direction: bool = True):
        super(CFGNodeEncoder, self).__init__()
        self.expression_encoder = expression_encoder
        self.pdg_node_control_kinds_vocab_size = len(pdg_node_control_kinds_vocab)
        self.pdg_node_control_kinds_embedding_dim = pdg_node_control_kinds_embedding_dim
        self.pdg_node_control_kinds_embeddings = nn.Embedding(
            num_embeddings=self.pdg_node_control_kinds_vocab_size,
            embedding_dim=self.pdg_node_control_kinds_embedding_dim,
            padding_idx=pdg_node_control_kinds_vocab.get_word_idx('<PAD>'))
        self.output_dim = self.pdg_node_control_kinds_embedding_dim + expression_encoder.expr_encoding_dim

        self.nr_rnn_layers = nr_rnn_layers
        self.nr_rnn_directions = 2 if rnn_bi_direction else 1
        rnn_type = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn_layer = rnn_type(
            input_size=self.output_dim, hidden_size=self.output_dim,
            bidirectional=rnn_bi_direction, num_layers=self.nr_rnn_layers)

    def forward(self, encoded_identifiers: torch.Tensor, cfg_nodes_expressions: torch.Tensor,
                cfg_nodes_expressions_mask: torch.BoolTensor, cfg_nodes_control_kind: torch.Tensor,
                cfg_nodes_mask: torch.BoolTensor):
        encoded_expressions: EncodedExpression = self.expression_encoder(
            expressions=cfg_nodes_expressions, expressions_mask=cfg_nodes_expressions_mask,
            encoded_identifiers=encoded_identifiers)
        assert len(cfg_nodes_control_kind.size()) == 2  # (batch_size, nr_cfg_nodes)
        assert len(cfg_nodes_mask.size()) == 2  # (batch_size, nr_cfg_nodes)
        assert cfg_nodes_control_kind.size() == cfg_nodes_mask.size()
        embedded_cfg_nodes_control_kind = self.pdg_node_control_kinds_embeddings(cfg_nodes_control_kind.flatten())\
            .view(cfg_nodes_control_kind.size() + (-1,))  # (batch_size, nr_cfg_nodes, control_kind_embedding)
        assert len(encoded_expressions.expr_encoded_merge.size()) == len(embedded_cfg_nodes_control_kind.size()) == 3
        assert encoded_expressions.expr_encoded_merge.size()[:-1] == embedded_cfg_nodes_control_kind.size()[:-1]
        cfg_nodes_encodings = torch.cat([encoded_expressions.expr_encoded_merge, embedded_cfg_nodes_control_kind], dim=-1)  # (batch_size, nr_cfg_nodes, expr_embed_dim + control_kind_embedding)

        batch_size = cfg_nodes_mask.size()[0]
        max_nr_cfg_nodes = cfg_nodes_mask.size()[1]
        nr_cfg_nodes = None if cfg_nodes_mask is None else cfg_nodes_mask.long().sum(dim=1)
        nr_cfg_nodes = torch.where(nr_cfg_nodes <= torch.zeros(1, dtype=torch.long, device=nr_cfg_nodes.device),
                                   torch.ones(1, dtype=torch.long, device=nr_cfg_nodes.device), nr_cfg_nodes)
        packed_input = pack_padded_sequence(
            cfg_nodes_encodings.permute(1, 0, 2), lengths=nr_cfg_nodes, enforce_sorted=False)
        rnn_outputs, (_, _) = self.rnn_layer(packed_input)
        rnn_outputs, _ = pad_packed_sequence(sequence=rnn_outputs)
        max_nr_cfg_nodes = rnn_outputs.size()[0]
        assert rnn_outputs.size() == (max_nr_cfg_nodes, batch_size, self.nr_rnn_directions * self.output_dim)
        if self.nr_rnn_directions > 1:
            rnn_outputs = rnn_outputs \
                .view(max_nr_cfg_nodes, batch_size, self.nr_rnn_directions, self.output_dim).sum(dim=-2)
        rnn_outputs = rnn_outputs.permute(1, 0, 2)  # (batch_size, max_nr_cfg_nodes, output_dim)
        assert rnn_outputs.size() == (batch_size, max_nr_cfg_nodes, self.output_dim)
        return EncodedCFGNode(
            encoded_cfg_nodes=rnn_outputs,
            encoded_cfg_nodes_expressions=encoded_expressions)
