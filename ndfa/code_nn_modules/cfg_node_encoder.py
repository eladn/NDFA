import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ndfa.code_nn_modules.vocabulary import Vocabulary
from ndfa.code_nn_modules.expression_encoder import ExpressionEncoder, EncodedExpression
from ndfa.nn_utils.scattered_encodings import ScatteredEncodings
from ndfa.code_nn_modules.code_task_input import PDGInputTensors


@dataclasses.dataclass
class EncodedCFGNode:
    encoded_cfg_nodes: torch.Tensor
    encoded_cfg_nodes_expressions: EncodedExpression


class CFGNodeEncoder(nn.Module):
    def __init__(self, expression_encoder: ExpressionEncoder, pdg_node_control_kinds_vocab: Vocabulary,
                 pdg_node_control_kinds_embedding_dim: int = 8, rnn_type: str = 'lstm',
                 nr_rnn_layers: int = 2, rnn_bi_direction: bool = True, nr_cfg_nodes_encoding_linear_layers: int = 2):
        assert nr_cfg_nodes_encoding_linear_layers >= 1
        super(CFGNodeEncoder, self).__init__()
        self.expression_encoder = expression_encoder
        self.pdg_node_control_kinds_vocab_size = len(pdg_node_control_kinds_vocab)
        self.pdg_node_control_kinds_embedding_dim = pdg_node_control_kinds_embedding_dim
        self.pdg_node_control_kinds_embeddings = nn.Embedding(
            num_embeddings=self.pdg_node_control_kinds_vocab_size,
            embedding_dim=self.pdg_node_control_kinds_embedding_dim,
            padding_idx=pdg_node_control_kinds_vocab.get_word_idx('<PAD>'))
        self.output_dim = expression_encoder.expr_encoding_dim

        self.cfg_node_projection_linear_layer = nn.Linear(
            self.expression_encoder.expr_encoding_dim + self.pdg_node_control_kinds_embedding_dim, self.output_dim)
        self.cfg_node_additional_linear_layers = nn.ModuleList(
            [nn.Linear(self.output_dim, self.output_dim) for _ in range(nr_cfg_nodes_encoding_linear_layers - 1)])

        self.nr_rnn_layers = nr_rnn_layers
        self.nr_rnn_directions = 2 if rnn_bi_direction else 1
        rnn_type = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn_layer = rnn_type(
            input_size=self.output_dim, hidden_size=self.output_dim,
            bidirectional=rnn_bi_direction, num_layers=self.nr_rnn_layers)

    def forward(self, encoded_identifiers: torch.Tensor, pdg: PDGInputTensors):
        encoded_expressions: EncodedExpression = self.expression_encoder(
            expressions=pdg.cfg_nodes_tokenized_expressions, encoded_identifiers=encoded_identifiers)
        assert pdg.cfg_nodes_control_kind.tensor.ndim == 1  # (nr_cfg_nodes_in_batch,)
        nr_cfg_nodes_in_batch = pdg.cfg_nodes_control_kind.tensor.size(0)
        # nr_cfg_expressions_in_batch = pdg.cfg_nodes_tokenized_expressions.token_type.sequences.size(0)
        embedded_cfg_nodes_control_kind = self.pdg_node_control_kinds_embeddings(pdg.cfg_nodes_control_kind.tensor)
        cfg_nodes_expressions_encodings = torch.zeros(
            size=(nr_cfg_nodes_in_batch, encoded_expressions.expr_encoded_merge.size(-1)),
            dtype=encoded_expressions.expr_encoded_merge.dtype, device=encoded_expressions.expr_encoded_merge.device)
        cfg_nodes_expressions_encodings.masked_scatter_(
            mask=pdg.cfg_nodes_has_expression_mask, source=encoded_expressions.expr_encoded_merge)
        cfg_nodes_encodings = torch.cat(
            [cfg_nodes_expressions_encodings, embedded_cfg_nodes_control_kind], dim=-1)  # (nr_cfg_nodes_in_batch, expr_embed_dim + control_kind_embedding)

        cfg_nodes_encodings = self.dropout_layer(cfg_nodes_encodings)
        final_cfg_nodes_encodings_projected = self.dropout_layer(F.relu(
            self.cfg_node_projection_linear_layer(cfg_nodes_encodings)))
        for linear_layer in self.cfg_node_additional_linear_layers:
            final_cfg_nodes_encodings_projected = self.dropout_layer(F.relu(linear_layer(
                final_cfg_nodes_encodings_projected)))

        nr_examples = pdg.cfg_nodes_control_kind.nr_examples
        max_nr_cfg_nodes = pdg.cfg_nodes_control_kind.max_nr_items
        unflattened_nodes_encodings = pdg.cfg_nodes_control_kind.unflatten(final_cfg_nodes_encodings_projected)
        packed_input = pack_padded_sequence(
            unflattened_nodes_encodings,
            lengths=pdg.cfg_nodes_control_kind.nr_items_per_example,
            enforce_sorted=False, batch_first=True)
        rnn_outputs, (_, _) = self.rnn_layer(packed_input)
        rnn_outputs, _ = pad_packed_sequence(sequence=rnn_outputs)
        assert rnn_outputs.size() == (nr_examples, max_nr_cfg_nodes, self.nr_rnn_directions * self.output_dim)
        if self.nr_rnn_directions > 1:
            rnn_outputs = rnn_outputs \
                .view(nr_examples, max_nr_cfg_nodes, self.nr_rnn_directions, self.output_dim).sum(dim=-2)
        # rnn_outputs = rnn_outputs.permute(1, 0, 2)  # (batch_size, max_nr_cfg_nodes, output_dim)
        assert rnn_outputs.size() == (nr_examples, max_nr_cfg_nodes, self.output_dim)

        return EncodedCFGNode(
            encoded_cfg_nodes=rnn_outputs,
            encoded_cfg_nodes_expressions=encoded_expressions)
