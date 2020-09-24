import torch
import torch.nn as nn

from ndfa.nn_utils.misc import get_activation_layer
from ndfa.code_nn_modules.vocabulary import Vocabulary
from ndfa.code_nn_modules.code_task_input import PDGInputTensors


class CFGNodeEncoder(nn.Module):
    def __init__(self, cfg_node_dim: int, cfg_combined_expression_dim: int, pdg_node_control_kinds_vocab: Vocabulary,
                 pdg_node_control_kinds_embedding_dim: int = 8, nr_cfg_nodes_encoding_linear_layers: int = 2,
                 dropout_rate: float = 0.3, activation_fn: str = 'relu'):
        assert nr_cfg_nodes_encoding_linear_layers >= 1
        super(CFGNodeEncoder, self).__init__()
        self.activation_layer = get_activation_layer(activation_fn)()
        self.pdg_node_control_kinds_vocab_size = len(pdg_node_control_kinds_vocab)
        self.pdg_node_control_kinds_embedding_dim = pdg_node_control_kinds_embedding_dim
        self.pdg_node_control_kinds_embeddings = nn.Embedding(
            num_embeddings=self.pdg_node_control_kinds_vocab_size,
            embedding_dim=self.pdg_node_control_kinds_embedding_dim,
            padding_idx=pdg_node_control_kinds_vocab.get_word_idx('<PAD>'))
        self.cfg_node_dim = cfg_node_dim
        self.cfg_combined_expression_dim = cfg_combined_expression_dim

        self.cfg_node_projection_linear_layer = nn.Linear(
            in_features=self.cfg_combined_expression_dim + self.pdg_node_control_kinds_embedding_dim,
            out_features=self.cfg_node_dim)
        self.cfg_node_additional_linear_layers = nn.ModuleList(
            [nn.Linear(self.cfg_node_dim, self.cfg_node_dim) for _ in range(nr_cfg_nodes_encoding_linear_layers - 1)])

        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, combined_cfg_expressions_encodings: torch.Tensor, pdg: PDGInputTensors):
        assert combined_cfg_expressions_encodings.size(-1) == self.cfg_combined_expression_dim
        assert pdg.cfg_nodes_control_kind.tensor.ndim == 1  # (nr_cfg_nodes_in_batch,)
        nr_cfg_nodes_in_batch = pdg.cfg_nodes_control_kind.tensor.size(0)
        # nr_cfg_expressions_in_batch = pdg.cfg_nodes_tokenized_expressions.token_type.sequences.size(0)
        embedded_cfg_nodes_control_kind = self.pdg_node_control_kinds_embeddings(pdg.cfg_nodes_control_kind.tensor)
        cfg_nodes_expressions_encodings = combined_cfg_expressions_encodings.new_zeros(
            size=(nr_cfg_nodes_in_batch, self.cfg_combined_expression_dim))
        cfg_nodes_expressions_encodings.masked_scatter_(
            mask=pdg.cfg_nodes_has_expression_mask.tensor.unsqueeze(-1).expand(cfg_nodes_expressions_encodings.size()),
            source=combined_cfg_expressions_encodings)
        cfg_nodes_encodings = torch.cat(
            [cfg_nodes_expressions_encodings, embedded_cfg_nodes_control_kind], dim=-1)  # (nr_cfg_nodes_in_batch, expr_embed_dim + control_kind_embedding)

        cfg_nodes_encodings = self.dropout_layer(cfg_nodes_encodings)
        final_cfg_nodes_encodings_projected = self.dropout_layer(self.activation_layer(
            self.cfg_node_projection_linear_layer(cfg_nodes_encodings)))
        for linear_layer in self.cfg_node_additional_linear_layers:
            final_cfg_nodes_encodings_projected = self.dropout_layer(self.activation_layer(linear_layer(
                final_cfg_nodes_encodings_projected)))
        assert final_cfg_nodes_encodings_projected.size() == (nr_cfg_nodes_in_batch, self.cfg_node_dim)
        return final_cfg_nodes_encodings_projected

        # TODO: remove!
        # nr_examples = pdg.cfg_nodes_control_kind.nr_examples
        # max_nr_cfg_nodes = pdg.cfg_nodes_control_kind.max_nr_items
        # unflattened_nodes_encodings = pdg.cfg_nodes_control_kind.unflatten(final_cfg_nodes_encodings_projected)
        # packed_input = pack_padded_sequence(
        #     unflattened_nodes_encodings,
        #     lengths=pdg.cfg_nodes_control_kind.nr_items_per_example,
        #     enforce_sorted=False, batch_first=True)
        # rnn_outputs, (_, _) = self.rnn_layer(packed_input)
        # rnn_outputs, _ = pad_packed_sequence(sequence=rnn_outputs)
        # assert rnn_outputs.size() == (max_nr_cfg_nodes, nr_examples, self.nr_rnn_directions * self.output_dim)
        # if self.nr_rnn_directions > 1:
        #     rnn_outputs = rnn_outputs \
        #         .view(max_nr_cfg_nodes, nr_examples, self.nr_rnn_directions, self.output_dim).sum(dim=-2)
        # rnn_outputs = rnn_outputs.permute(1, 0, 2)  # (batch_size, max_nr_cfg_nodes, output_dim)
        # assert rnn_outputs.size() == (nr_examples, max_nr_cfg_nodes, self.output_dim)
        #
        # return rnn_outputs
