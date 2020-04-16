import torch
import torch.nn as nn

from ddfa.code_nn_modules.vocabulary import Vocabulary


class CFGNodeEncoder(nn.Module):
    def __init__(self, expression_encoder: nn.Module, pdg_node_control_kinds_vocab: Vocabulary,
                 pdg_node_control_kinds_embedding_dim: int = 8):
        super(CFGNodeEncoder, self).__init__()
        self.expression_encoder = expression_encoder
        self.pdg_node_control_kinds_vocab_size = len(pdg_node_control_kinds_vocab)
        self.pdg_node_control_kinds_embeddings = nn.Embedding(
            num_embeddings=self.pdg_node_control_kinds_vocab_size,
            embedding_dim=pdg_node_control_kinds_embedding_dim)

    def forward(self, encoded_identifiers: torch.Tensor, cfg_nodes_expressions: torch.Tensor,
                cfg_nodes_control_kind: torch.Tensor):
        encoded_expressions = self.expression_encoder(
            expressions=cfg_nodes_expressions, encoded_identifiers=encoded_identifiers)
        embedded_cfg_nodes_control_kind = self.pdg_node_control_kinds_embeddings(cfg_nodes_control_kind)
        return torch.cat([encoded_expressions, embedded_cfg_nodes_control_kind], dim=-1)
