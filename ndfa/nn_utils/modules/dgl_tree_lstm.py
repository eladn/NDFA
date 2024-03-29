import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from typing import Optional


__all__ = ['TreeLSTM']


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(self, node_embedding_size: int, hidden_size: Optional[int] = None,
                 cell_type='nary', dropout_rate: float = 0.3):
        super(TreeLSTM, self).__init__()
        self.node_embedding_size = node_embedding_size
        self.hidden_size = self.node_embedding_size if hidden_size is None else hidden_size
        cell = TreeLSTMCell if cell_type == 'nary' else ChildSumTreeLSTMCell
        self.cell = cell(self.node_embedding_size, self.hidden_size)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, nodes_embeddings: torch.Tensor,
                tree: dgl.DGLGraph,
                h: torch.Tensor, c: torch.Tensor,
                direction: str = 'leaves_to_root'):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        nodes_embeddings : torch.Tensor
            Initial nodes embeddings.
        tree : dgl.DGLGraph
            Tree for computation.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        direction : str
            Message passing direction over the tree.
            Either `root_to_leaves` or `leaves_to_root`.
        """
        assert direction in {'root_to_leaves', 'leaves_to_root'}
        if direction == 'root_to_leaves':
            u, v = tree.edges('uv')
            tree = dgl.graph(data=(v, u), num_nodes=tree.num_nodes())
        # feed embedding
        tree.ndata['iou'] = self.cell.W_iou(nodes_embeddings)  # * batch.mask.float().unsqueeze(-1)
        tree.ndata['h'] = h
        tree.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(
            graph=tree, message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            # reverse=(direction == 'root_to_leaves'),
            apply_node_func=self.cell.apply_node_func)
        h = self.dropout_layer(tree.ndata.pop('h'))
        return h
