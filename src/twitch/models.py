import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn.dense.linear import Linear


def get_edge_weight(edge_index, N):
    """
    edge_index should contain self loops
    """
    src, dst = edge_index

    deg = degree(dst, num_nodes=N)  # src and dst is the same for undirected

    deg_src = deg[src].pow(-0.5)  # 1/d^0.5(v_i)
    deg_src.masked_fill_(deg_src == float("inf"), 0)
    deg_dst = deg[dst].pow(-0.5)  # 1/d^0.5(v_j)
    deg_dst.masked_fill_(deg_dst == float("inf"), 0)
    edge_weight = deg_src * deg_dst

    return edge_weight


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = Linear(in_channels, out_channels, weight_initializer="glorot")

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, adj):
        x = self.lin(x)
        x = torch.sparse.mm(adj, x)

        return x


class GeoMix1(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        alpha=0,
        hops=1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNLayer(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNLayer(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.alpha = alpha
        self.hops = hops

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        N = x.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        edge_weight = get_edge_weight(edge_index, N)

        adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N]))
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x

    def augment(self, x, y, edge_index):
        N = x.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        edge_weight = get_edge_weight(edge_index, N)

        adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N]))

        for _ in range(self.hops):
            x_prime = torch.sparse.mm(adj, x)
            y_prime = torch.sparse.mm(adj, y.float())

            if self.alpha > 0:
                x_prime = (1 - self.alpha) * x_prime + self.alpha * x
                y_prime = (1 - self.alpha) * y_prime + self.alpha * y

            x = x_prime
            y = y_prime

        return x, y


class GeoMix2(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        hops=2,
        alpha=0.8,
        dropout=0.5,
        use_bn=True,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNLayer(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNLayer(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.hops = hops
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        N = x.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        edge_weight = get_edge_weight(edge_index, N)

        adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N]))

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x

    def augment(self, x, y, edge_index):
        N = x.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        edge_weight = get_edge_weight(edge_index, N)

        adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N]))

        x0, y0 = x, y
        for k in range(self.hops):
            x = (1 - self.alpha) * torch.sparse.mm(adj, x) + self.alpha * x0
            y = (1 - self.alpha) * torch.sparse.mm(adj, y) + self.alpha * y0

        return x, y


class GeoMix3(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        res_weight=0.5,
        hops=1,
        graph_weight=0.8,
        use_weight=False,
        attn_emb_dim=64,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNLayer(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNLayer(hidden_channels, out_channels))

        if use_weight:
            self.WQs = nn.ModuleList()
            self.WKs = nn.ModuleList()
            for _ in range(hops):
                self.WQs.append(nn.Linear(in_channels, attn_emb_dim))
                self.WKs.append(nn.Linear(in_channels, attn_emb_dim))

        self.dropout = dropout
        self.use_bn = use_bn
        self.activation = F.relu

        self.res_weight = res_weight
        self.graph_weight = graph_weight
        self.hops = hops
        self.use_weight = use_weight
        self.adj = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.use_weight:
            for lin in self.WQs:
                lin.reset_parameters()
            for lin in self.WKs:
                lin.reset_parameters()

    def get_edge_weight(self, edge_index, N):
        """
        edge_index should contain self loops
        """
        src, dst = edge_index

        # src and dst is the same for undirected
        deg = degree(dst, num_nodes=N)

        deg_src = deg[src].pow(-0.5)  # 1/d^0.5(v_i)
        deg_src.masked_fill_(deg_src == float("inf"), 0)
        deg_dst = deg[dst].pow(-0.5)  # 1/d^0.5(v_j)
        deg_dst.masked_fill_(deg_dst == float("inf"), 0)
        edge_weight = deg_src * deg_dst

        return edge_weight

    def forward(self, x, edge_index):
        if self.adj is None:
            N = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

            edge_weight = self.get_edge_weight(edge_index, N)

            self.adj = torch.sparse_coo_tensor(
                edge_index, edge_weight, torch.Size([N, N])
            )

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, self.adj)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, self.adj)
        return x

    def attention_conv(self, qs, ks, vs, ys):
        """
        qs, ks: (N, attn_emb_dim)
        vs: (N, in_channels)
        ys: (N, C), labels to mix
        """
        qs = qs / torch.norm(qs, p=2)
        ks = ks / torch.norm(ks, p=2)
        N = qs.shape[0]

        kvs = torch.matmul(ks.T, vs)  # (attn_emb_dim, in_channels)
        qkv = torch.matmul(qs, kvs)  # (N, in_channels)
        num_x = qkv + N * vs  # (N, in_channels)

        kys = torch.matmul(ks.T, ys)
        qky = torch.matmul(qs, kys)
        num_y = qky + N * ys  # (N, C)

        ks_sum = torch.sum(ks.T, dim=-1, keepdim=True)  # (attn_emb_dim, 1)
        denominator = torch.matmul(qs, ks_sum)  # (N, 1)
        denominator += torch.ones_like(denominator) * N

        output_x = num_x / denominator
        output_y = num_y / denominator

        return output_x, output_y

    def gcn_conv(self, x, y, adj):
        for _ in range(self.hops):
            x = torch.sparse.mm(adj, x)
            y = torch.sparse.mm(adj, y)

        return x, y

    def augment(self, x, y, edge_index):
        if self.adj is None:
            N = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

            edge_weight = self.get_edge_weight(edge_index, N)

            self.adj = torch.sparse_coo_tensor(
                edge_index, edge_weight, torch.Size([N, N])
            )

        attn_x = x
        attn_y = y
        for i in range(self.hops):
            if self.use_weight:
                qs = self.WQs[i](attn_x)
                ks = self.WKs[i](attn_x)
            else:
                qs, ks = attn_x, attn_x
            new_attn_x, new_attn_y = self.attention_conv(qs, ks, attn_x, attn_y)

            if self.graph_weight > 0:
                gnn_x, gnn_y = self.gcn_conv(attn_x, attn_y, self.adj)
                new_attn_x = (
                    self.graph_weight * gnn_x + (1 - self.graph_weight) * new_attn_x
                )
                new_attn_y = (
                    self.graph_weight * gnn_y + (1 - self.graph_weight) * new_attn_y
                )

            if self.res_weight > 0:
                attn_x = self.res_weight * attn_x + (1 - self.res_weight) * new_attn_x
                attn_y = self.res_weight * attn_y + (1 - self.res_weight) * new_attn_y
            else:
                attn_x, attn_y = new_attn_x, new_attn_y

        return attn_x, attn_y
