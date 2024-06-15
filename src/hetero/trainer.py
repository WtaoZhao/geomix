import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import remove_self_loops, add_self_loops, degree

from utils import soft_ce

TRAINERS = [
    "augTrain",
]


def get_trainer_class(name):
    if name not in globals():
        raise ValueError(f"Trainer not found: {name}.")
    return globals()[name]


def get_augmod(args, in_channels=None):
    name = args.aug_mod
    if name == "geomix_1":
        aug = GeoMix1(args.hops, args.alpha)
    elif name == "geomix_2":
        aug = GeoMix2(args.hops, args.alpha)
    elif name == "geomix_3":
        aug = GeoMix3(
            in_channels,
            args.hops,
            args.res_weight,
            args.graph_weight,
            args.use_weight,
            args.attn_emb_dim,
        )
    else:
        raise ValueError

    return aug


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


class GeoMix1(nn.Module):
    def __init__(self, hops, alpha=0):
        super().__init__()

        self.adj = None
        self.hops = hops
        self.alpha = alpha

    def augment(self, x, y, edge_index):
        if self.adj is None:
            N = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

            edge_weight = get_edge_weight(edge_index, N)
            self.adj = torch.sparse_coo_tensor(
                edge_index, edge_weight, torch.Size([N, N])
            )

        for _ in range(self.hops):
            x_prime = torch.sparse.mm(self.adj, x)
            y_prime = torch.sparse.mm(self.adj, y.float())

            if self.alpha > 0:
                x_prime = (1 - self.alpha) * x_prime + self.alpha * x
                y_prime = (1 - self.alpha) * y_prime + self.alpha * y

            x = x_prime
            y = y_prime

        return x, y


class GeoMix2(nn.Module):
    def __init__(self, hops, alpha):
        super().__init__()

        self.adj = None
        self.hops = hops
        self.alpha = alpha

    def augment(self, x, y, edge_index):
        if self.adj is None:
            N = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

            edge_weight = get_edge_weight(edge_index, N)
            self.adj = torch.sparse_coo_tensor(
                edge_index, edge_weight, torch.Size([N, N])
            )

        x0, y0 = x, y
        for k in range(self.hops):
            x = (1 - self.alpha) * torch.sparse.mm(self.adj, x) + self.alpha * x0
            y = (1 - self.alpha) * torch.sparse.mm(self.adj, y) + self.alpha * y0

        return x, y


class GeoMix3(nn.Module):
    def __init__(
        self, in_channels, hops, res_weight, graph_weight, use_weight, attn_emb_dim
    ):
        super().__init__()

        self.adj = None
        self.hops = hops
        self.res_weight = res_weight
        self.use_weight = use_weight
        self.graph_weight = graph_weight

        if use_weight:
            self.WQs = nn.ModuleList()
            self.WKs = nn.ModuleList()
            for _ in range(hops):
                self.WQs.append(nn.Linear(in_channels, attn_emb_dim))
                self.WKs.append(nn.Linear(in_channels, attn_emb_dim))

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

            edge_weight = get_edge_weight(edge_index, N)
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


class augTrain:
    def __init__(self, model, data, criterion, args):
        self.x, self.y, self.edge_index = data.x, data.y, data.edge_index
        if len(self.y.shape) == 2:
            self.y = self.y.squeeze(1)
        self.onehot_true = F.one_hot(self.y, self.y.max().item() + 1)

        self.args = args
        self.criterion = criterion
        self.model = model
        self.opt = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        self.aug_mod = get_augmod(args, data.x.shape[1]).to(data.x.device)

    def init_opt(self, args):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def update_idx(self, train_idx):
        train_idx = train_idx
        node_idx = torch.arange(self.x.shape[0])
        combined = torch.cat((node_idx, train_idx))
        uniques, counts = combined.unique(return_counts=True)
        self.unlabeled_idx = uniques[counts == 1]
        self.train_idx = train_idx

    def update(self):
        self.model.train()

        # take gradient w.r.t pseudolabels
        if self.args.label_grad:
            pred = F.softmax(self.model(self.x, self.edge_index), dim=-1)
            input_y = pred.clone()
        else:
            with torch.no_grad():
                input_y = F.softmax(self.model(self.x, self.edge_index), dim=-1)
        input_y[self.train_idx] = self.onehot_true[self.train_idx].float()

        aug_x, aug_y = self.aug_mod.augment(self.x, input_y, self.edge_index)

        logits = self.model(aug_x, self.edge_index)
        sup_loss = self.criterion(logits[self.train_idx], self.y[self.train_idx])

        aug_loss = soft_ce(logits[self.unlabeled_idx], aug_y[self.unlabeled_idx])
        loss = sup_loss + self.args.aug_lamb * aug_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.detach().item()
