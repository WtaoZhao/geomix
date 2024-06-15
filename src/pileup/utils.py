import torch
import torch.nn.functional as F
import numpy as np


def soft_ce(pred, true, apply_softmax=True):
    """
    soft cross entropy loss
    pred: (n, c)
    true: (n, c)
    """
    if apply_softmax:
        pred = torch.log_softmax(pred, dim=-1)
    loss = -torch.sum(pred * true, dim=1).mean()
    return loss


def count_right(pred, true):
    """
    pred: (n,C)
    true: (n,)
    """
    pred = pred.max(dim=1)[1]

    count = torch.sum((pred == true).int())
    total = pred.shape[0]

    return count, total


@torch.no_grad()
def evaluate_batch(model, loader, device):
    count, total = 0, 0
    for graph in loader:
        graph = graph.to(device)
        out = model(graph.x, graph.edge_index)
        cur_count, cur_total = count_right(
            out[graph.neutral_index], graph.y[graph.neutral_index]
        )
        count += cur_count
        total += cur_total
    acc = count / total

    return acc
