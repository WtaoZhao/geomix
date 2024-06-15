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


