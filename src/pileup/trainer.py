import torch

import torch.nn as nn
import torch.nn.functional as F

from utils import soft_ce

TRAINERS = [
    "augTrain",
]


def get_trainer_class(name):
    if name not in globals():
        raise ValueError(f"Trainer not found: {name}.")
    return globals()[name]


class augTrain:
    def __init__(self, model, criterion, args):

        self.args = args
        self.criterion = criterion
        self.model = model
        self.opt = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def init_opt(self, args):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    def update(self, data):
        self.model.train()

        # take gradient w.r.t pseudolabels
        raw_pred = self.model(data.x, data.edge_index)
        pred = F.softmax(raw_pred, dim=-1)
        if self.args.label_grad:
            input_y = pred.clone()
        else:
            with torch.no_grad():
                input_y = F.softmax(self.model(data.x, data.edge_index), dim=-1)
        n_class = data.y.max().item() + 1
        onehot_true = F.one_hot(data.y[data.labeled_index], n_class)
        input_y[data.labeled_index] = onehot_true.float()

        aug_x, aug_y = self.model.augment(data.x, input_y, data.edge_index)

        logits = self.model(aug_x, data.edge_index)
        sup_loss = self.criterion(
            logits[data.labeled_index], data.y[data.labeled_index]
        )

        aug_loss = soft_ce(logits[data.neutral_index], aug_y[data.neutral_index])

        loss = sup_loss + self.args.aug_lamb * aug_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.detach().item()
