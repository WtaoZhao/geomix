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
    def __init__(self, model, data, criterion, args):
        self.x, self.y, self.edge_index = data.x, data.y, data.edge_index
        train_idx = data.train_idx
        if len(self.y.shape) == 2:
            self.y = self.y.squeeze(1)
        self.onehot_true = F.one_hot(self.y, self.y.max().item() + 1)

        node_idx = torch.arange(self.x.shape[0])
        combined = torch.cat((node_idx, train_idx))
        uniques, counts = combined.unique(return_counts=True)
        self.unlabeled_idx = uniques[counts == 1]

        self.train_idx = train_idx

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

        aug_x, aug_y = self.model.augment(self.x, input_y, self.edge_index)

        logits = self.model(aug_x, self.edge_index)
        sup_loss = self.criterion(logits[self.train_idx], self.y[self.train_idx])

        aug_loss = soft_ce(logits[self.unlabeled_idx], aug_y[self.unlabeled_idx])
        loss = sup_loss + self.args.aug_lamb * aug_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.detach().item()
