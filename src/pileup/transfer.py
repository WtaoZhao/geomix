"""
Pileup dataset. Transfer between different PU levels.
"""

import argparse
import os, random
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from logger import Logger, print_encoder, print_method, printable_encoder
from prep_data import load_pileup
from parse import parse_method, parser_add_main_args
from utils import evaluate_batch
from trainer import *


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description="General Training Pipeline")
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

trainer_list = ["augTrain"]

if args.cpu:
    device = torch.device("cpu")
else:
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

if args.dataset == "pileup":
    src_dir = os.path.join(
        args.data_dir, "pileup", f"test_{args.src_sig}_PU{args.src_pu}.root"
    )
    train_graphs = load_pileup(args.src_event, args, src_dir)
    tgt_dir = os.path.join(
        args.data_dir, "pileup", f"test_{args.tgt_sig}_PU{args.tgt_pu}.root"
    )
    tgt_graphs = load_pileup(args.tgt_event, args, tgt_dir)
    n_graph = len(tgt_graphs)
    n_val = int(n_graph * args.val_ratio)
    val_graphs = tgt_graphs[:n_val]
    test_graphs = tgt_graphs[n_val:]

else:
    raise ValueError("Non supported dataset: {args.dataset}")

train_loader = DataLoader(train_graphs, batch_size=args.batch_size)
val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

sample_graph = train_graphs[0]
d = sample_graph.x.shape[1]
c = sample_graph.y.max().item() + 1


criterion = nn.CrossEntropyLoss(
    reduction="none" if args.method in ["irm", "groupdro"] else "mean"
)
if args.method in trainer_list:
    model = parse_method(args, c, d, device)
    trainer = get_trainer_class(args.method)(model, criterion, args)
else:
    raise ValueError

logger = Logger(args.runs, args)
for run in range(args.runs):
    model.reset_parameters()
    if run > 0:
        trainer.init_opt(args)
    for epoch in range(args.epochs):
        model.train()
        for graph in train_loader:
            graph = graph.to(device)
            loss = trainer.update(graph)

        # eval
        model.eval()
        train_acc = evaluate_batch(model, train_loader, device)
        val_acc = evaluate_batch(model, val_loader, device)
        test_acc = evaluate_batch(model, test_loader, device)

        result = [train_acc, val_acc, test_acc]
        logger.add_result(run, result)

        if epoch == 0 or (epoch + 1) % args.display_step == 0:
            m = f"Epoch: {epoch+1:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}% "
            print(m)

    logger.print_statistics(run)

logger.print_statistics()

print_str = (
    f"src:PU{args.src_pu}, tgt:{args.tgt_pu}, src_event:{args.src_event}, tgt_event:{args.tgt_event}, labeled_ratio:{args.labeled_ratio}, train_ratio:{args.train_ratio}\n method:{args.method}, encoder:{args.encoder}, "
    f"hidden:{args.hidden_channels}, num_layers:{args.num_layers}, decay:{args.weight_decay}, "
    f"dropout:{args.dropout}, lr:{args.lr}, use_bn:{args.use_bn}"
)
if args.encoder in printable_encoder:
    print_str += ", " + print_encoder(args)

print_str += print_method(args)

s = logger.output()
print_str += s

parent_folder = args.output_folder
if args.src_sig == args.tgt_sig and args.src_sig == "gg":
    folder = f"{parent_folder}/PU{args.src_pu}_trans_{args.tgt_pu}"
else:
    folder = f"{parent_folder}/PU{args.src_pu}{args.src_sig}_trans_{args.tgt_pu}{args.tgt_sig}"
if not os.path.exists(folder):
    os.makedirs(folder)
path = os.path.join(folder, f"{args.encoder}.txt")

with open(path, "a") as f:
    f.write(print_str)
    f.write("\n\n")
