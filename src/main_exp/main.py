import argparse
import os, random
import numpy as np
import torch
import torch.nn as nn

from logger import Logger, print_encoder, print_method, printable_encoder
from dataset import *
from eval import evaluate_full, eval_acc
from parse import parse_method, parser_add_main_args
from trainer import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Parse args ###
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

if args.dataset in ("cora", "citeseer", "pubmed"):
    dataset = load_planetoid(args.data_dir, args.dataset, args.no_feat_norm)
elif args.dataset in ("cs", "physics"):
    dataset = load_coauthor(args.data_dir, args.dataset, args.no_feat_norm)
elif args.dataset == "stl10" or args.dataset == "cifar10":
    dataset = load_image(
        args.data_dir,
        args.dataset,
        train_per_class=args.label_num_per_class,
        valid_num=args.valid_num,
    )
elif args.dataset == "20news":
    dataset = load_20news(
        args.data_dir,
        train_per_class=args.label_num_per_class,
        valid_num=args.valid_num,
    )
else:
    raise ValueError("Invalid dataname")

if len(dataset.y.shape) == 1:
    dataset.y = dataset.y.unsqueeze(1)

c = max(dataset.y.max().item() + 1, dataset.y.shape[1])
d = dataset.x.shape[1]
n = dataset.num_nodes

print(
    f"dataset {args.dataset}: all nodes {dataset.num_nodes} | edges {dataset.edge_index.size(1)} | "
    + f"classes {c} | feats {d}"
)


criterion = nn.CrossEntropyLoss(
    reduction="none" if args.method in ["irm", "groupdro"] else "mean"
)

dataset.x, dataset.y, dataset.edge_index = (
    dataset.x.to(device),
    dataset.y.to(device),
    dataset.edge_index.to(device),
)

if args.method in trainer_list:
    model = parse_method(args, c, d, device)
    trainer = get_trainer_class(args.method)(model, dataset, criterion, args)
else:
    raise ValueError(f"Invalid method:{args.method}.")

eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()

for run in range(args.runs):
    model.reset_parameters()

    for epoch in range(args.epochs):
        model.train()
        loss = trainer.update()

        result = evaluate_full(model, dataset, eval_func)
        logger.add_result(run, result)

        if epoch % args.display_step == 0:
            m = f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * result[0]:.2f}%, Valid: {100 * result[1]:.2f}%, Test: {100 * result[2]:.2f}% "
            print(m)

    logger.print_statistics(run)

logger.print_statistics()


print_str = ""
if args.dataset in ("stl10", "cifar10", "20news"):
    print_str += (
        f"label_num_per_class:{args.label_num_per_class}, valid_num:{args.valid_num}, "
    )
print_str += (
    f"dataset:{args.dataset}, method:{args.method}, encoder:{args.encoder}, "
    f"hidden:{args.hidden_channels}, num_layers:{args.num_layers}, decay:{args.weight_decay}, "
    f"dropout:{args.dropout}, lr:{args.lr}, use_bn:{args.use_bn}\n"
)
if args.encoder in printable_encoder:
    print_str += print_encoder(args)
print_str += ", no_feat_norm, " if args.no_feat_norm else ", "
print_str += print_method(args)


s = logger.output(args)
print_str += s


folder = f"{args.output_folder}/{args.dataset}"

if not os.path.exists(folder):
    os.makedirs(folder)

file = f"{args.encoder}.txt"
path = os.path.join(folder, file)
with open(path, "a") as f:
    f.write(print_str)
    f.write("\n\n")
