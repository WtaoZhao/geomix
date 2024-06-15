import argparse
import os
import random
import numpy as np
import torch

import torch.nn as nn

from logger import Logger
from dataset import load_dataset
from torch_geometric.utils import to_undirected
from data_utils import evaluate, eval_acc, \
    load_fixed_splits, class_rand_splits
from parse import parse_method, parser_add_main_args
from trainer import get_trainer_class

import warnings
warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")

dataset = load_dataset(args)

if len(dataset.y.shape) == 1:
    dataset.y = dataset.y.unsqueeze(1)
dataset.y = dataset.y.to(device)

dataset_name = args.dataset
if args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.y, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset)

n = dataset.x.shape[0]
c = max(dataset.y.max().item() + 1, dataset.y.shape[1])
d = dataset.x.shape[1]

dataset.edge_index = to_undirected(dataset.edge_index)
dataset.edge_index, dataset.x = \
    dataset.edge_index.to(
        device), dataset.x.to(device)

print(f"num nodes {n} | num classes {c} | num node feats {d}")


criterion = nn.CrossEntropyLoss()
eval_func = eval_acc

logger = Logger(args.runs, args)

model = parse_method(args, c, d, device)
trainer=get_trainer_class(args.method)(model,dataset,criterion,args)

model.train()

patience = 0

for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train']
    
    model.reset_parameters()
    trainer.update_idx(train_idx)

    best_val = float('-inf')
    patience = 0
    for epoch in range(args.epochs):
        loss=trainer.update()

        result = evaluate(model, dataset, split_idx,
                          eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()
print(results)


def print_method(args):
    print_str=''
    if args.method in ['augTrain']:
        print_str+=f'label_grad:{args.label_grad}, aug_lamb:{args.aug_lamb}'
        print_str+='\n'
    
    return print_str

def print_augmod(args):
    print_str=''
    if args.method=='augTrain':
        if args.aug_mod in ['geomix_1','geomix_2']:
            print_str+=f'hops:{args.hops}, alpha:{args.alpha}'
        elif args.aug_mod=='geomix_3':
            print_str+=f'res_weight:{args.res_weight}, hops:{args.hops}, graph_wegiht:{args.graph_weight}, use_weight:{args.use_weight}, attn_emb_dim:{args.attn_emb_dim}'

    return print_str

print_str=''
print_str+=f'dataset:{args.dataset}, method:{args.method}, encoder:{args.encoder}, '\
                f'hidden:{args.hidden_channels}, num_layers:{args.num_layers}, decay:{args.weight_decay}, '\
                f'dropout:{args.dropout}, lr:{args.lr}, use_bn:{args.use_bn}\n'

aug_mod_str=print_augmod(args)
print_str+=aug_mod_str
print_str+=', ' if len(aug_mod_str)>0 else ''
print_str+=print_method(args)
print_str+=results

folder=f'results/{args.dataset}'
if not os.path.exists(folder):
    os.makedirs(folder)

file=f'{args.aug_mod}.txt'
path=os.path.join(folder,file)
with open(path,'a') as f:
    f.write(print_str)
    f.write('\n\n')
