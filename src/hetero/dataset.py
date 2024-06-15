import numpy as np
import os
import torch
import torch.nn.functional as F

from data_utils import normalize_feat

from torch_geometric.data import Data

DATAPATH = "../../data/"


def load_dataset(args):
    """Loader for NCDataset
    Returns NCDataset
    """
    dataname = args.dataset
    print(dataname)
    if dataname in ("chameleon", "squirrel"):
        dataset = load_wiki_new(dataname, args.no_feat_norm)
    else:
        raise ValueError("Invalid dataname")
    return dataset


def load_wiki_new(name, no_feat_norm=False):
    path = os.path.join(DATAPATH, f"wiki_new/{name}/{name}_filtered.npz")
    data = np.load(path)
    node_feat = data["node_features"]  # unnormalized
    labels = data["node_labels"]
    edges = data["edges"]  # (E, 2)
    edge_index = edges.T

    if not no_feat_norm:
        node_feat = normalize_feat(node_feat)

    edge_index = torch.as_tensor(edge_index)
    node_feat = torch.as_tensor(node_feat)
    labels = torch.as_tensor(labels)

    dataset = Data(node_feat, edge_index, y=labels)

    return dataset
