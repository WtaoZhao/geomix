from collections import defaultdict
import numpy as np
import torch
import torch_geometric.transforms as T

from data_utils import class_rand_splits, class_rand_splits_2

from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from torch_geometric.nn import knn_graph
from sklearn.neighbors import kneighbors_graph


import pickle as pkl
import networkx as nx
import os


def load_planetoid(data_dir, name, no_feat_norm):
    transform = None if no_feat_norm else T.NormalizeFeatures()
    if name in ["cora", "citeseer", "pubmed"]:
        torch_dataset = Planetoid(
            root=f"{data_dir}Planetoid", name=name, transform=transform
        )
        data = torch_dataset[0]
    else:
        raise ValueError(f"Invalid dataset name: {name}.")

    node_idx = torch.arange(data.x.shape[0])
    data.train_idx = node_idx[data.train_mask]
    data.valid_idx = node_idx[data.val_mask]
    data.test_idx = node_idx[data.test_mask]

    return data


def load_image(data_dir, name, k=5, train_per_class=10, valid_num=4000):
    path = os.path.join(data_dir, name, "features.pkl")
    data = pkl.load(open(path, "rb"))
    x_train, y_train, x_test, y_test = (
        data["x_train"],
        data["y_train"],
        data["x_test"],
        data["y_test"],
    )
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test))

    if name == "cifar10":
        num_image = 15000
        x = x[:num_image]
        y = y[:num_image]

    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)

    edge_path = os.path.join(data_dir, name, "edge.pt")
    if not os.path.exists(edge_path):
        edge_index = knn_graph(x, k)
        torch.save(edge_index, edge_path)
    else:
        edge_index = torch.load(edge_path)
    edge_index = to_undirected(edge_index)

    train_idx, valid_idx, test_idx = class_rand_splits_2(y, train_per_class, valid_num)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
    )

    return data


def load_coauthor(data_dir, name, no_feat_norm):
    transform = None if no_feat_norm else T.NormalizeFeatures()
    if name.lower() in ["cs", "physics"]:
        torch_dataset = Coauthor(
            root=os.path.join(data_dir, "coauthor"), name=name, transform=transform
        )
        data = torch_dataset[0]
    else:
        raise ValueError(f"Invalid dataset name: {name}.")

    train_idx, valid_idx, test_idx = class_rand_splits(data.y)
    data.train_idx = train_idx
    data.valid_idx = valid_idx
    data.test_idx = test_idx

    return data


def load_20news(data_dir, k=5, train_per_class=100, valid_num=2000):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    categories = [
        "alt.atheism",
        "comp.sys.ibm.pc.hardware",
        "misc.forsale",
        "rec.autos",
        "rec.sport.hockey",
        "sci.crypt",
        "sci.electronics",
        "sci.med",
        "sci.space",
        "talk.politics.guns",
    ]
    data = fetch_20newsgroups(
        data_home=os.path.join(data_dir, "20news"), subset="all", categories=categories
    )
    vectorizer = CountVectorizer(stop_words="english", min_df=0.05)
    X_counts = vectorizer.fit_transform(data.data).toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    features = transformer.fit_transform(X_counts).todense()

    adj = kneighbors_graph(features, n_neighbors=k, include_self=True)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    edge_index = to_undirected(edge_index)

    features = torch.Tensor(features)
    y = data.target
    y = torch.LongTensor(y)
    train_idx, valid_idx, test_idx = class_rand_splits_2(y, train_per_class, valid_num)
    data = Data(
        x=features,
        edge_index=edge_index,
        y=y,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
    )

    return data
