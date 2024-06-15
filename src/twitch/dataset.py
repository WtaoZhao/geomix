import torch
import torch_geometric.transforms as T

from torch_geometric.datasets import Twitch
from torch_geometric.data import Data


def load_twitch_dataset(
    data_dir, method, train_num=3, train_ratio=0.5, valid_ratio=0.25
):
    transform = T.NormalizeFeatures()
    sub_graphs = ["DE", "EN", "ES", "FR", "PT", "RU"]
    x_list, edge_index_list, y_list, env_list = [], [], [], []
    node_idx_list = []
    idx_shift = 0
    for i, g in enumerate(sub_graphs):
        torch_dataset = Twitch(root=f"{data_dir}Twitch", name=g, transform=transform)
        data = torch_dataset[0]
        x, edge_index, y = data.x, data.edge_index, data.y
        x_list.append(x)
        y_list.append(y)
        edge_index_list.append(edge_index + idx_shift)
        env_list.append(torch.ones(x.size(0)) * i)
        node_idx_list.append(torch.arange(data.num_nodes) + idx_shift)

        idx_shift += data.num_nodes

    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    env = torch.cat(env_list, dim=0)
    dataset = Data(x=x, edge_index=edge_index, y=y)
    dataset.env = env
    dataset.env_num = len(sub_graphs)

    assert train_num <= 5

    ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    idx = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx[: int(idx.size(0) * train_ratio)]
    valid_idx_ind = idx[
        int(idx.size(0) * train_ratio) : int(idx.size(0) * (train_ratio + valid_ratio))
    ]
    test_idx_ind = idx[int(idx.size(0) * (train_ratio + valid_ratio)) :]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]
    dataset.test_ood_idx = (
        [node_idx_list[-1]] if train_num >= 4 else node_idx_list[train_num:]
    )

    return dataset
