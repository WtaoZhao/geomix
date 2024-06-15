import torch
import torch.nn.functional as F


def class_rand_splits(label, train_per_class=20, valid_per_class=30):
    train_idx, val_idx, test_idx = [], [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:train_per_class]
        val_idx += rand_idx[train_per_class : train_per_class + valid_per_class]
        test_idx += rand_idx[train_per_class + valid_per_class :]
    train_idx = torch.as_tensor(train_idx)
    val_idx = torch.as_tensor(val_idx)
    test_idx = torch.as_tensor(test_idx)

    return train_idx, val_idx, test_idx


def class_rand_splits_2(label, label_num_per_class, valid_num):
    """use all remaining data points as test data"""
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        # print(f'class {c_i}, num {n_i}')
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = non_train_idx[:valid_num], non_train_idx[valid_num:]
    # print(f"train:{train_idx.shape}, valid:{valid_idx.shape}, test:{test_idx.shape}")

    return train_idx, valid_idx, test_idx
