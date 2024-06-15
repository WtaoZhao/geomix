import torch


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true
    y_pred = y_pred.argmax(dim=-1, keepdim=True)

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(torch.sum(correct) / correct.shape[0])

    return sum(acc_list) / len(acc_list)


@torch.no_grad()
def evaluate_full(model, dataset, eval_func):
    model.eval()

    train_idx, valid_idx, test_idx = (
        dataset.train_idx,
        dataset.valid_idx,
        dataset.test_idx,
    )
    y = dataset.y

    out = model(dataset.x, dataset.edge_index)

    train_acc = eval_func(y[train_idx], out[train_idx])
    valid_acc = eval_func(y[valid_idx], out[valid_idx])
    test_acc = eval_func(y[test_idx], out[test_idx])
    result = [train_acc, valid_acc, test_acc]

    return result
