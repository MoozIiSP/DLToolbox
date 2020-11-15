import torch
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix


# TODO compute MAE and FLOPs
def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int):
    assert y_pred.size(1) != 1, "y_pred need be applied by argmax."
    bs = y_pred.size(0)

    acc = []
    for i in range(bs):
        acc.append(accuracy_score(y_pred.view(-1), y_true.view(-1)))
    acc = np.array(acc)
    return np.mean(acc)


if __name__ == '__main__':
    bs = 8
    num_classes = 10

    res = []
    for i in range(100000):
        y_pred = torch.randint(0, num_classes, size=(bs, num_classes))
        y_true = torch.randint(0, num_classes, size=(bs, num_classes))

        res.append(accuracy(y_pred, y_true, num_classes))
    print(sum(res)/len(res))
    print(1/num_classes)