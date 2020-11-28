from typing import Any
import numpy as np


class _Loss:
    def __init__(self) -> None:
        pass
    def forward(self, *args) -> np.ndarray:
        pass
    def __call__(self, *args) -> Any:
        return self.forward(*args)


class LogLoss(_Loss):
    def __init__(self, reduction: str = 'none', eps = 1e-7) -> None:
        self.eps = eps
        self.reduction = reduction
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2 and x.shape[1] == 1, \
            "x must be (nc, 1), which only has one class."
        assert len(y.shape) == 1, \
            "y must be (nc,)."
        y = y[..., np.newaxis]
        loss = np.sum(y * np.log(x) + (1 - y) * np.log(1 - x), axis=1)
        if self.reduction == 'sum':
            return np.sum(-loss)
        elif self.reduction == 'mean':
            return np.mean(-loss)
        return -loss
    def __call__(self, *args):
        return self.forward(*args)


class CELoss:
    """Same as LogLoss when number of classes is equal to 2."""
    def __init__(self, reduction: str = 'none', eps = 1e-7) -> None:
        self.eps = eps
        self.reduction = reduction
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2, \
            "x must be (nc, 1)."
        assert len(y.shape) == 1, \
            "y must be (nc,), it will be converted to one-hot encoding."
        y = np.eye(x.shape[1])[y]  # According to index, this will be generate one-hot encoding.
        loss = np.sum(y * np.log(x), axis=1)
        if self.reduction == 'sum':
            return np.sum(-loss)
        elif self.reduction == 'mean':
            return np.mean(-loss)
        return -loss
    def __call__(self, *args):
        return self.forward(*args)



if __name__ == "__main__":
    x = np.random.rand(8, 1)
    x_hat = np.concatenate([1 - x, x], axis=1)
    y = np.random.randint(0, 2, size=(8,), dtype=np.int32)

    loss = LogLoss(reduction='mean')
    print('LogLoss', loss(x, y))
    loss = CELoss(reduction='mean')
    print('CELoss', loss(x_hat, y))
    
    import torch
    import torch.nn as nn
    x_hat = torch.tensor(x_hat, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    y_oh = torch.nn.functional.one_hot(y).type(torch.float)
    print('nn.BCELoss', nn.BCELoss()(x_hat, (y_oh)))
    print('nn.CrossEntropyLoss', nn.CrossEntropyLoss()(x_hat, y))