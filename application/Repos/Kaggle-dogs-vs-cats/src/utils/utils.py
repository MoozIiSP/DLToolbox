import importlib
import os
import random
import shutil
from itertools import product
from typing import Any, Dict, Generator, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )
    return getattr(module_obj, obj_name)


def set_seed(seed: int = 666):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_useful_info():
    shutil.copytree(os.path.join(hydra.utils.get_original_cwd(), 'src'),
                    os.path.join(os.getcwd(), 'code/src'))
    shutil.copy2(os.path.join(hydra.utils.get_original_cwd(), 'hydra_run.py'), os.path.join(os.getcwd(), 'code'))


def collate_fn(batch):
    # FIXME: Performance Error?
    inputs, targets, idxs = tuple(zip(*batch))
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    return inputs, targets, idxs


# Refer to https://github.com/PyTorchLightning/pytorch-lightning/pull/1564/files
BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
def _make_trainable(module: torch.nn.Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: torch.nn.Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module: torch.nn.Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


# def filter_params(module: torch.nn.Module,
#                   train_bn: bool = True) -> Generator:
#     """Yields the trainable parameters of a given module.
#     Args:
#         module: A given module
#         train_bn: If True, leave the BatchNorm layers in training mode
#     Returns:
#         Generator
#     """
#     children = list(module.children())
#     if not children:
#         if not (isinstance(module, BN_TYPES) and train_bn):
#             for param in module.parameters():
#                 if param.requires_grad:
#                     yield param
#     else:
#         for child in children:
#             for param in filter_params(module=child, train_bn=train_bn):
#                 yield param


# def _unfreeze_and_add_param_group(module: torch.nn.Module,
#                                   optimizer: Optimizer,
#                                   lr: Optional[float] = None,
#                                   train_bn: bool = True):
#     """Unfreezes a module and adds its parameters to an optimizer."""
#     _make_trainable(module)
#     params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
#     optimizer.add_param_group(
#         {'params': filter_params(module=module, train_bn=train_bn),
#          'lr': params_lr / 10.,
#          })

