import ast
import os
import sys
from typing import Dict, Tuple

# sys.path.append('/home/mooziisp/GitRepos/wheat/')

import numpy as np
import torch
import albumentations as A
from albumentations.core.composition import Compose
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.model_selection import train_test_split
from src.utils.utils import load_obj
from torch.utils.data import Dataset


class KaggleDataset(Dataset):

    def __init__(self,
                 cfg: DictConfig = None,
                 data_dir: str = '',
                 data_ids: list = None,
                 classnames: Dict = None,
                 mode: str = 'train',
                 transforms: Compose = None):
        """
        Prepare data for Kaggle Competition.
        """
        self.cfg = cfg
        self.data_dir = data_dir
        self.data_ids = data_ids
        self.classnames = classnames
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, idx: int):
        data_ind = self.data_ids[idx].split('.')[0]

        image = np.array(Image.open(f'{self.data_dir}/img_train/{data_ind}'), dtype=np.float32)
        if self.mode != 'test':
            label = np.array(self.classnames[data_ind.split('.')[0]], dtype=np.long)

        # normalization.
        if max(image) > 1:
            image /= 255.0

        if self.mode != 'test':
            # for train and valid test create target dict.
            data_dict = {
                'image': image,
            }
            image = self.transforms(**data_dict)['image']
        else:
            data_dict = {
                'image': image,
            }
            image = self.transforms(**data_dict)['image']
       
        if self.mode != 'test':
            return image, label, data_ind
        else:
            return image, data_ind

    def __len__(self) -> int:
        return len(self.data_ids)


def evaluate_loss(y_hat, y, **kwargs):
    criterion_fn = torch.nn.functional.binary_cross_entropy
    assert False not in (y_hat.shape == y.shape), \
        f"y_hat.shape must be same as y.shape: {y_hat.shape} vs {y.shape}"
    return criterion_fn(y_hat, y, **kwargs)


def unzip(fr, to):
    from zipfile import ZipFile
    if os.path.isdir(to):
        pass
    else:
        os.mkdir(to)
    with ZipFile(fr) as f:
        for name in f.namelist():
            f.extract(name, to)


def get_trainval_datasets(cfg: DictConfig) -> dict:
    def is_valid_file(fp: str):
        def is_valid_size(size: Tuple, threshold: Tuple = (16, 16)):
            return size[0] > threshold[0] and size[1] > threshold[1]

        im = Image.open(fp)
        return is_valid_size(im)

    data_dir = f'{cfg.data.folder_path}'
    # check existed cache index file
    if os.path.exists(os.path.join(data_dir, 'trainval.txt')):
        with open(os.path.join(data_dir, 'trainval.txt', 'r')) as f:
            data_ids = f.readlines().split('\n')
    else:
        data_ids = [
            fname for fname in os.listdir(f'{data_dir}/train')
        ]
        # save index to cache file
        with open(os.path.join(data_dir, 'trainval.txt'), 'w') as f:
            f.writelines('\n'.join(filter(is_valid_file, data_ids)))

    train_ids, valid_ids = train_test_split(
        data_ids, test_size=0.1, random_state=cfg.training.seed)

    # for fast training
    if cfg.training.debug:
        train_ids = train_ids[:10]
        valid_ids = valid_ids[:10]

    # dataset
    dataset_class = load_obj(cfg.dataset.class_name)

    # initialize augmentations
    train_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['train']['augs']]
    train_augs = A.Compose(train_augs_list)

    valid_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['valid']['augs']]
    valid_augs = A.Compose(valid_augs_list)

    train_dataset = dataset_class(train_ids,
                                  'train',
                                  data_dir,
                                  cfg,
                                  train_augs)

    valid_dataset = dataset_class(valid_ids,
                                  'valid',
                                  data_dir,
                                  cfg,
                                  valid_augs)

    return {'train': train_dataset, 'valid': valid_dataset}


def get_test_dataset(cfg: DictConfig):
    """
    Get test dataset

    Args:
        cfg:

    Returns:

    """

    test_img_dir = f'{cfg.data.folder_path}/data55401/img_testA'

    valid_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['valid']['augs']]
    valid_augs = A.Compose(valid_augs_list)
    dataset_class = load_obj(cfg.dataset.class_name)

    test_dataset = dataset_class(None,
                                 'test',
                                 test_img_dir,
                                 cfg,
                                 valid_augs)

    return test_dataset


if __name__ == "__main__":
    import hydra
    import matplotlib.pyplot as plt
    import torch
    
    @hydra.main(config_path='../conf', config_name='config.yaml')
    def m(cfg: DictConfig):
        trainsets = get_trainval_datasets(cfg)['train']
        dataloader = torch.utils.data.DataLoader(trainsets, batch_size=16, num_workers=0)
        for i, (x, y, ind) in enumerate(dataloader):
            print(f'x: {x.shape} {x.dtype}')
            print(f'y: {y.shape} {y.dtype}')
            break
    m()
