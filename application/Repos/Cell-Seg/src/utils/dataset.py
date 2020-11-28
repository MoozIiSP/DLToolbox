import ast
import os
import sys

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


class BDCIDataset(Dataset):

    def __init__(self,
                 data_ids: list = None,
                 mode: str = 'train',
                 data_dir: str = '',
                 cfg: DictConfig = None,
                 transforms: Compose = None):
        """
        Prepare data for BDCI Seg competition.
        """
        self.data_ids = data_ids
        self.data_dir = data_dir
        self.mode = mode
        self.cfg = cfg
        self.transforms = transforms

    def __getitem__(self, idx: int):
        data_id = self.data_ids[idx].split('.')[0]
        # print(image_id)
        image = np.array(Image.open(f'{self.data_dir}/img_train/{data_id}.jpg'), dtype=np.float32)
        mask = np.array(Image.open(f'{self.data_dir}/lab_train/{data_id}.png'), dtype=np.long)

        # normalization.
        image /= 255.0

        # for train and valid test create target dict.
        data_dict = {
            'image': image,
            'mask':  mask,
        }
        # TODO: only apply flip to mask
        image, mask = self.transforms(**data_dict).values()

        return image, mask, data_id

    def __len__(self) -> int:
        return len(self.data_ids)


def get_training_datasets(cfg: DictConfig) -> dict:

    data_dir = f'{cfg.data.folder_path}'
    data_ids = [
        fname.split('.')[0] for fname in os.listdir(f'{data_dir}/img_train')
    ]

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
    
    @hydra.main(config_path='/home/aistudio/work/bdci2/conf', config_name='config.yaml')
    def m(cfg: DictConfig):
        trainsets = get_training_datasets(cfg)['train']
        dataloader = torch.utils.data.DataLoader(trainsets, batch_size=16, num_workers=0)
        for i, (x, y, ind) in enumerate(dataloader):
            print(f'x: {x.shape} {x.dtype}')
            print(f'y: {y.shape} {y.dtype}')
            break
    m()
