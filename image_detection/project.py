# coding: utf-8
from __future__ import absolute_import

import os
import shutil
from pickle import load, dump

import yaml

from .utils import hint_color, warning_color, input_with_y_or_n


def init_config(path):
    """this is a default configuration."""
    config = dict()

    # TODO Project Configuration
    proj_config = dict()
    proj_config['proj_name'] = None
    proj_config['proj_dir'] = path
    # FIXME You need manually change it
    proj_config['enable_GUI_support'] = False
    proj_config['enable_augmented'] = True
    proj_config['multi_gpu'] = 1

    # TODO Image Configuration
    image_config = dict()
    image_config['categories'] = None  # FIXME 预处理之后会赋值
    image_config['category_rounds'] = None
    image_config['size'] = None
    image_config['grayscale'] = None
    image_config['meanify'] = True  # FIXME

    # FIXME model definition
    model_config = dict()
    model_config['input_shape'] = None  # FIXME 需要指定
    model_config['epochs'] = 25
    model_config['batch_size'] = 32
    model_config['model'] = 'ResNet50'
    model_config['include_top'] = None
    model_config['pooling_type'] = 'max'
    model_config['optimizer'] = 'SGD'  # TODO 扩展
    model_config['optimizer_argments'] = None
    model_config['loss'] = 'categorical_crossentropy'
    model_config['metrics'] = ['categorical_accuracy']

    # TODO augment configuration
    augmentations = {'rounds': 8,
                     'featurewise_center': False,  # FIXME false to ture
                     'samplewise_center': False,
                     'featurewise_std_normalization': False,
                     'samplewise_std_normalization': False,
                     'zca_whitening': False,
                     'rotation_range': 15,
                     'width_shift_range': 0.,
                     'height_shift_range': 0.,
                     'shear_range': 0.,
                     'zoom_range': 0.,
                     'channel_shift_range': 20.,
                     'fill_mode': 'nearest',
                     'cval': 0.,
                     'horizontal_flip': True,
                     'vertical_flip': True,
                     'rescale': 1 / 255,
                     'preprocessing_function': None,
                     'data_format': "channels_last"}

    config['project_config'] = proj_config
    config['image_config'] = image_config
    config['model_config'] = model_config
    config['augment_config'] = augmentations

    return config


# FIXME Temporary
def save_config(name, path, default=True):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.mkdir(path)
        print('{0} created.'.format(path))
    else:
        print('{0} founded.'.format(path))

    config = init_config(path)
    config['project_config']['proj_name'] = name

    if os.path.exists(os.path.join(config['project_config']['proj_dir'], 'images')):
        config['image_config']['categories'] = os.listdir(os.path.join(config['project_config']['proj_dir'], 'images'))
    else:
        print(warning_color('Please check whether your images directory exists.'))

    if default == False:
        config['project_config']['enable_augmented'] = input_with_y_or_n('Enable augmented dataset [Y/n]: ', 'y')
        config['project_config']['multi_gpu'] = int(input(hint_color('How many GPUs do you have? '))) or 1
        # FIXME
        config['image_config']['size'] = eval(
            input(hint_color('If all images are same, then images size (ex. (224,224)): '))) or None
        config['image_config']['grayscale'] = input_with_y_or_n(hint_color('Is image grayscale? [y/N]: '), 'n')
        config['model_config']['input_shape'] = tuple(
            list((256, 384)) + [1] if config['image_config']['grayscale'] else [3])
        config['model_config']['epochs'] = int(input(hint_color('How many epochs? ')))
        config['model_config']['batch_size'] = int(input(hint_color('How many batch size? ')))
        config['model_config']['model'] = input(
            hint_color('Choose a model to fit your datasets (default ResNet50): ')) or 'ResNet50'
        config['model_config']['include_top'] = input(hint_color('Do you want to train a model? [Y/n]: ')) or True
        config['model_config']['optimizer'] = input(
            hint_color('Choose a optimizer for your model (default SGD): ')) or 'SGD'
        config['model_config']['optimizer_argments'] = input(
            hint_color('Please provide some argments for optimizer or leave blank (ex. [0.001]): ')) or None
        config['model_config']['loss'] = input(hint_color(
            'Choose a loss for your model (default categorical_crossentropy): ')) or 'categorical_crossentropy'
        # FIXME
        x = input(hint_color('Choose a metrics for your model (default [\'categorical_accuracy\']): ')) \
            or "['categorical_accuracy']"
        config['model_config']['metrics'] = eval(x)
        config['model_config']['pooling_type'] = input(
            hint_color('Choose one of pooling types for your model (default MaxPooling): ')) or 'max'

        # TODO augmentation configuration settings

    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        config = yaml.load(f)
    return config


def save_yaml(config, filepath):
    if os.path.exists(filepath):
        shutil.rmtree(filepath, ignore_errors=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f)


def save_pk(pk, filepath):
    with open(filepath, 'wb') as f:
        dump(pk, f)


def load_pk(filepath):
    with open(filepath, 'rb') as f:
        res = load(f)
    return res