# coding: utf-8
# FIXME 将训练从主程序中剥离出去，训练的代码由Jupyter Notebook完成。所以本程序只提供以下几个功能
# 图像的预处理、分割、模型的可视化
from __future__ import absolute_import

from math import ceil
from random import shuffle

import pandas as pd

from . import data_processing
from .models import wtf_model
from .project import *
from .transfer_tools import *
from .utils import timecost_of, hint_color
from .visualizer import *
from .weights import *


def splitor_for_images():
    # TODO 应该独立出来
    # 1 Split into 4 parts
    # if not GUI_support:
    #     # TODO will pass all dataset into the QtApplication. separated from this.
    #     for file in files:
    #         data_processing.image_processor(file, os.path.join(proj_dir, 'array'))
    # else:
    #     print("Current environment don't support QtGui."
    #           "If you want to use split tool, please run it in the desktop environment.")
    raise NotImplementedError


def preprocess(config):
    """processing data"""
    # configuration
    proj_dir = config['project_config']['proj_dir']
    enb_gui = config['project_config']['enable_GUI_support']
    enb_augment = config['project_config']['enable_augmented']

    image_dir = os.path.join(proj_dir, 'images')
    # Get all path stored images
    files = data_processing.get_file_from_path(image_dir)

    # Categorize - gen array
    if os.path.exists(os.path.join(proj_dir, 'array')):
        print('array founded.')
    else:
        print('Categorizing all files...')
        data_processing.categorize(config,
                                   os.path.join(proj_dir, 'images'),
                                   os.path.join(proj_dir, 'array'))

    # Augment data
    if os.path.exists(os.path.join(proj_dir, 'augmented')):
        print('augmented founded.')
    elif enb_augment:
        print('Augmenting all files...')
        data_processing.augment_arrays(config, os.path.join(proj_dir, 'array'))

    # Ready for train, and generate file list
    coefficient = 0.6

    if enb_augment:
        array_dir = os.path.join(proj_dir, 'augmented')
        array_files = list(filter(lambda x: r'-img-' in x,
                                  os.listdir(array_dir)))
    else:
        array_dir = os.path.join(proj_dir, 'array')
        array_files = list(filter(lambda x: r'-img-' in x,
                                  os.listdir(array_dir)))

    # Split files needs to save as a list.
    shuffle(array_files)
    s1 = int(ceil(coefficient * len(array_files)))
    s2 = int(ceil((1 + coefficient) / 2 * len(array_files)))

    print('{} train samples, {} valid samples, {} test samples'.format(s1, s2 - s1, len(array_files) - s2))

    # Save with pickle
    # FIXME need to clean proj dir
    save_pk(array_files[:s1], os.path.join(proj_dir, 'train_index'))
    save_pk(array_files[s1:s2], os.path.join(proj_dir, 'valid_index'))
    save_pk(array_files[s2:], os.path.join(proj_dir, 'test_index'))

    print('Pre-processing Done.')


def train(config, model):
    # Configuration
    proj_dir = config['project_config']['proj_dir']
    enb_augment = config['project_config']['enable_augmented']
    batch_size = config['model_config']['batch_size']
    epochs = config['model_config']['epochs']

    # Loading Dataset to prepare for train
    if enb_augment:
        array_dir = os.path.join(proj_dir, 'augmented')
    else:
        array_dir = os.path.join(proj_dir, 'array')

    train_index = load_pk(os.path.join(proj_dir, 'train_index'))
    valid_index = load_pk(os.path.join(proj_dir, 'valid_index'))

    gen_train = data_processing.gen_minibatches(array_dir,
                                                train_index,
                                                batch_size)
    gen_valid = data_processing.gen_minibatches(array_dir,
                                                valid_index,
                                                batch_size)
    
    history = model.fit_generator(gen_train,
                                  steps_per_epoch=ceil(len(train_index) / batch_size),
                                  epochs=epochs,
                                  validation_data=gen_valid,
                                  validation_steps=ceil((len(valid_index)) / batch_size))

    model_filename = '{}-{}-loss{:.02f}.h5'.format(config['project_config']['proj_name'],
                                                   config['model_config']['model'],
                                                   history.history['val_loss'][-1])

    print(hint_color('Save model as {} and all as {}.'.format(model_filename,
                                                              'tmp_whole_model.h5')))
    model.save_weights(os.path.join(proj_dir, model_filename))
    model.save(os.path.join(proj_dir, 'tmp_whole_model.h5'))

    return history, model


# FIXME Provide a Dataset as a parameter of function
def evaluate(config, model):
    proj_dir = config['project_config']['proj_dir']
    enb_augment = config['project_config']['enable_augmented']
    batch_size = config['model_config']['batch_size']

    if enb_augment:
        array_dir = os.path.join(proj_dir, 'augmented')
    else:
        array_dir = os.path.join(proj_dir, 'array')

    test_index = load_pk(os.path.join(proj_dir, 'test_index'))
    gen_test = data_processing.gen_minibatches(array_dir,
                                               test_index,
                                               batch_size)

    score, timer = timecost_of(1, model.evaluate_generator, gen_test, ceil(len(test_index) / batch_size))
    print("Total loss on Testing Set:", score[0])
    print("Accuracy of Testing Set:", score[1])
    print("All avg cost: {:.04f} sec".format(timer / len(test_index)))


def clean():
    """Clean all file except images and config files."""
    raise NotImplementedError


def load_model():
    raise NotImplementedError


# FIXME model may be a class
def predict(config, model, array_dir=None):
    """预测一个正常的图片"""
    # categories = ['freedefect', 'defect']
    categories = config['image_config']['categories']
    pred_file_dir = os.path.join(config['project_config']['proj_dir'], 'array') if array_dir else array_dir
    pred_files = list(filter(lambda x: r'-img-' in x, os.listdir(pred_file_dir)))

    # Build a generator with samplewise
    augmentations = {'rounds': 1,
                     'featurewise_center': False,  # FIXME false to ture
                     'samplewise_center': True,
                     'featurewise_std_normalization': False,
                     'samplewise_std_normalization': True,
                     'zca_whitening': False,
                     'rotation_range': 0.,
                     'width_shift_range': 0.,
                     'height_shift_range': 0.,
                     'shear_range': 0.,
                     'zoom_range': 0.,
                     'channel_shift_range': 0.,
                     'fill_mode': 'nearest',
                     'cval': 0.,
                     'horizontal_flip': False,
                     'vertical_flip': False,
                     'rescale': 1 / 255,
                     'preprocessing_function': None,
                     'data_format': "channels_last"}

    # Building a csv format
    columns = ['filename', 'predicted', 'categories']
    output = []
    for f in pred_files:
        # Load image
        filename = os.path.join(pred_file_dir, f)
        x = np.load(filename)

        # Format
        auggen = data_processing.gen_augment_arrays(x, np.array([]), augmentations)
        x = next(auggen)

        # Predict
        pred_val = model.predict(np.expand_dims(x[0], axis=0))
        pred = categories[np.argmax(pred_val)]
        output.append([filename, pred_val, pred])
    pred_df = pd.DataFrame(output, columns=columns)
    pred_df.to_csv('~/pred.csv', index=False)

    # Compute accuracy
    cnt = 0
    for fn, pred, cat in output:
        if '-{}-'.format(cat) in fn:
            cnt += 1
    print(hint_color('Accuracy: {:0.4f}'.format(cnt / len(output))))


def main(*args):
    # parser = argparse.ArgumentParser("a tool to train CNN model.")
    #
    # parser.add_argument()

    proj_dir = os.path.expanduser(input('Project Path: '))

    if os.path.exists(os.path.join(proj_dir, 'config.yaml')):
        config = load_yaml(os.path.join(proj_dir, 'config.yaml'))
        print('Config Loaded.')
    else:
        proj_name = input('Project Name: ')
        save_config(proj_name, proj_dir, default=False)
        print('Save to {0}'.format(proj_dir))
        config = load_yaml(os.path.join(proj_dir, 'config.yaml'))
        print('Config Loaded.')

    # parser = argparse.ArgumentParser(
    #     description='A tool to processing images and to train via those dataset.')

    print(hint_color('Step 1: Preparing dataset for train...'))
    preprocess(config)

    print(hint_color('Eval Model Enable in the future...'))
    eval_mode = False
    if not os.path.exists(os.path.join(proj_dir, 'tmp_whole_model.h5')):
        model = wtf_model(config)
    else:
        model = load_model(os.path.join(proj_dir, 'tmp_whole_model.h5'))
        eval_mode = True

    switch = input_with_y_or_n('Load first convolution layer weights of pre-trained model [y/N]: ', 'N')
    if switch:
        layer_name, weight = load_weights(config['model_config']['model'], grayscale=config['image_config']['grayscale'])
        inject_weights_to_layer(model, layer_name, weight)
        print(hint_color('{} Loaded.'.format(layer_name)))

    # history visualization
    if not eval_mode:
        print(hint_color('Step 2: Starting feed dataset to model...'))
        # FIXME
        ret, timer = timecost_of(1, train, config, model)

    print(hint_color('Step 3: Evaluating performance on test set...'))
    evaluate(config, ret[1])

    print(hint_color('Step 4: Save history of training...'))
    save_pk(ret[0].history, os.path.join(proj_dir, 'history.log'))

    # print(hint_color('Step 5: Output visualization of model...'))
    # layer_idx = ret[1].layers[-1]
    # plot_dense_layer()

    print(hint_color('All Done: Configuration saved.'))
    save_yaml(config, os.path.join(proj_dir, 'config.yaml'))

    
if __name__ == '__main__':
    main()
