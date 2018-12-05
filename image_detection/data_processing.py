# coding: utf-8
# data_processor modules
# only read from dirs to generate hdf5 file
#
# Space and Time Trade
# ====================== 
# Our workflow is fellowing:
#   image files -> numpy file -> augmented file -> train model
# Old workflow:
#   image files -> hdf5 file -> train model
# if all data is stored in memory, then will be out of memory of GPU.
# So, we make a decision to choose a solution that stores all datas in disk,
# reading those into program when need to train model.
from __future__ import absolute_import

import os
import shutil

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array, save_img, array_to_img
from tqdm import tqdm

from .visualizer import plot_images


# Basic and some Tools
#=============================================================================
def walk(path, depth = 0, dirs = None):
    """读取path下所有可能的路径，并返回结果。
    path:       str
    depth:      int
    dirs:       dic
    Return:     dic"""
    if dirs == None:
        dirs = {}
    
    if depth in dirs.keys():
        dirs[depth].append(path)
    else:
        dirs[depth] = [path]
        
    subpaths = list(os.scandir(path))
    if len(subpaths) == 0:
        return 
    elif not subpaths[0].is_dir():
        return 
    
    for sp in subpaths:
        if sp.is_dir():
            walk(sp.path, depth + 1, dirs)
            
    return dirs


def get_file_from_path(path):
    """获取PATH下所有文件的路径。
    path:   str
    Return: list"""
    paths = list(walk(path).items())[-1][-1]

    files = []
    for p in paths:
        files.extend([ x.path for x in list(os.scandir(p)) ])

    return files
        

# 未分类图片的切割和预处理
# TODO 之后需要用到PyQt来显示分割图像，并打上相对应的标签
def split_generator(image, shape, N = 1):
    """均衡切分图片，将图片对半切成四分或者4^N分。
    Example:
    gen = split_generator(image, (384,256), N = 2)
    res1 = next(gen) # Split into 4 parts
    res2 = next(gen) # Split into 16 parts, again

    image:  Numpy object
    shape:  tuple
    N:      int
    Return: None
    """
    Queue = [image]
    h, w = shape
    for i in range(N):
        res = []
        x_splitline, y_splitline = map(int, (h/2, w/2))
        while Queue:
            _ = Queue.pop(0)
            # Fixed issue which PIL is converted to numpy.
            #if _.shape[0] != h:
            #    _ = _.transpose((1,0))
            res.extend([_[:x_splitline, :y_splitline],
                        _[:x_splitline, y_splitline:],
                        _[x_splitline:, :y_splitline],
                        _[x_splitline:, y_splitline:]])
        Queue = res.copy()
        h, w = x_splitline, y_splitline
        yield res


# FIXME
def image_processor(filepath, dest,
                    grayscale = True, keywords = None, num_split = 1):
    """将图片均匀分割为四个部分，保存为Numpy矩阵。
    fr是读取文件的完整路径，to是文件保存的完整路径。
    
    filepath:   str
    dset:         str
    grayscale:  boolean
    keywords:   dic, 分类
    num_split:  int, 切割次数
    Return:     None"""
    basename = os.path.basename(filepath)
    image_name, image_ext = os.path.splitext(basename)

    # Read grayscale image
    image = np.asarray(load_img(filepath, grayscale = grayscale))
    if image.ndim == 2:
        # Fixed issue which PIL is converted to numpy.
        #image = image.transpose((1,0))
        h, w = image.shape
        
    # Split image into 4 or more part
    gen = split_generator(image, (h, w), N = num_split)
    for i in range(num_split):
        imgs = next(gen)

    # TODO PyQt Rewrite
    # Re-Checker - thread unsafe, so you need run it under IPython.
    plot_images(imgs, (h, w), None, None, 2 ** num_split, 2 ** num_split, grayscale)

    # Set Label
    labeled = list(map(int, input("input position included defect (None is -1): ").split()))
    labels = [ 0 for i in range(4**num_split) ]
    for idx in labeled:
        labels[idx] = 1
        
    # Write images to disk
    for idx, img in enumerate(imgs):
        if keywords is None:
            img_name = '{}-{:02d}-{}'.format(image_name, idx, labels[idx])
        else:
            img_name = '{}-{:02d}-{}'.format(image_name, idx, keywords[labels[idx]])
        #np.save(os.path.join(to, img_name), img)
        # Save as a file, for example, png.
        # FIXME Need to test
        save_img(os.path.join(dest, image_name) + image_ext, array_to_img(img))


# Dataset Preprocessing
#==========================================================================================
def categorize(config, src_dir, dest_dir):
    """读取SRC_PATH下所有文件，然后生成对应标签，保存到DEST_DIR中。
    img_config: dict, 图像配置信息
    src_path:   str, 图像源目录
    dest_path:  str, 图像目标目录
    Return:     int"""
    grayscale = config['image_config']['grayscale']
    categories = config['image_config']['categories']
    input_shape = config['model_config']['input_shape']
    # For any datasets
    filters = [r'/{}'.format(x) for x in categories]

    # Categorize all files into two part or more.
    files = [sorted(list(filter(lambda x: f in x,
                                get_file_from_path(src_dir))))
             for f in filters]

    # The number of files of every category
    category_lengths = list(map(len, files))

    array_path = dest_dir
    shutil.rmtree(array_path, ignore_errors = True)
    os.makedirs(array_path)
    
    print('Iterating over all categories: ', categories)
    for category_idx, category in enumerate(categories):
        # Iterating over all files under every categories.
        for img_idx, img in tqdm(enumerate(files[category_idx]), ascii = True):
            color_mode = 'grayscale' if grayscale else None
            # TODO It's OK, you don't need to fix channels_last
            img = img_to_array(load_img(img, target_size=input_shape, color_mode=color_mode))
        
            img_name = '{}-img-{}-{}'.format(img_idx, category, category_idx)
            label_name = '{}-label-{}-{}'.format(img_idx, category, category_idx)
            
            label = np.eye(len(categories), dtype = np.float32)[category_idx]
            
            img_array_path = os.path.join(array_path, img_name)
            img_label_path = os.path.join(array_path, label_name)

            np.save(img_array_path, equalizeHist1D(img))
            np.save(img_label_path, label)

    # FIXME Calculus category_rounds
    category_lengths = np.array(category_lengths) / sum(category_lengths)
    category_lengths = list(category_lengths / max(category_lengths))
    category_rounds = {cat: min(int(np.round(1 / l)), 10) for cat, l in zip(categories, category_lengths)}

    # FIXME Save rounds to config
    config['image_config']['category_rounds'] = category_rounds
    print('category rounds', category_rounds)

    return category_rounds
    

def gen_arrays_from_dir(array_dir):
    """生成器，读取ARRAY_DIR下所有文件并生成Numpy矩阵。
    array_dir:  str
    Return:     None"""
    array_files = sorted(os.listdir(array_dir))
    array_names = list(filter(lambda x: r'-img-' in x, array_files))
    label_names = list(filter(lambda x: r'-label-' in x, array_files))

    assert len(array_names) == len(label_names)

    for array_name, label_name in zip(array_names, label_names):
        array = np.load(os.path.join(array_dir, array_name))
        label = np.load(os.path.join(array_dir, label_name))
        yield array, label, label_name


def gen_augment_arrays(array, label, augmentations, rounds = 1):
    """生成器，从ARRAY和LABEL中生成增强矩阵和相应标签。
    array:          Numpy object
    label:          list
    augmentations:  dic or json
    rounds:         int
    Return:         None"""
    if augmentations is None:
        yield array, label
    else:
        # FIXME
        auggen = ImageDataGenerator(featurewise_center = augmentations['featurewise_center'],
                                    samplewise_center = augmentations['samplewise_center'],
                                    featurewise_std_normalization = augmentations['featurewise_std_normalization'],
                                    samplewise_std_normalization = augmentations['samplewise_std_normalization'],
                                    zca_whitening = augmentations['zca_whitening'],
                                    rotation_range = augmentations['rotation_range'],
                                    width_shift_range = augmentations['width_shift_range'],
                                    height_shift_range = augmentations['height_shift_range'],
                                    shear_range = augmentations['shear_range'],
                                    zoom_range = augmentations['zoom_range'],
                                    channel_shift_range = augmentations['channel_shift_range'],
                                    fill_mode = augmentations['fill_mode'],
                                    cval = augmentations['cval'],
                                    horizontal_flip = augmentations['horizontal_flip'],
                                    vertical_flip = augmentations['vertical_flip'],
                                    rescale = augmentations['rescale'],
                                    data_format = augmentations['data_format'])

        # FIXME only for featurewise
        # auggen.fit(array)
        array_augs, label_augs = next(auggen.flow(np.tile(array[np.newaxis],
                                                          (rounds * augmentations['rounds'], 1, 1, 1)),
                                                  np.tile(label[np.newaxis],
                                                          (rounds * augmentations['rounds'], 1)),
                                                  batch_size=rounds * augmentations['rounds']))

        for array_aug, label_aug in zip(array_augs, label_augs):
            yield array_aug, label_aug


def augment_arrays(config, path):
    """图像增强
    path:   str
    Return: None"""
    proj_dir = config['project_config']['proj_dir']
    augmentations = config['augment_config']
    categories = config['image_config']['categories']
    category_rounds = config['image_config']['category_rounds']
    
    array_path = os.path.join(proj_dir, 'array')
    augmented_path = os.path.join(proj_dir, 'augmented')
    shutil.rmtree(augmented_path, ignore_errors=True)
    os.makedirs(augmented_path)
    
    if augmentations is None:
        print('No augmentations selected: copying train arrays as is.')
        files = os.listdir(array_path)
        for file in tqdm(files, ascii=True):
            shutil.copy(os.path.join(array_path, file),augmented_path)
    else:
        print('Generating image augmentations:')

        for img_idx, (array, label, label_name) in tqdm(enumerate(gen_arrays_from_dir(array_path))):
            split_label_name = '-'.join(label_name.split('-')[2:-1])
            for aug_idx, (array_aug, label_aug) in enumerate(gen_augment_arrays(array, label, augmentations, category_rounds[split_label_name])):
                cat_idx = np.argmax(label_aug)
                cat = categories[cat_idx]
                img_name = '{}-{:02d}-img-{}-{}'.format(img_idx, aug_idx,
                                                            cat, cat_idx)
                label_name = '{}-{:02d}-label-{}-{}'.format(img_idx, aug_idx,
                                                            cat, cat_idx)
                aug_path = os.path.join(augmented_path, img_name)
                label_path = os.path.join(augmented_path, label_name)

                np.save(aug_path, array_aug)
                np.save(label_path, label_aug)


# NOTE Train Model
def gen_minibatches(array_dir, array_files, batch_size):
    '''生成器，从ARRAY_DIR目录中读取BATCH_SIZE数量的文件。
    array_dir:      str
    array_files:    list
    batch_size:     int
    Return:         None'''
    while True:
        np.random.shuffle(array_files)
        array_names_mb = array_files[:batch_size]

        arrays = []
        labels = []
        for array_name in array_names_mb:
            img_path = os.path.join(array_dir, array_name)

            arrays.append(np.load(img_path))
            labels.append(np.load(img_path.replace('-img-', '-label-')))
        yield np.array(arrays), np.array(labels)


def equalizeHist1D(src):
    from cv2 import equalizeHist
    shape = src.shape
    x = src.copy().astype('uint8')
    return equalizeHist(x).astype('float32').reshape(shape)
