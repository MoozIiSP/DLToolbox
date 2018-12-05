"""ResNet18 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from keras_applications import get_keras_submodule

backend = get_keras_submodule('backend')
layers = get_keras_submodule('layers')
models = get_keras_submodule('models')
keras_utils = get_keras_submodule('utils')

from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import _obtain_input_shape

preprocess_input = imagenet_utils.preprocess_input


# ResNet18 doesn't have pre-trained model.
# WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
#                 'releases/download/v0.2/'
#                 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
# WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
#                        'releases/download/v0.2/'
#                        'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


def r18_identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, kernel_size,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    print(x.shape)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    print(x.shape, input_tensor.shape)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def r18_conv_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    print(input_tensor.shape)

    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, kernel_size, strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    print(x.shape)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    print(x.shape)

    shortcut = layers.Conv2D(filters2, (1, 1), strides=strides,
                             padding='valid',
                             kernel_initializer='he_normal',
                             # kernel_regularizer=l2(0.0001),
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    print(shortcut.shape, input_tensor)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet18(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet18 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # No available pre-training on ImageNet
    if not (weights in {None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization),'
                         'or the path to the weights file to be loaded.')

    # if weights == 'imagenet' and include_top and classes != 1000:
    #     raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
    #                      ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    print(x.shape)

    # Conv2_x
    x = r18_conv_block(x, 3, [64, 64], stage=2, block='a', strides=(1, 1))
    x = r18_identity_block(x, 3, [64, 64], stage=2, block='b')

    # Conv3_x
    x = r18_conv_block(x, 3, [128, 128], stage=3, block='a')
    x = r18_identity_block(x, 3, [128, 128], stage=3, block='b')

    # Conv4_x
    x = r18_conv_block(x, 3, [256, 256], stage=4, block='a')
    x = r18_identity_block(x, 3, [256, 256], stage=4, block='b')

    # Conv5_x
    x = r18_conv_block(x, 3, [512, 512], stage=5, block='a')
    x = r18_identity_block(x, 3, [512, 512], stage=5, block='b')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet18(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet18')

    # Load weights.
    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = keras_utils.get_file(
    #             'resnet18_weights_tf_dim_ordering_tf_kernels.h5',
    #             WEIGHTS_PATH,
    #             cache_subdir='models',
    #             md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    #     else:
    #         weights_path = keras_utils.get_file(
    #             'resnet18_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #             WEIGHTS_PATH_NO_TOP,
    #             cache_subdir='models',
    #             md5_hash='a268eb855778b3df3c7506639542a6af')
    #     model.load_weights(weights_path)
    #     if backend.backend() == 'theano':
    #         keras_utils.convert_all_kernels_in_model(model)
    # elif weights is not None:
    if weights is not None:
        model.load_weights(weights)

    return model


def ResNet18_model(config):
    return ResNet18(include_top=config['model_config']['include_top'],
                    weights=None,
                    input_shape=config['model_config']['input_shape'],
                    pooling=config['model_config']['pooling_type'],
                    classes=len(config['image_config']['categories']))
