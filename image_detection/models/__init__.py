from __future__ import absolute_import

from keras.applications import *
from keras.optimizers import *
from keras.utils import multi_gpu_model

from .alexnet import AlexNet
from .lenet import LeNet5
from .resnet import ResnetBuilder
from .resnet18 import ResNet18_model
from .resnet50 import ResNet50_model


def optimizer_contructor(config):
    args = config['model_config']['optimizer_argments']

    opt_type = eval(config['model_config']['optimizer'])
    if opt_type == 'SGD':
        optimizer = SGD(args)
    elif opt_type == 'RMSprop':
        optimizer = RMSprop(*args)
    elif opt_type == 'Adagrad':
        optimizer = Adagrad(*args)
    elif opt_type == 'Adadelta':
        optimizer = Adadelta(*args)
    elif opt_type == 'Adam':
        optimizer = Adam(*args)
    elif opt_type == 'Adamax':
        optimizer = Adamax(*args)
    elif opt_type == 'Nadam':
        optimizer = Nadam(*args)
    else:
        raise ValueError('Please check supported optimizer of keras.')

    return optimizer


def wtf_model(config):
    selected_model = config['model_config']['model']
    input_shape = config['model_config']['input_shape']
    classes = len(config['image_config']['categories'])

    # Construct model
    if selected_model == 'ResNet50':
        model = ResNet50_model(config)
    elif selected_model == 'AlexCNN':
        model = AlexNet(input_shape=input_shape,
                        classes=classes)
    elif selected_model == 'LeNet5':
        model = LeNet5(input_shape=input_shape,
                       classes=classes)
    elif selected_model == 'ResNet18':
        # FIXME need to fix model = ResNet18_model(config)
        builder = ResnetBuilder()
        model = builder.build_resnet_18(input_shape=input_shape,
                                        num_outputs=classes)
    elif selected_model == 'DenseNet121':
        model = DenseNet121(include_top=True,
                            weights=None,
                            input_shape=input_shape,
                            classes=classes)
    elif selected_model == 'DenseNet169':
        model = DenseNet169(include_top=True,
                            weights=None,
                            input_shape=input_shape,
                            classes=classes)

    # Construct optimizer
    if not config['model_config']['optimizer_argments']:
        optimizer = optimizer_contructor(config)
    else:
        optimizer = config['model_config']['optimizer']

    multi_gpu = config['project_config']['multi_gpu']
    if multi_gpu >= 2:
        model = multi_gpu_model(model, gpus=multi_gpu)

    # Compile Model
    model.compile(optimizer=optimizer,
                  loss=config['model_config']['loss'],
                  metrics=config['model_config']['metrics'])

    return model
