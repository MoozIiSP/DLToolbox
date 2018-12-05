from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import multi_gpu_model


def LeNet5(input_shape=[28, 28, 1], classes=10):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     input_shape=input_shape,
                     padding='valid',
                     activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    return model


def compiled_model(config):
    model = LeNet5(input_shape=config['model_config']['input_shape'],
                   classes=len(config['image_config']['categories']))

    multi_gpu = config['project_config']['multi_gpu']
    if multi_gpu >= 2:
        model = multi_gpu_model(model, gpus=multi_gpu)

    # Compile Model
    model.compile(optimizer=config['model_config']['optimizer'],
                  loss=config['model_config']['loss'],
                  metrics=config['model_config']['metrics'])

    return model
