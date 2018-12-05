# encode: utf-8
# WARNING: keras-vis has a bug, causing to throw a tensorflow exception
# when you want to plot convolution or activation on any GPU platforms.
# You need to install latest keras-vis from Github to solve this problem.
from __future__ import absolute_import

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from keras import activations
from keras.preprocessing.image import save_img
from vis.utils import utils
from vis.visualization import get_num_filters
from vis.visualization import visualize_activation


# FIXME from tensorflow.keras.preprocessing.image import save_img
# from keras.applications import resnet50


def plot_images(imgs, shape,
                labels, keywords,
                rows, columns, grayscale=False):
    """在单个Figure同时显示多个图像
    imgs:       list
    shape:      tuple
    labels:     list
    keywords:   dic"""
    #Keywords example
    #dic = {0:'free-defect', 1:'defect'}
    
    fig = plt.figure()
    for idx, img in enumerate(imgs):
        fig.add_subplot(rows, columns, idx+1)
        if labels != None:
            plt.title(keywords[labels[idx - 1]])
        else:
            plt.title(str(idx))

        plt.axis("off")
        if grayscale:
            plt.imshow(img, cmap=cm.Greys_r)
        else:
            plt.imshow(img)
    plt.show()


def plot_history(history):
    # disable_gui = True if os.environ['QT_QPA_PLATFORM'] == 'offscreen' else False
    fig = plt.figure()
    keys = history.keys()

    datas = [[x for x in keys if 'acc' in x],
             [x for x in keys if 'loss' in x]]

    for idx, data in enumerate(datas):
        fig.add_subplot(1, 2, idx + 1)
        for d in data:
            plt.plot(history[d])
        key = 'accuracy' if idx == 0 else 'loss'
        plt.title('model {0}'.format(key))
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['valid', 'train'], loc='upper left')

    # if disable_gui:
    #     plt.imsave('history.png')
    # else:
    #     plt.show()
    plt.show()


# FIXME dont fit for DenseNet, only ResNet
def plot_weights_of_layer(fpath, model, layer_name, filter_num=None):
    layer_idx = utils.find_layer_idx(model, layer_name)

    if filter_num:
        weights, bias = model.layers[layer_idx].get_weights()[:filter_num]
    else:
        weights, bias = model.layers[layer_idx].get_weights()

    vis_weights = []
    for idx in range(weights.shape[-1]):
        vis_weights.append(weights[..., idx])
    stitched = utils.stitch_images(vis_weights, cols=5)

    save_img(fpath, stitched)


# TODO
def plot_image_after_propagation(model, layer_name, image):
    layer_idx = utils.find_layer_idx(model, layer_name)
    layers_list = model.layers[:layer_idx+1]

    x = image
    for layer in layers_list:
        x = layer(x)

    # Need vary of processer for output.
    raise NotImplementedError


def plot_dense_layer(model, layer_name, cat):
    """Visualizing a specific output category.
    model must include top.
    Easy"""
    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    plt.rcParams['figure.figsize'] = (18, 6)
    # filter_indices is the imagenet category
    img = visualize_activation(model, layer_idx, filter_indices=cat, max_iter=1, verbose=True)

    if img.shape[2] == 1:
        plt.imshow(img.reshape(img.shape[:1]))
    else:
        plt.imshow(img)


def plot_random_dense_layer():
    # [layer.name for layer in model.layers if 'conv' in layer.name]
    # for layer_name in conv_layers:
    #     ...: layer_idx = utils.find_layer_idx(model, layer_name)
    #     ...: filters = np.random.permutation(get_num_filters(model.layers[layer_idx]))[:10]
    #     ...: selected_indices.append(filters)
    #     ...: vis_images = []
    #     ...:
    #     for idx in filters:
    #         ...: img = visualize_activation(model, layer_idx, filter_indices=idx, tv_weight=0,
    #                                         ...: input_modifiers = [Jitter(0.05)])
    #         ...: vis_images.append(img)
    #         ...: stitched = utils.stitch_images(vis_images, cols=5)
    #         ...: save_img('/home/mooziisp/drive/xr-{}.png'.format(layer_name), stitched)

    # # Generate input image for each filter.
    # new_vis_images = []
    # for i, idx in enumerate(filters):
    #     # We will seed with optimized image this time.
    #     img = visualize_activation(model, layer_idx, filter_indices=idx,
    #                                seed_input=vis_images[i],
    #                                input_modifiers=[Jitter(0.05)])
    #
    #     # Utility to overlay text on image.
    #     img = utils.draw_text(img, 'Filter {}'.format(idx))
    #     new_vis_images.append(img)
    #
    # # Generate stitched image palette with 5 cols so we get 2 rows.
    # stitched = utils.stitch_images(new_vis_images, cols=5)
    raise NotImplementedError


def plot_conv_filters(fdir, model, layer_name):
    # The name of the layer we want to visualize
    # You can see this in the model definition.
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Visualize all filters in this layer.
    filters = np.arange(get_num_filters(model.layers[layer_idx]))

    # Generate input image for each filter.
    for g in range(len(filters)//16):
        vis_images = []
        for idx in filters[g*16:(g+1)*16]:
            img = visualize_activation(model, layer_idx, filter_indices=idx)

            # Utility to overlay text on image.
            # img = utils.draw_text(img, 'Filter {}'.format(idx))
            vis_images.append(img)

        # Generate stitched image palette with 8 cols.
        stitched = utils.stitch_images(vis_images, cols=8)
        save_img(fdir + 'conv-vis-{:02d}.png'.format(g), stitched)

    # plt.axis('off')
    # plt.imshow(stitched)
    # plt.title(layer_name)
    # plt.show()


def plot_attention():
    raise NotImplementedError


# # 特征图显示
# def plot_layer(layer_name, layer_feature, num_of_filter):
#     import numpy as np
#     import time
#     from keras.preprocessing.image import save_img
#     from keras import backend as K
#
#     img_size = [256, 384]
#     layer_name = 'conv1'
#
#     def deprocess_image(x):
#         x -= x.mean()
#         x /= (x.std() + K.epsilon())
#         x *= 0.1
#
#         # clip to [0, 1]
#         x += 0.5
#         x = np.clip(x, 0, 1)
#
#         # convert to RGB array
#         x *= 255
#         if K.image_data_format() == 'channels_first':
#             x = x.transpose((1, 2, 0))
#         x = np.clip(x, 0, 255).astype('uint8')
#         return x
#
#     # build the VGG16 network with ImageNet weights
#     # model = vgg16.VGG16(weights='imagenet', include_top=False)
#     config = {}
#     config['optimizer'] = 'SGD'
#
#     model = alexcnn.prototype_model(config)
#     print('Model loaded.')
#
#     model.summary()
#
#     # this is the placeholder for the input images
#     input_img = model.input
#
#     # get the symbolic outputs of each "key" layer (we gave them unique names).
#     layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
#
#     def normalize(x):
#         # utility function to normalize a tensor by its L2 norm
#         return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
#
#     kept_filters = []
#     for filter_index in range(200):
#         # we only scan through the first 200 filters,
#         # but there are actually 512 of them
#         print('Processing filter %d' % filter_index)
#         start_time = time.time()
#
#         # we build a loss function that maximizes the activation
#         # of the nth filter of the layer considered
#         layer_output = layer_dict[layer_name].output
#         if K.image_data_format() == 'channels_first':
#             loss = K.mean(layer_output[:, filter_index, :, :])
#         else:
#             loss = K.mean(layer_output[:, :, :, filter_index])
#
#         # we compute the gradient of the input picture wrt this loss
#         grads = K.gradients(loss, input_img)[0]
#
#         # normalization trick: we normalize the gradient
#         grads = normalize(grads)
#
#         # this function returns the loss and grads given the input picture
#         iterate = K.function([input_img], [loss, grads])
#
#         # step size for gradient ascent
#         step = 1.
#
#         # we start from a gray image with some random noise
#         if K.image_data_format() == 'channels_first':
#             input_img_data = np.random.random((1, 3, img_width, img_height))
#         else:
#             input_img_data = np.random.random((1, img_width, img_height, 3))
#         input_img_data = (input_img_data - 0.5) * 20 + 128
#
#         # we run gradient ascent for 20 steps
#         for i in range(20):
#             loss_value, grads_value = iterate([input_img_data])
#             input_img_data += grads_value * step
#
#             print('Current loss value:', loss_value)
#             if loss_value <= 0.:
#                 # some filters get stuck to 0, we can skip them
#                 break
#
#         # decode the resulting input image
#         if loss_value > 0:
#             img = deprocess_image(input_img_data[0])
#             kept_filters.append((img, loss_value))
#         end_time = time.time()
#         print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
#
#     # we will stich the best 64 filters on a 8 x 8 grid.
#     n = 8
#
#     # the filters that have the highest loss are assumed to be better-looking.
#     # we will only keep the top 64 filters.
#     kept_filters.sort(key=lambda x: x[1], reverse=True)
#     kept_filters = kept_filters[:n * n]
#
#     # build a black picture with enough space for
#     # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
#     margin = 5
#     width = n * img_width + (n - 1) * margin
#     height = n * img_height + (n - 1) * margin
#     stitched_filters = np.zeros((width, height, 3))
#
#     # fill the picture with our saved filters
#     for i in range(n):
#         for j in range(n):
#             img, loss = kept_filters[i * n + j]
#             stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
#             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
#
#     # save the result to disk
#     save_img('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
#
#     # for i in range(num_of_filter):
#     #    plt.imsave('{0}-{:2d}.png'.format(layer_name, i), layer_feature[:,:,i])
