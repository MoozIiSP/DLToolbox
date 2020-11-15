import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from toolbox.application.Detection.UNet.models import UNet


class HookTriggered(Exception):
    def __init__(self, message):
        super(HookTriggered, self).__init__()


class BaseHook(object):
    def __init__(self, module):
        self.module = module
        self.hook_handler = None

    def module_hook(self, *args):
        def hook_fn(module, grad_in, grad_out):
            raise NotImplementedError
        # Hook the selected layer
        raise NotImplementedError


def preprocess_image(pil_im, resize_im=True):
    # mean and std list for channels (CIFAR10)
    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.2470, 0.2435, 0.2616]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
    #     im_as_arr[channel] -= mean[channel]
    #     im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = torch.autograd.Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    #reverse_mean = [-0.485, -0.456, -0.406]
    #reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    # for c in range(3):
    #     recreated_im[c] /= reverse_std[c]
    #     recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


class CNNLayerForwardHook(BaseHook):
    def __init__(self, net, module, input_size):
        super(CNNLayerForwardHook, self).__init__(module)
        self.net = net
        self.selected_module = module
        self.input_size = input_size
        self.ims = []

    def module_hook(self, selected_filter):
        def hook_fn(module, grad_in, grad_out):
            self.conv_output = grad_out[0, selected_filter]
            raise HookTriggered('The hook is triggered and stop forward')
        self.hook_handler = self.selected_module.register_forward_hook(hook_fn)

    def visualize_filter(self, selected_filter, iterations=30, input_image=None, norm=False):
        # Hook
        self.module_hook(selected_filter)
        fig = plt.figure()
        # noise_image = np.uint8(np.random.uniform(150, 150, self.input_size))
        if input_image is None:
            noise_image = np.uint8(np.random.random(self.input_size) * 30 + 127)
        else:
            noise_image = input_image
        processed_image = preprocess_image(noise_image, False)
        optimizer = optim.Adam([processed_image], lr=0.1)  # , weight_decay=1e-6)
        for i in range(1, iterations + 1):
            try:
                optimizer.zero_grad()
                x = self.net(processed_image)
            except HookTriggered as e:
                pass
            else:
                print('Unknown Exception Error and quit iteration')
                break
            loss = -torch.mean(self.conv_output)
            print(f'Filter {selected_filter} Iteration: {str(i)} Loss {loss.data.detach().numpy():.2f}')
            loss.backward()
            # Normalize grad
            if norm:
                for p in list(filter(lambda x: x.grad is not None, self.net.parameters())):
                    p.grad /= torch.sqrt(torch.mean(p.grad ** 2) + 1e-5)
            optimizer.step()
            # self.ims.append([plt.imshow(recreate_image(processed_image), animated=True)])
        # ani = animation.ArtistAnimation(fig, self.ims, interval=500, blit=True,
        #                                 repeat_delay=1000)
        # ani.save("vis.mp4")
        plt.imsave(f"results/{selected_filter}.png", recreate_image(processed_image)[:, :, 0], cmap='gray')
        # Remove the hook
        self.hook_handler.remove()


# net = models.vgg11(pretrained=True)
net = UNet(in_channels=1)
net.load_state_dict(torch.load('/home/mooziisp/GitRepos/DLToolbox/toolbox/snippets/unet.pth', map_location='cpu'))
print(net)

#deepdream = plt.imread('ada.jpg')

# selected_module = net.features[17]
# selected_module = net.classifier[-1]  # 1
selected_module = net.decoder1[5]
vis = CNNLayerForwardHook(net, selected_module, input_size=(512, 512, 1))
vis.visualize_filter(16, iterations=100, norm=False)

