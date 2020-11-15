import PIL
import torch
import torchvision
from torchvision import models, transforms

import numpy as np
import matplotlib.pyplot as plt


def imshow_v2(tensor, labels, nclass):
    """

    :param tensor: 
    :param labels: 
    :param nclass: 

    """
    plt.figure(figsize=(10,6))
    for i, img in enumerate(tensor):
        plt.subplot(4,4,i+1)
        plt.axis('off')
        # print labels
        plt.title(nclass[labels[i]])
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.tight_layout()
    plt.show()


def grayscale_to_pil(tensor, nrow, padding=2, pad_value=0):
    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(1) + padding), int(tensor.size(2) + padding)
    num_channels = 1
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    print(grid.shape)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def get_inter_act_map(x, model):
    with torch.no_grad():
        for name, mod in model.named_children():
            x = mod(x.cuda())
            if x.size(1) == 3:
                yield name, make_grid(x, 16).cpu().permute(1,2,0)
            else:
                yield name, grayscale_to_pil(intmap, 8)[0].cpu()


if __name__ == '__main__':
    NUM_CLASSES = 10
    device = torch.device('cpu')

    # Define categories of dataset and point to direction to generate data flow to train.
    classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    samples = torchvision.datasets.ImageFolder('../dataset/samples',
                                               transform=transforms.Compose([
                                                   transforms.Resize(100),
                                                   transforms.RandomCrop(96, 4),
                                                   transforms.RandomRotation((-180,180), resample=PIL.Image.BILINEAR),
                                                   transforms.ToTensor()
                                               ]))
    sample_loader = torch.utils.data.DataLoader(samples, batch_size=1,
                                                shuffle=False, num_workers=2)

    # Load a existed net into CPU device to visualize
    # Loading net shouldn't to require NetTrainer module
    checkpoint = torch.load('../classifier/resnet18-STL10-checkpoint.pth', map_location=device)
    net = models.resnet18(num_classes=NUM_CLASSES)
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()

    dataiter = iter(sample_loader)
    data, _ = dataiter.next()
    # plt.imshow(np.transpose(data[0].numpy(), (1, 2, 0)))
    # plt.show()
    # imshow_v2(data, net(data).argmax(dim=1), classes)

    for name, layer in net._modules.items():
        pass
