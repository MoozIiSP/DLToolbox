"""
用于测试特征编码中元素的局部模式的偏移对检测效果的影响，同时展示出该局
  部模式对最终检测结果的贡献程度。
特征图中的局部模式：
- 1D :: 通过对一维向量平移offset个单位，来改变对检测结果有所贡献的局部
  模式。
- 2D :: 通过对特征图移动若干个坐标，来改变对检测结果又所贡献的局部模式。
  可能会破坏局部模式。一般而言具有一定的稳定性。

可知，CNN能够对图像中的各种目标进行编码，从而生成对应的局部编码模式来
  代表该目标。随后，再将该特征图转递至分类器（组合？），以进行分类。而
  类似在分类器之前使用AvgPool池化层输出1x1大小的特征的ResNet网络结构
  而言，其特征提前深层所检测到的局部模式应该是更加稀疏，例如池化层中的
  特征激活图的一部分只对蓝色眼睛或者其他局部的复合特征有响应，而不是对
  整体特征具有响应。然而，将特征图扁平化之后，原来的特征模式则不再相邻
  ，而是按规律排列在一维特征向量中。当平移n个单位的时候，可能会破坏那些
  对检测结果有显著贡献的局部模式（视觉模式），导致检测结果偏移。
同时，参考NLP的常见模型，几乎都是对句子中，词组和词组之间的关系模式，来
  进行特定的推断，已得到检测结果。所以，CV和NLP在该方向上应该是一致的。
特征向量即是知识。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Manifold2D(nn.Module):
    """When eval mode, this Layer will disable."""
    def __init__(self) -> None:
        super(Manifold2D, self).__init__()
        pass

    def _manifold(self, x: torch.Tensor, pos=None):
        w, h = x.shape[2:]
        if isinstance(pos, list) or isinstance(pos, tuple):
            x_pos, y_pos = pos
        else:
            x_pos = random.randint(0, w)
            y_pos = random.randint(0, h)

        pad_x, pad_y = w // 2, h // 2
        x_hat = F.pad(x, [pad_x, pad_x, pad_y, pad_y], mode='circular')
        return x_hat[:, :, x_pos:x_pos + w, y_pos:y_pos + h]

    def forward(self, x, pos=None):
        if isinstance(pos, list) or isinstance(pos, tuple):
            return self._manifold(x, pos)
        return self._manifold(x)


class Manifold1D(nn.Module):
    def __init__(self) -> None:
        super(Manifold1D, self).__init__()

    def _manifold(self, x: torch.Tensor, pos: int = None):
        assert len(x.shape) == 2, 'x.shape must be 2.'
        l = x.shape[1]
        if pos > l:
            raise ValueError
        if not isinstance(pos, int):
            pos = random.randint(0, l - 1)
        return torch.cat((x[:, pos:], x[:, :pos]), dim=1)

    def forward(self, x, pos=None):
        if isinstance(pos, int):
            return self._manifold(x, pos)
        return self._manifold(x)


# if __name__ == "__main__":
#     from PIL import Image
#     import torchvision.transforms.functional as TF
#     pim = Image.open('lena.jpg')
#     tensor = TF.to_tensor(pim).unsqueeze(0)

#     m = Manifold2D()
#     tensor = m(tensor, pos=(0, 0))

#     pim = TF.to_pil_image(tensor.squeeze())
#     pim.save('lena_manifold.jpg')


if __name__ == "__main__":
    import torchvision.transforms.functional as TF
    from torchvision import models
    from PIL import Image
    from itertools import product
    import matplotlib.pyplot as plt
    import numpy as np
    import tqdm
    model = models.vgg11(pretrained=True)
    manifold2d = Manifold2D()
    manifold1d = Manifold1D()

    # for resnet
    # def features_extract(model, x):
    #     x = model.conv1(x)
    #     x = model.bn1(x)
    #     x = model.relu(x)
    #     x = model.maxpool(x)
    #     x = model.layer1(x)
    #     x = model.layer2(x)
    #     x = model.layer3(x)
    #     x = model.layer4(x)
    #     return x

    # def classifier(model, x):
    #     x = model.avgpool(x)
    #     x = model.fc(torch.flatten(x, 1))
    #     return x

    def features_extract(model, x):
        return model.features(x)

    def classifier(model, x):
        return model.classifier(x)

    def loadClassNames():
        classes_name = []
        with open('imagenet_labels.txt', 'r') as f:
            classes_name.extend(f.readlines())
        return classes_name

    im = Image.open('cock_manifold.jpg')
    classes_name = loadClassNames()
    # 224 x 224
    im = im.resize((224, 224))
    w, h = im.size
    im.show()
    tim = TF.to_tensor(im).unsqueeze(0)

    # NOTE If avgpool layer of resnet is output 1x1 feature map, then
    # Manifold2D layer will lose their function.
    # with torch.no_grad():
    #     feat = features_extract(model, tim)
    #     pred_cls_idx = classifier(model, torch.flatten(feat, 1)).argmax(1)
    #     print(classes_name[pred_cls_idx])

    #     feat_x, feat_y = feat.shape[2:]
    #     # pred_cls_idx - 当前图像的ground-truth值
    #     # cur_cls - 基于特征图位置的预测结果
    #     # spec_cls - 特定类的预测概率
    #     cur_cls = np.zeros((feat_x, feat_y))
    #     spec_cls = np.zeros((feat_x, feat_y))
    #     for i, j in product(range(feat_x), range(feat_y)):
    #         x = manifold2d(feat, pos=(i, j))
    #         y_hat = classifier(model, x)
    #         pred_cls = y_hat.argmax(1)
    #         cur_cls[i, j] = int(pred_cls.item())
    #         spec_cls[i, j] = torch.softmax(y_hat, dim=1)[0, pred_cls_idx].item()

    # NOTE Manifold 1D
    with torch.no_grad():
        pred_cls_idx = model(tim).argmax(1)
        print(classes_name[pred_cls_idx])

        feat = features_extract(model, tim)
        feat_x, feat_y = feat.shape[2:]
        # pred_cls_idx - 当前图像的ground-truth值
        # cur_cls - 基于特征图位置的预测结果
        # spec_cls - 特定类的预测概率
        cur_cls = np.zeros((feat_x * feat_y))
        spec_cls = np.zeros((feat_x * feat_y))
        for i in tqdm.tqdm(range(feat_x * feat_y), total=feat_x * feat_y):
            x = manifold1d(torch.flatten(feat, 1), pos=i)
            y_hat = classifier(model, x)
            pred_cls = y_hat.argmax(1)
            cur_cls[i] = int(pred_cls.item())
            spec_cls[i] = torch.softmax(y_hat, dim=1)[0, pred_cls_idx].item()

    # # plot results of Manifold2D transformation.
    # def plot(values, classnames=None):
    #     plt.ioff()

    #     plt.figure()
    #     plt.imshow(values, interpolation='nearest', cmap=plt.cm.Blues)
    #     # plt.colorbar()

    #     w, h = values.shape
    #     # for i, j in product(range(w), range(h)):
    #     #     text = f'{values[i, j]:.2f}'
    #     #     if classnames:
    #     #         text += f'\n{classnames[int(values[i,j])]}'
    #     #     plt.text(j, i, text, fontsize=1, horizontalalignment="center", verticalalignment="center")

    #     plt.tight_layout()
    #     plt.ylabel('Y')
    #     plt.xlabel('X')
    #     plt.savefig(f'{random.randint(0,100)}.jpg')

    # plot(cur_cls, classnames=classes_name)
    # plot(spec_cls)
