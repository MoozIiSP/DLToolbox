import sys, os
sys.path.append(os.path.join(os.environ['HOME'], 'GitRepos/DLToolbox/third_party/vision/references/detection'))

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datasets.custom import SpineDetection
import transforms as T
from engine import train_one_epoch, evaluate
import utils

from PIL import ImageDraw, ImageColor, Image
import numpy as np


num_classes = 3

# finetuning
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# # load a model pre-trained pre-trained on COCO
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#
# # replace the classifier with a new one, that has
# # num_classes which is user-defined
# num_classes = 2  # 1 class (person) + background
# # get number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# # replace the pre-trained head with a new one
# model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes)
#
# # 2 - Modifying the model to add a different backbone
# # load a pre-trained model for classification and return
# # only the features
# backbone = torchvision.models.vgg11(pretrained=True).features
# # FasterRCNN needs to know the number of
# # output channels in a backbone. For mobilenet_v2, it's 1280
# # so we need to add it here
# backbone.out_channels = 512
#
# # let's make the RPN generate 5 x 3 anchors per spatial
# # location, with 5 different sizes and 3 different aspect
# # ratios. We have a Tuple[Tuple[int]] because each feature
# # map could potentially have different sizes and
# # aspect ratios
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512), ),
#                                    aspect_ratios=((0.5, 1.0, 1.5), ))
#
# # let's define what are the feature maps that we will
# # use to perform the region of interest cropping, as well as
# # the size of the crop after rescaling.
# # if your backbone returns a Tensor, featmap_names is expected to
# # be [0]. More generally, the backbone should return an
# # OrderedDict[Tensor], and in featmap_names you can choose which
# # feature maps to use.
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
#                                                 output_size=7,
#                                                 sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = get_model_instance_segmentation(num_classes)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

# our dataset has three classes only - background, normal, abnormal
# use our dataset and defined transformations

data_root = os.path.join(os.environ['HOME'], 'GitRepos/yolact/data/custom/')
trainset = SpineDetection(
    data_root + 'train',
    data_root + 'train/annotations.json',
    transforms=get_transform(train=False)
)
validset = SpineDetection(
    data_root + 'valid',
    data_root + 'valid/annotations.json',
    transforms=get_transform(train=False)
)

# split the dataset in train and test set
indices = torch.randperm(len(trainset)).tolist()
dataset = torch.utils.data.Subset(trainset, indices[:-50])
dataset_test = torch.utils.data.Subset(validset, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=2,
                                               shuffle=False,
                                               num_workers=4,
                                               collate_fn=utils.collate_fn)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,
                            lr=0.005,
                            momentum=0.9,
                            weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # FIXME evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)

print("That's it!")

# %%
# Evaluation
# pick one image from the test set

classes = {0: 'background',
           1: 'normal',
           2: 'abnormal'}

img, _ = validset[np.random.randint(len(validset))]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
# prediction

im_with_boxes = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
# FIXME if you use byte method, the result will be abnormal.
boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()
draw = ImageDraw.Draw(im_with_boxes)
for [x1, y1, x2, y2] in boxes:
    draw.rectangle([x1, y1, x2, y2], outline=ImageColor.getrgb('#FF0000'))
    draw.text([x1, y1], f'{classes[labels[0]]} % {scores[0]:.2f}')
im_with_boxes.save('output.png')