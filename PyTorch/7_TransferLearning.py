import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

"""
Most of the pretrained models require the input to be 224x224 images. 
Also, we will need to match the normalization used when the models were trained. '
         'Each color channel was normalized separately, the means are [0.485, 0.456, 0.406] '
         'and the standard deviations are [0.229, 0.224, 0.225].)
         
Normalization ? 
 - in this context refers to transforming the image pixel values to a common scale, typically with a mean of 0 and a standard deviation of 1, which helps the model learn more effectively.
(Red, Green, and Blue)         
Mean: [0.485, 0.456, 0.406]    => These are the mean values for each channel (Red, Green, and Blue) in an RGB image.      
Standard Deviation (std): [0.229, 0.224, 0.225] => This means the pixel values in the Red channel have a standard deviation of 0.229, Green has 0.224, and Blue has 0.225.
"""

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder('~/.pytorch/Cat_Dog_data/train/', transform=train_transforms)
test_data = datasets.ImageFolder('~/.pytorch/Cat_Dog_data/test/', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# We can load in a model such as DenseNet
model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier