# Import necessary packages

import numpy as np
import torch
import torchvision.transforms as transforms

import helper

import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

print("Total number of images : ", len(trainloader))  # Total number of images :  938

dataiter = iter(trainloader)  # will batch the total 938, in to 64 batches. Meaning for each iter, 64 images
images, labels = next(dataiter)
# print(type(images)) # <class 'torch.Tensor'> == a tensor class
# print(images.shape) # torch.Size([64, 1, 28, 28])  == 64 images, each 25 X 25 pixedl
# print(labels.shape)
# print(images[1].numpy()) # an image's pixel value in 2D vector with 28 X 28 size
# print(images[1].shape) # for an image, torch.Size([1, 28, 28])
# print([labels[1]]) # the label of the image, eg : 7'

flat_images = images.view(images.shape[0], -1)
# print(flat_images.shape) # torch.Size([64, 784])
# print(flat_images[0].numpy())

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.show()

"""
Build a multi layer neural network, with 782 input units, 256 hidden units, 10 outputs(0 to 9)

image = [64, 1, 28, 28]
input = flat_image = [64, 784]

input = 64 images, each represented by 784 1D vector 
1 : [1,2,3...784] 
2 : [1,2,3...784]
3 : [1,2,3...784]
:
64 :[1,2,3...784]

W1 = 784 X 256 
b1 = 1 X 257
W2 = 256 X 10
b2 = 1 X 10

h shape :  torch.Size([64, 256])
out shape :  torch.Size([64, 10])

"""


def activation(x):
    return 1 / (1 + torch.exp(-x))


torch.manual_seed(7)
inputs = flat_images  # first sample data , flat_images[0]

n_input = inputs.shape[1]  # 784
n_hidden = 256
n_output = 10

W1 = torch.randn(n_input, n_hidden)
B1 = torch.randn(1, n_hidden)

W2 = torch.randn(n_hidden, n_output)
B2 = torch.randn(1, n_output)

h = activation(torch.mm(inputs, W1) + B1)
out = activation(torch.mm(h, W2) + B2)
print("--------OUT-------------------------")
print("h shape : ", h.shape)
print("out shape : ", out.shape)
# print(out)
print("------------------------------------")

"""
x = 64 X 10 
 [[0,2,3,4. . . .9]
  [0,2,3,4. . . .9]
  :
  :
  [0,2,3,4. . . .9]]
  
do softmax for each row
exp_sum = 62, 1
[
[]
[]
:
[]
] 
---------
dim=1, takes the sum across the colum
dim=0, takes th sum across the row.
"""


def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)


print("--------Probablity-------------------------")
probabilities = softmax(out)
print(probabilities.shape)
print("probability : ", probabilities)
print("----------sum of probabilities per row results 1. For each row..... --------------------")
print("probability : ", probabilities.sum(dim=1))
print("-----------------------------------------")
