from nn_module_with_relu import Network as ReLUNetwork
from nn_module_with_sigmoid import Network as SigmoidNetwork
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import numpy as np
import my_helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)
flat_images = images.view(images.shape[0], -1)
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.show()


print("---------SigmoidNetwork----------------")
model = SigmoidNetwork()
print(model)
"""
Network(
  (hidden): Linear(in_features=784, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
  (sigmoid): Sigmoid()
  (softmax): Softmax(dim=1)
)
"""

inputs = flat_images

out  = model.forward(inputs)
print("--------OUT-------------------------")
print("probability : ", out)
print("probability sum row : " , out.sum(dim=1))
print("------------------------------------")

# print("---------ReLUNetwork----------------")
# model = ReLUNetwork()
# print(model)
# """
# Network(
#   (fc1): Linear(in_features=784, out_features=128, bias=True)
#   (fc2): Linear(in_features=128, out_features=64, bias=True)
#   (fc3): Linear(in_features=64, out_features=10, bias=True)
# )
# """
# inputs = flat_images
#
# out  = model.forward(inputs)
# print("--------OUT-------------------------")
# print("probability : ", out)
# print("probability sum row : " , out.sum(dim=1))


# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
my_helper.view_classify(img.view(1, 28, 28), ps)


