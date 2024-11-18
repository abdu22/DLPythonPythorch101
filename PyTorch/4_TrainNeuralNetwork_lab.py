import torch

from torch import nn, optim
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)


# Define the loss
"""
1) nn.CrossEntropyLoss() 
this criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class

we need to pass the raw output of our network to the loss, not the output of the softmax function
this raw output is called = logits

2) NLLLoss = Negative log likelihood Loss
"""

# 1) Using the CrossEntropyLoss

# # Build a feed-forward network model
# model = nn.Sequential(nn.Linear(784, 128),
#                       nn.ReLU(),
#                       nn.Linear(128, 64),
#                       nn.ReLU(),
#                       nn.Linear(64, 10))
#
# # this criterion combines nn.LogSoftmax() and nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
#
# # Flatten images
# images = images.view(images.shape[0], -1)
# # Forward pass, get our logits
# logits = model(images)
# # Calculate the loss with the logits and the labels
# loss = criterion(logits, labels)
# print("--------OUT-------------------------")
# print(loss)
# # tensor(2.2763, grad_fn=<NllLossBackward0>)
print("------------------------------------")

# 2) Using the NLLLossq


# Build a feed-forward network model
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# we have to add nn.LogSoftmax() network in the model
criterion = nn.NLLLoss()
# Flatten images
images = images.view(images.shape[0], -1)
# forward pass, get log-probability
logps = model(images)
# calculate the loss
loss = criterion(logps, labels)
print("--------OUT-------------------------")
print(loss)
# tensor(2.3375, grad_fn=<NllLossBackward0>)
print("------------------------------------")


print('Initial weight backward pass: \n', model[0].weight)
print('Grad Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('Grad After backward pass: \n', model[0].weight.grad)

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)

"""
--------OUT-------------------------
tensor(2.2799, grad_fn=<NllLossBackward0>)
------------------------------------
Initial weight backward pass: 
 Parameter containing:
tensor([[-0.0167,  0.0139, -0.0080,  ...,  0.0209, -0.0245, -0.0071],
        [-0.0277,  0.0008,  0.0163,  ..., -0.0341, -0.0062,  0.0307],
        [-0.0183,  0.0348, -0.0108,  ..., -0.0126,  0.0343,  0.0025],
        ...,
        [ 0.0260, -0.0048,  0.0212,  ..., -0.0334, -0.0127, -0.0298],
        [ 0.0194, -0.0096, -0.0218,  ...,  0.0149, -0.0319, -0.0272],
        [ 0.0227, -0.0106, -0.0162,  ...,  0.0228, -0.0086,  0.0029]],
       requires_grad=True)
Grad Before backward pass: 
 None
Grad After backward pass: 
 tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0051,  0.0051,  0.0051,  ...,  0.0051,  0.0051,  0.0051],
        [-0.0020, -0.0020, -0.0020,  ..., -0.0020, -0.0020, -0.0020],
        ...,
        [-0.0004, -0.0004, -0.0004,  ..., -0.0004, -0.0004, -0.0004],
        [ 0.0002,  0.0002,  0.0002,  ...,  0.0002,  0.0002,  0.0002],
        [ 0.0007,  0.0007,  0.0007,  ...,  0.0007,  0.0007,  0.0007]])
Updated weights -  Parameter containing:
tensor([[-0.0167,  0.0139, -0.0080,  ...,  0.0209, -0.0245, -0.0071],
        [-0.0277,  0.0008,  0.0163,  ..., -0.0341, -0.0062,  0.0307],
        [-0.0183,  0.0348, -0.0108,  ..., -0.0126,  0.0343,  0.0025],
        ...,
        [ 0.0260, -0.0048,  0.0212,  ..., -0.0334, -0.0127, -0.0298],
        [ 0.0194, -0.0096, -0.0218,  ...,  0.0149, -0.0319, -0.0272],
        [ 0.0227, -0.0106, -0.0162,  ...,  0.0228, -0.0086,  0.0029]],
       requires_grad=True)
"""
