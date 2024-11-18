import torch
import my_helper
import matplotlib.pyplot as plt
from Classifier import Classifier

from torch import nn, optim
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

print("Total number of images : ", len(trainloader))
image, label = next(iter(trainloader))
# my_helper.imshow(image[0,:]);
plt.imshow(image[1].numpy().squeeze(), cmap='Greys_r')
plt.show()

"""
Each image is 28 X 28 = 784 pixels 
10 classes
Minimum one hidden layer ( we can add if we want)
Suggest to use ReLU activations for the layers
and return logits or log-softmax from the forward pass.
"""

"""
1st step : Define network architecture.
"""

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)

# model = Classifier()

"""
2nd Step : Define Criterion. Can use ADAM or SGD
"""
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")


"""
Total number of images :  938
Training loss: 2.1427552107808943
Training loss: 1.269015831161918
Training loss: 0.8032569312401163
Training loss: 0.6778372901080768
Training loss: 0.6221482988867932
Training loss: 0.5839389772621045
Training loss: 0.5540313319420256
Training loss: 0.5281756367288164
Training loss: 0.5072543750058359
Training loss: 0.48953968440609447
"""


images, labels = next(iter(testloader))
img = images[0].view(1, 784)
# Turn off gradients to speed up this part
# with torch.no_grad():
#     logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(model(img))
my_helper.view_classify(img.view(1, 28, 28), ps,  version='Fashion')