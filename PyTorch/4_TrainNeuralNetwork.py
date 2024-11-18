import torch
import my_helper

from torch import nn, optim
from torchvision import datasets, transforms


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)  # 1 - input

        # TODO: Training pass
        optimizer.zero_grad()

        output = model(images)  # 2 - forward & get output
        loss = criterion(output, labels)  # 3 - loss
        loss.backward()   # 4 - backward & get gradient
        optimizer.step()  # 5 - update the  weight based

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")


"""
Training loss: 1.9045912402270953
Training loss: 0.8632259076909978
Training loss: 0.5251576381483312
Training loss: 0.4263732662714366
Training loss: 0.3815245982458088

"""

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
my_helper.view_classify(img.view(1, 28, 28), ps)
