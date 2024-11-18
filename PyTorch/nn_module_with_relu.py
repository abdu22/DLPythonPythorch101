import torch.nn.functional as F
from torch import nn

"""
Exercise: Create a network with 784 input units, 
a hidden layer with 128 units and a ReLU activation, 
then a hidden layer with 64 units and a ReLU activation, 
and finally an output layer with a softmax activation as shown above. 
You can use a ReLU activation with the nn.ReLU module or F.relu function.
"""


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # fc stands for fully connected
        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x
