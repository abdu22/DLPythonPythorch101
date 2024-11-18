from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # x is input data, image
        # Pass the input tensor through each of our operations
        x = self.hidden(x)  # x running through W1x + b1 & it auto create W1,b1
        x = self.sigmoid(x)  # x became h
        # x = F.sigmoid(x)
        x = self.output(x)  # x running through W2x + b2 & get output
        x = self.softmax(x)  # x became probability
        # x = F.softmax(x, dim=1)


        return x


