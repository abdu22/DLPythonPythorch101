import torch


def activation(x):
   """
   sigmoid activation function

   :param x:  tourch.tensor
   :return: sigmoid function
   """
   return 1 / (1 + torch.exp(-x))


"""
torch.randn()
"""
torch.manual_seed(7)
features = torch.randn((1, 5))
weights = torch.rand_like(features)
bias = torch.randn((1,1))

print(features)
print(weights)
print(bias)

"""
feature = [[-0.1468,  0.7861,  0.9468, -1.1143,  1.6908]]
weight  = [[0.2868, 0.2063, 0.4451, 0.3593, 0.7204]]
bias    = [[-0.8948]]
"""

print("----------concat the bias as node--------")

# converting the bias as its own weight with feature = 1
weights2 = torch.cat((weights, bias), dim=1)
features2 = torch.cat((features, torch.tensor([[1]])), dim=1)
print(weights2)
print(features2)
print(features2.shape)
print(features2.shape)

features2_DOT_weights2 = (features2 * weights2)

print("Y : ", features2_DOT_weights2)

"""
torch.sum()
torch.shape
"""
prediction = activation(torch.sum(features2_DOT_weights2))
print(prediction)



print("-----------add bias at the end------------")
# wight * features + b
print("Y : ", activation(torch.sum(features*weights) + bias))
print(features.shape)
print(weights.shape)

"""
if we use torch.mm(features2, weights2), it will throw exception

try it 
mm = torch.mm(features2 * weights2)
print(activation(mm))

b/c we can't do mm with [1 X 5] && [1 X 5], but it has to be b/n [1 X 5] && [5 X 1]

has to be like 
[[-0.1468,  0.7861,  0.9468, -1.1143,  1.6908]] X  [[0.2868], 
                                                   [0.2868],
                                                   [0.2063],
                                                   [0.4451],
                                                   [0.3593],
                                                   [0.7204]]

"""
print("-----------re-shape weight and matrix multiplication------------")
"""
torch.mm()
torch.reshape(5,1)   ==> return new tensor with same data sometimes and sometimes clone data 
torch.resize_(5,1)   ==> return same tensor, perform in-place
torch.view(5,1)   ==> return new tensor (new shape) but with same data, (without messing the data in memory)
"""
weights = weights.view(5,1)
print(weights.shape)
mm  = activation(torch.sum( torch.mm(features, weights) + bias))
print("Y : ", mm)







    
    

# x = torch.rand(5, 3)
# print(x)