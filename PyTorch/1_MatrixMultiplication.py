import torch


def activation(x):
    """
   sigmoid activation function

   :param x:  tourch.tensor
   :return: sigmoid function
   """
    return 1 / (1 + torch.exp(-x))


"""
features : tensor([[0.5349, 0.1988, 0.6592]])
n_input = 3
n_hidden = 2
n_output = 1

features :  tensor([[0.5349, 0.1988, 0.6592]])

w1 :  tensor([[0.6569, 0.2328],
              [0.4251, 0.2071],
              [0.6297, 0.3653]])
              
w2 :  tensor([[0.8513],
              [0.8549]])
              
B1 :  tensor([[0.5509, 0.2868]])

B2 :  tensor([[0.2063]])


mm_PLUS_bias :  tensor([[1.4020, 0.6933]])
-------------------------------------------
new_feature :  tensor([[0.8025, 0.6667]])
-------------------------------------------
output :  tensor([[0.8114]])
-------------------------------------------
            

"""
torch.manual_seed(7)
features = torch.randn(1, 3)

print("features : ", features)
n_input = features.shape[1]  # = 3
n_hidden = 2
n_output = 1

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)
print("w1 : ", W1)
print("w2 : ", W2)

B1 = torch.randn(1, n_hidden)
B2 = torch.randn(1, n_output)
print("B1 : ", B1)
print("B2 : ", B2)

mm_PLUS_bias = torch.mm(features, W1) + B1
print("mm_PLUS_bias : ", mm_PLUS_bias)

print("-------------------------------------------")
h = activation(mm_PLUS_bias)
print("new_feature : ", h)

print("-------------------------------------------")
output = activation(torch.mm(h, W2) + B2)
print("output : ", output)
print("-------------------------------------------")
