import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# TODO: Feel free to try out your own images here by changing img_path

# load color image
bgr_img = cv2.imread('data/udacity_sdc.png')
# convert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32")/255

# plot image
plt.imshow(gray_img, cmap='gray')
plt.show()

## TODO: Feel free to modify the numbers here, to try out another filter!
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

print('Filter shape: ', filter_vals.shape)

# Defining four different filters,
# all of which are linear combinations of the `filter_vals` defined above

# define four filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# For an example, print out the values of filter 1
# print('Filter 1: \n', filter_1)
# print('Filter 2: \n', filter_2)
# print('Filter 3: \n', filter_3)
# print('Filter 4: \n', filter_4)

# visualize all four filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')


# Define a convolutional layer

# define a neural network with a single convolutional layer with four filters
class Net(nn.Module):

    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # assumes there are 4 grayscale filters, in_channels = 1 (gray color) , out_channels = 4 (filter layers)
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        # returns both layers
        return conv_x, activated_x


# instantiate the model and set the weights
# convert numpy List of [2D matrix], to torch of List of [3D matrix] with the 3rd-D value = 1 ~> List[[2D matrix]]
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
#print(weight)
model = Net(weight)

# print out the layer in the network
print(model)
# Net(
#   (conv): Conv2d(1, 4, kernel_size=(4, 4), stride=(1, 1), bias=False)
# )

# -------------------------------------------------------------------
# helper function for visualizing the output of a given layer
# default number of filters is 4
def viz_layer(layer, n_filters=4):
    fig = plt.figure(figsize=(20, 20))

    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i + 1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))

# -------------------------------------------------------------------

# visualize all filters
# fig = plt.figure(figsize=(12, 6))
# fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
# for i in range(4):
#     ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
#     ax.imshow(filters[i], cmap='gray')
#     ax.set_title('Filter %s' % str(i + 1))


# convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

# get the convolutional layer (pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)

#visualize the output of a conv layer
viz_layer(conv_layer)
plt.show()

viz_layer(activated_layer)
plt.show()