import matplotlib.pyplot as plt
import my_helper as helper
import torch

from torchvision import datasets, transforms

# .Compose is a pipeline
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder('~/.pytorch/Cat_Dog_data/train/', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)
plt.show()