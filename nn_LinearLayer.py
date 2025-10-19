import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
dataLoader = DataLoader(dataset, batch_size=64,shuffle=False,num_workers=0,drop_last=True)

class Cj(nn.Module):
    def __init__(self):
        super(Cj, self).__init__()
        self.linear = nn.Linear(196608,10)
    def forward(self, x):
        return self.linear(x)
cj = Cj()

for data in dataLoader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(cj(output).shape)

