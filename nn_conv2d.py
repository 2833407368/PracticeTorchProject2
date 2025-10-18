import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter("./logs")

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)

class Cj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3,6,3,1,0)
    def forward(self, x):
        x = self.conv1(x)
        return x
cj = Cj()

step = 0
for data in dataLoader:
    imgs, targets = data
    output = cj(imgs)
    print("img.shape:{}".format(imgs.shape))
    print("output.shape:{}".format(output.shape))
    writer.add_images('input', imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('output', output, step)
    step += 1

writer.close()