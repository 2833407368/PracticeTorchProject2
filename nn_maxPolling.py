import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

writer = SummaryWriter("./logs")

imput = torch.tensor([
    [1,2,0,3,1],
    [0,1,2,3,1],
    [1,2,1,0,0],
    [5,2,3,1,1],
    [2,1,0,1,1]
],dtype =torch.float )
imput = torch.reshape(imput, (-1,1,5,5))
print(imput.shape)

class Cj(nn.Module):
    def __init__(self):
        super(Cj, self).__init__()
        self.maxpool1 = nn.MaxPool2d(3, 2,ceil_mode=True)
    def forward(self, input):
        input = self.maxpool1(input)
        return input

cj = Cj()
output = cj(imput)

step=0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output = cj(imgs)
    writer.add_images("output",output,step)
    print("input.shape:{}".format(imgs.shape))
    print("output.shape:{}".format(output.shape))
    step += 1
writer.close()