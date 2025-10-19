import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

input = torch.tensor([
    [1,-0.5],
    [-1,3]
],dtype =torch.float )

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

class Cj(nn.Module):
    def __init__(self):
        super(Cj, self).__init__()
        # self.relu = nn.ReLU(False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        # return self.relu(input)
        return self.sigmoid(input)
cj = Cj()
print(cj(input))

step = 0
writer = SummaryWriter("./logs")
for data in dataloader:
    print(step,data)
    imgs, targets = data
    writer.add_images('input', imgs, step)
    output = cj(imgs)
    writer.add_images('output', output, step)
writer.close()