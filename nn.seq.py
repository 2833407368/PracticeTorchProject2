from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter

from nn_conv2d import writer


class Cj(nn.Module):
    def __init__(self):
        super(Cj, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2,stride=1)
        # self.maxPool1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2,stride=1)
        # self.maxPool2 = nn.MaxPool2d(2)
        # self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2,stride=1)
        # self.maxPool3 = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(in_features=1024,out_features=64)
        # self.linear2 = nn.Linear(in_features=64,out_features=10)

        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2,stride=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2,stride=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2,stride=1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=1024,out_features=64),
            nn.Linear(in_features=64,out_features=10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxPool1(x)
        # x = self.conv2(x)
        # x = self.maxPool2(x)
        # x = self.conv3(x)
        # x = self.maxPool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        # return x
        return self.model(x)

cj = Cj()
print(cj)

input = torch.ones(64,3,32,32)
output = cj(input)
print(output.shape)

writer = SummaryWriter('./logs')
writer.add_graph(cj, input)
writer.close()
