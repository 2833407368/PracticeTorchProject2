import torch
from torch import nn


class Cj(torch.nn.Module):
    def __init__(self):
        super(Cj, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2, stride=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2, stride=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2, stride=1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    cj = Cj()
    # 64张，3通道，32*32尺寸
    input = torch.ones(64, 3, 32, 32)
    output = cj(input)
    print(output.shape)
