import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)
writer = SummaryWriter("./logs")

class Cj(nn.Module):
    def __init__(self):
        super(Cj, self).__init__()
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
        return self.model(x)

cj = Cj()
cj = cj.cuda()

loss = nn.CrossEntropyLoss()
loss = loss.cuda()

optim = torch.optim.SGD(cj.parameters(), lr=0.01)
step = 0

for epoch in range(20):
    print('epoch:{}'.format(epoch))
    running_loss = 0.0
    for data in dataLoader:
        imgs, targets = data

        imgs = imgs.cuda()
        targets = targets.cuda()

        output = cj(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss.item()
    print("epoch:{}, loss:{}".format(epoch, running_loss))
