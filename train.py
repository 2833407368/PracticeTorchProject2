import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Cj

writer = SummaryWriter("./logs")

#下载数据集
train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())

#看有多少图片
train_data_size = len(train_data)
test_data_size = len(test_data)
# print("train_data_size:{}".format(train_data_size))
print(f"train_data_size:{train_data_size}")
print("test_data_size:{}".format(test_data_size))

#加载数据
train_dataLoader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataLoader = DataLoader(test_data, batch_size=64, shuffle=True)

#创建模型
cj = Cj().cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss().cuda()

#优化器
lr = 1e-2
optimizer = torch.optim.SGD(cj.parameters(), lr)

#设置训练
total_train_step = 0
total_test_step = 0
epoch = 10


for i  in range(epoch):
    print(f"-----epoch {i}-----")
    for data in train_dataLoader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()

        outputs = cj(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"total_train_step:{total_train_step},loss:{loss}")
            writer.add_scalar("train_loss", loss, total_train_step)

    #每轮测试
    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in test_dataLoader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()

            outputs = cj(imgs)
            loss = loss_fn(outputs, targets)
            total_test_step += 1
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f"epoch:{epoch},total_test_step:{total_test_step},loss:{total_test_loss}")
    print(f"total_accuracy:{total_accuracy/total_test_step}")
    writer.add_scalar("test_loss", loss, total_test_step)
    writer.add_scalar("test_accuracy", accuracy, total_test_step)
    total_test_loss += 1

    #保存每轮训练后的模型
    # torch.save(cj,"cj_{}.pth".format(epoch))
    print("保存了")

writer.close()