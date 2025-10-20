import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Cj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("./logs/SGD")

# 数据集
train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())

print(f"train_data_size:{len(train_data)}")
print(f"test_data_size:{len(test_data)}")

# 数据加载
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 模型
cj = Cj().to(device)

# 损失函数  优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cj.parameters(), lr=0.0001)
# optimizer = torch.optim.Adam(cj.parameters(), lr=0.0001)

total_train_step = 0
epoch = 100

for i in range(epoch):
    print(f"----- Epoch {i+1}/{epoch} -----")

    # 训练
    cj.train()
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = cj(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"Train Step: {total_train_step}, Loss: {loss.item():.4f}")
            writer.add_scalar("Train/Loss", loss.item(), total_train_step)

    # 测试
    cj.eval()
    total_test_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = cj(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == targets).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    avg_accuracy = total_accuracy / len(test_data)

    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    writer.add_scalar("Test/Loss", avg_test_loss, i+1)
    writer.add_scalar("Test/Accuracy", avg_accuracy, i+1)

    if i % 10 == 0:
        torch.save(cj.state_dict(), f"cj_SGD_epoch_{i+1}.pth")
        print("模型已保存\n")

writer.close()
