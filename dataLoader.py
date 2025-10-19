import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./DL")
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True, transform=dataset_transform)
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False, transform=dataset_transform, download=True)

test_loader = DataLoader(test_data, batch_size=64,shuffle=False,num_workers=0,drop_last=False)
img, target = test_data[0]
print(img.shape)
print(target)


# torch.Size([4, 3, 32, 32]) batch_size=4 3通道 32*32
# tensor([9, 8, 0, 3])  target 9 8 0 3
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        print("------")
        print(imgs.shape)
        print(targets)
        print("------")
        # 注意writer.add_images和writer.add_image 差个s
        writer.add_images("epoc:{}".format(epoch), imgs, step)
        step += 1
writer.close()

