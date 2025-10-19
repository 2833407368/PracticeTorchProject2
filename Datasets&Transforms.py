import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./DAT")

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True, transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False, transform=dataset_transform)

img, target = test_set[0]
print(img)
print(target)

for i in range(10):
    # kk = train_set[i]
    img = train_set[i][0]
    writer.add_image("train_set", img, i)
for i in range(10):
    img = test_set[i][0]
    writer.add_image("test_set", img, i)
writer.close()


