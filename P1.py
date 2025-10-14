import torchvision
import six

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True,transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False,transform=transform)
