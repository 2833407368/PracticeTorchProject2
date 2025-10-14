import torchvision
import six

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,transform=transform)
