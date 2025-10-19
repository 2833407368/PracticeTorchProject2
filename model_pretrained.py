import torch
import torchvision
from torchvision.models import vgg16

# train_data = torchvision.datasets.ImageNet(root='./dataset', split='train', transform=torchvision.transforms.Compose([]),download=True)
vgg16_true = torchvision.models.vgg16(pretrained=True).cuda()
train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True,)

vgg16_true.add_module("add_linear",torch.nn.Linear(1000,10))
# vgg16_true.classifier.add_module("add_linear",torch.nn.Linear(1000,10))
vgg16_true.classifier[6] = torch.nn.Linear(4096,10)