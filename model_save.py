import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
#1.torch.save() 不仅保存结构还保存参数
torch.save(vgg16,"vgg16_method1.pth")

#2.不保存结构，只保存参数，字典格式
torch.save(vgg16.state_dict(),"vgg16_method2.pth")