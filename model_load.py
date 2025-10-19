#1
import torch
import torchvision

model = torch.load("vgg16_method1.pth")
print(model)

#2 先自己创建模型，再导入参数
model2 = torchvision.models.vgg16(pretrained=False)
model2.load_state_dict(torch.load("vgg16_method1.pth"))
# model2 = torch.load("vgg16_method2.pth")
print(model2)

#trap
class Cj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,64,3)
    def forward(self, x):
        x = self.conv1(x)
        return x
cj = Cj()
torch.save(cj,"cj.pth")