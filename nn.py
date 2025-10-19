import torch
from torch import nn

class Cj(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, input):
        output = input + 1
        return output

Cj = Cj()
x = torch.tensor([1, 2, 3])
y = Cj(x)
print(y)