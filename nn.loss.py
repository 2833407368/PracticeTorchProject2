import torch

input = torch.tensor([[1,2,3]],dtype=torch.float32)
target = torch.tensor([[1,2,5]],dtype=torch.float32)

inputs = torch.reshape(input,(1,1,1,3))
targets = torch.reshape(target,(1,1,1,3))

loss = torch.nn.L1Loss()
result = loss(inputs,targets)
print(result)

loss2 = torch.nn.MSELoss()
print(loss2(inputs,targets))

x = torch.tensor([[0.1,0.2,0.3]])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = torch.nn.CrossEntropyLoss()
print(loss_cross(x,y))