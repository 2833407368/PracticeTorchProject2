from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

writer = SummaryWriter("logs")

img = Image.open("hymenoptera_data/train/bees/16838648_415acd9e3f.jpg")
print(img)

# ToTensor
transform = transforms.ToTensor()
img_tensor = transform(img)
writer.add_image("toTensor",img_tensor,1)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize(
    [0.485,0.456,0.406],
    [0.229,0.224,0.225]
)
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("normalize",img_norm,1)


# resize
print(img.size)
writer.add_image("reSize",img_tensor,1)
trans_resize = transforms.Resize((224,224))
img_resize = trans_resize(img_tensor)
print(img_resize.size)
writer.add_image("reSize",img_resize,2)


#compose - resize -2
# trans_resize_2 = transforms.Resize((512,512))
# compose:操作序列 args：[transforms,transforms,...]
trans_compose = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
img_compose = trans_compose(img)
writer.add_image("compose",img_compose,1)


# 随机裁剪
#random crop
trans_randomCrop = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([transforms.RandomCrop(200),transforms.ToTensor()])
for i in range(10) :
    trans_compose_i = trans_compose_2(img)
    writer.add_image("randomCorp",trans_compose_i,i)



#




writer.close()