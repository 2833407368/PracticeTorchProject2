from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2


img_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("tensor_img",tensor_img)

writer.close()
