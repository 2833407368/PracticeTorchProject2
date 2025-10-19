from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter("./logs")
img_path = "hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
writer.add_image("img",img_array, 1,dataformats="HWC")
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)
writer.close()