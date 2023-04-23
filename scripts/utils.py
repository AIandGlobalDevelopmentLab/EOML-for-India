import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import scipy.misc
from PIL import Image
import os

image_extension = '.png'
# image_root = '/scratch/cse/dual/cs5150459/sat/villageimages_400_7sqkm_png/'
image_root = '/home/makkunda/sat/images/'
mean = np.array((107.23084313, 112.90604264, 108.86382704), dtype=np.float32)

def load_next_image(image_path):

    image_path=os.path.join(image_root,image_path)
    im = Image.open(image_path)
    im = im.convert('RGB')
    # im = scipy.misc.imresize(im, im_shape)
    # im = np.asarray(im, dtype=np.float32)

    # im = im[:, :, ::-1]  # swap channels
    # im -= mean
    # im = np.ascontiguousarray(im.transpose((2, 0, 1)))  # channel major

    return im

# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename)
    return img

# assumes data comes in batch form (ch, h, w)
def save_image(filename, data):
    img = data
    img=img.transpose((1,2,0))
    # print(img)
    img*=255
    img = (img).clip(0, 255).astype("uint8")
    # img=img[:,:,::-1]
    img = Image.fromarray(img)
    img.save(filename)

# using ImageNet values
def normalize_tensor_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_R2(y, y_):
#     y = y.data.cpu().numpy()
#     y_ = y_.data.cpu().numpy()
    y_mean = np.mean(y, axis=0)
    ss_pred = np.linalg.norm(y-y_, axis=0)
    ss_reg = np.linalg.norm(y-y_mean, axis=0)
    r_2 = ss_pred**2/ss_reg**2
    return 1-r_2
