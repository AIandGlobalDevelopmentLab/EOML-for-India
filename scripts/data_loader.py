import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import os
from utils import load_next_image
from PIL import Image
import torch

class CensusGoogleSatellite(Dataset):
    def __init__(self, filename, transform=None,mode='train',image_root=""):
        self.transform = transform
        self.image_root = image_root
        census = pd.read_csv(filename, index_col=0)
        self.features = ['rooms-under-3', 'household-size-under-5', 'water-treated', 'water-untreated', 'water-natural','electric-like', 'oil-like', 'electronics','has-phone', 'transport-cycle', 'transport-motorized', 'no-assets', 'banking-services-availability', 'cook-fuel-processed', 'bathroom-within', 'permanent-house']
        self.census = census[self.features].values
        self.villageids = census.index.values
        self.mode = mode
        length=len(self.villageids)
        
        if(mode=='train'):
            start=int(0*length)
            end=int(0.9*length)
        elif(mode=='val'):
            start=int(0.9*length)
            end=int(length)
        else:
            start=int(0*length)
            end=int(length)
            
        self.census=self.census[start:end]
        self.villageids=self.villageids[start:end]

    def __len__(self):
        return self.census.shape[0]

    def __getitem__(self, idx):

        y = self.census[idx]
        y = torch.from_numpy(y)

        village_code = self.villageids[idx]
        village_code='{:016}'.format(village_code)
        image_path = os.path.join(self.image_root, village_code+".png")
#         x = load_next_image(image_path)
        im = Image.open(image_path)
        x = im.convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x, y
