import os
import numpy as np
import rasterio as rio
from affine import Affine
import pandas as pd
from skimage.transform import pyramid_gaussian
from PIL import Image
import sys

def makeName(s):
    return s.strip().lower().replace(' ','_')

class villageImage():
    def __init__(self,village,coords,images_dir='images',villages_loc_path='villages_batch.csv'):
        self.coords  = [float(x) for x in coords]
        images_dir   = os.path.join(os.pardir,'local',images_dir)
        villages_loc = pd.read_csv(os.path.join(os.pardir,'local',villages_loc_path),dtype=str)
        batch    = villages_loc.loc[villages_loc.village.eq(makeName(village)),'batch'].tolist()[0]
        self.msp_path = os.path.join(images_dir,'BMSP_{0}.tif'.format(batch))
        pan_path = os.path.join(images_dir,'BPAN_{0}.tif'.format(batch))
        self.pan=False
        if os.path.exists(pan_path):
            self.pan=True
            self.pan_path=pan_path

    def multispectral(self,dim):
        return One(self.coords,self.msp_path,dim)

    def panchromatic(self,dim):
        return One(self.coords,self.pan_path,dim)

    def combine(self,dim):
        half = int(dim/2)
        msp   = self.multispectral(half)
        pan   = self.panchromatic(half*2)
        return Combine(msp,pan)
    
class One():
    def __init__(self,coords,msp_path,dim):
        self.img      = rio.open(msp_path)
        self.final_meta = self.img.meta
        self.res      = self.img.res[0]
        self.coords  = coords
        self.dim=dim
        self.read_array()
        self.img.close()

    def read_array(self):
        # Read data in bbox defined by cords+-p
        x_c,y_c = self.coords
        x_ul,y_ul=x_c-0.5*self.dim*self.res,y_c+0.5*self.dim*self.res
        xoff,yoff = [int(round(x)) for x in ~self.img.transform*[x_ul,y_ul]]
        bbox = [xoff,yoff,xoff+self.dim,yoff+self.dim]
        self.array = self.img.read()[:,bbox[1]:bbox[3],bbox[0]:bbox[2]]
        self.transform = Affine(self.res,0,x_ul,0,-self.res,y_ul)

    def to_tif(self,dst_tif,bands=None):
        data = self.array
        b,h,w = data.shape
        if not bands:
            bands = [x for x in range(b)]
        meta  = self.final_meta.copy()
        meta.update({'transform':self.transform,
                        'height':h,
                         'width':w})

        with rio.open(dst_tif,'w',driver=meta['driver'],width=meta['width'],height=meta['height'],
                      transform=meta['transform'],crs=meta['crs'],dtype=meta['dtype'],count=len(bands)) as out:
            for i,j in enumerate(bands):
                out.write(data[j,:,:],i+1)
                
class Combine():
    def __init__(self,msp,pan):
        self.msp_array = msp.array
        self.pan_array = pan.array
        self.meta      = pan.final_meta
        self.transform = pan.transform

    def pyramid(self):
        pyramid = tuple(pyramid_gaussian(self.pan_array[0,:,:],max_layer=2,downscale=2,multichannel=False))
        noise = np.zeros(self.pan_array.shape[1:])
        for i,p in enumerate(pyramid):
            if i==0:
                continue
            local_noise = pyramid[i-1]-np.kron(pyramid[i],np.ones((2,2)))
            noise  = noise +  np.kron(local_noise,np.ones((2**(i-1),2**(i-1))))
        approx = np.kron(pyramid[len(pyramid)-1],np.ones((2**(len(pyramid)-1),2**(len(pyramid)-1))))
        return [noise,approx]

    def pansharpen(self,bands):
        if not bands:
            bands = [x for x in range(self.msp_array.shape[0])]
        inyection, approx = self.pyramid()
        msp_array = np.kron(self.msp_array[bands],np.ones((1,2,2),dtype=np.uint8))
        G =  msp_array / np.kron(approx,np.ones((self.msp_array[bands].shape[0],1,1)))
        result = msp_array+G*inyection
        return result.astype(np.uint8)

    def stack(self):
        return np.concatenate([np.kron(self.msp_array,np.ones((1,2,2),dtype=np.uint8)),self.pan_array],axis=0)

    def to_tif(self,dst_tif,what='stack',bands=None):
        if what=='stack':
            data = self.stack()
        elif what=='pansharpen':
            data = self.pansharpen(bands)
        else:
            print('Argument what must be stack or pansharpen')
        data = np.array(data,dtype=np.uint8)
        b,h,w = data.shape
        meta  = self.meta.copy()
        meta.update({'transform':self.transform,
                        'height':h,
                         'width':w})

        with rio.open(dst_tif,'w',driver=meta['driver'],width=meta['width'],height=meta['height'],
                      transform=meta['transform'],crs=meta['crs'],dtype=meta['dtype'],count=b) as out:
            for i in range(b):
                out.write(data[i,:,:],i+1)
                
in_file = sys.argv[1]

f = open(in_file,'r')

vil_id = []

for line in f :
    line = line.strip()
    vil_id.append(line.split('.')[0])
    
villages_batch=pd.read_csv('../local/villages_batch.csv', dtype=str,index_col=0,delimiter=',')

for i in range(len(vil_id)):
    x = villageImage(str(int(vil_id[i])),[villages_batch['lon'][int(vil_id[i])],villages_batch['lat'][int(vil_id[i])]])
    com = x.combine(224)
    com.stack()
    com.pansharpen([0,1,2])
    com.to_tif('../local/images_tif_2001/' + vil_id[i]  +'.tif','pansharpen',[0,1,2])
    
for i in range(len(vil_id)):
    im = Image.open('../local/images_tif_2001/' + vil_id[i]  +'.tif')
    im.save('../local/images_png_2001/' + vil_id[i]  +'.png')