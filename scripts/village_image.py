# -*- coding: utf-8 -*-
"""
This script provides a class that retrieves the imagery from one village from the tile where it is located (see scripts d0_assingBatch.py and d1_downloadBatches.py for details on the download process).
The main class villageImage, provides a variety of methods to retrieve imagery from the imagery (see demonstration at the end of the script). The village name and coords as written in the villages_loc.csv file
(created during the downloading) must be provided, as well as the directory with the imagery.

Author:        Felipe Jordan
Date created:  04/08/2020
Last modified: 04/08/2020
"""

import os
import numpy as np
import rasterio as rio
from download_batch import makeName
from affine import Affine
import pandas as pd
from skimage.transform import pyramid_gaussian

class villageImage():
    def __init__(self,village,coords,images_dir='images',villages_loc_path='villages_loc.csv'):
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


'''
Demonstration
'''
if False:
    # Select a village for the demonstration
    villages_batch=pd.read_csv('../local/villages_batch.csv', dtype=str)
    v = villages_batch.loc[villages_batch.batch.eq('1')].loc[213554]

    # Instance of class makeImage, requires village name and village lon and lat in list, everyting should be a string
    x = villageImage(v.village,[v.lon,v.lat])

    # The multispectral method returns an instance of the class One initialized with the multispectral bands. The size argument, that is the size of the desired height and width of the image in pixels, is required to initialize the class.
    ms = x.multispectral(112)
    # When this instance is initialize, it saves the multispectral bands in the box around the village coordinates given by the size in a numpy array, which can be acceded with the attribute array
    ms.array
    # The to_tif(path) method saves the array as a tif image
    ms.to_tif('../local/examples/msp_allBands.tif')
    # You can select only a subsample of the bands with the second argument, provided as a list with the band number (starting in zero)
    ms.to_tif('../local/examples/msp_rgb.tif',[0,1,2])

    # The panchromatic method returns an instance of the class One initialized with the multispectral bands. The size argument, that is the size of the desired height and width of the image in pixels, is required to initialize the class.
    pan = x.panchromatic(224)
    # When this instance is initialize, it saves the panchromatic bands in the box around the village coordinates given by the size in a numpy array, which can be acceded with the attribute array
    pan.array
    # The to_tif(path) method saves the array as a tif image
    pan.to_tif('../local/examples/pan_allBands.tif')

    # The combine method combines the multispectral and panchromatic bands. It must be initialized with the size of the final image.
    com = x.combine(224)
    # The stack method will stack the multispectral and panchromatic band, expanding the former to match the pancromatic size using a the kroneger product of the multispectral imagery and np.ones((1,2,2)). It returns the array.
    com.stack()
    # The panchromatic method injects the details of the panchromatic band into the selected bands of the multispectral imagery. Returns the array.
    com.pansharpen([0,1,2])
    # The to_tif method saves a stacked or pansharpen image to a tif file. The what argument determines whether it is the former or the later ('stack' for the former, 'pansharpen' for the later). The bands argument can be provided to select bands of the multispectral bands to be passed to the pansharpen method.
    com.to_tif('../local/examples/com_stack.tif')
    com.to_tif('../local/examples/com_pansharpen_rgb.tif','pansharpen',[0,1,2])

