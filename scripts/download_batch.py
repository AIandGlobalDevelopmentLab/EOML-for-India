'''
---- DOWNLOAD IMAGERY FOR BATCH OF VILLAGES ----
This file downloads a satellite imagery that contain a batch of villages. The batch must be a group a villages
that are close to each other, ideally within the same intersection of WRS2 path-row polygons.
For a batch-year, the script collect all the available imagery and creates a quality mozaic by prioritizing imagery with higher average levels of
NDVI. The results are saved to a Google Drive folder.

Positional argumentes:
    - folder:   Name of folder on user's google drive where images will be saved.
    - year:     Year for which satellite images are collected to create quality mozaic.
    - sensor:   Sensor used (L5 for landsat 5, L7 for landsat 7)
    - batch:    Number of the batch

Optional arguments:
    - size: Box with width and height around each village, in pixels at the scale of the imagery. Must be even number (default 256).
            When a panchromatic band is available (Landsat 7), the scale of the panchromatic band (15m) is used. Otherwise (Landsat 5), the scale of the multispectral bands is used (30m).

Output:
    - Drive/{folder}/BMSP_{batch}.tif: Image with the multispectral bands at a 30m resolution, with soil-reflectance bands as uInt8.
    - Drive/{folder}/BPAN_{batch}.tif: If panchromatic band is available, tif image with the panchromatic band at a 15m resolution, with top-of-the-atmosphere reflectance band as uInt8.
    - local/villages_loc.csv:          CSV file with village names and batch number for future reference. Script will append new information to existing file if available.
    - log.csv                          CSV file showing time spend in each step of the script. Script will append new information to existing file if available.

Note: To successfully run this file, run the script assign_batch.py to create the villages_batch.csv file that this script uses.

Author:        Felipe Jordan
Date created:  12/16/2019
Last modified: 04/08/2020
'''

# Import the Earth Engine Python Package:

import ee

# Initializa ee library
ee.Initialize()

########################## Helper functions and main class ---------
def keepClear(region,sat):
    '''
    Initializes a function that is passed to the map method in GEE
    The child function takes an image as an argument, and mask all pixels that either:
    - The FMASK QA band assigns to it a medium or high confidence of cloud
    - The FMASK QA band assigns to it a medium or high confidence of cloud shadow
    - The saturation QA band determines that at least one band is saturated
    Finally, the function also adds to the image as a property the fraction of valid pixels within the region of interest,
    which is passed to the function that intializes the child function
    '''
    def keepClear_child(image):
        # Select FMASK QA band and select bits associated to clouds and clouds cover
        im  = ee.Image(image)
        qa  = im.select('pixel_qa')
        qa_clouds            = extractQABits(qa,6,7)
        qa_cloudsShadows     = extractQABits(qa,3,3)
        # Select Saturation QA band and select bite associated to saturation
        qa2 = im.select('radsat_qa')
        qa_saturation        = extractQABits(qa2,0,0)
        # Create mask where valid pixels have low confidence of clound and clound shadow and are not saturated
        mask                 = qa_clouds.lte(1).And(qa_cloudsShadows.eq(0)).And(qa_saturation.eq(0))
        if sat=='L8':
            # Cirrus clounds and terrain oclussion in landsat 8
            qa_cirrus  = extractQABits(qa,8,9)
            qa_terrain = extractQABits(qa,10,10)
            mask       = mask.And(qa_cirrus.lte(1)).And(qa_terrain.eq(0))
        # Claculate fraction of valid pixels in the region and return image with QI property with fraction of valid pixels
        valid = mask.reduceRegion(ee.Reducer.sum(),region).get('pixel_qa')
        tot   = mask.reduceRegion(ee.Reducer.count(),region).get('pixel_qa')
        return im.updateMask(mask).copyProperties(im).set({'QI':ee.Number(valid).divide(tot)})
    return keepClear_child

def extractQABits(qaBand, bitStart, bitEnd):
    '''
    From a QA band, this function extract the information from bit bitStart to bit bitEnd and return in
    its decimal representation.
    '''
    numBits = bitEnd - bitStart + 1
    qaBits = qaBand.rightShift(bitStart).mod(2**numBits)
    return qaBits

def addNDVI(region,ndvi_bands):
    '''
    Initializes a function that is passed to the map method in GEE
    The child function takes an image as an argument, calculates the NDVI for each pixel, set negative values to zero, and saves in
    the image a property called NDVI that stores the average NDVI within a given region.
    The region is passed to the child_function from the main function.
    '''
    def addNDVI_child(image):
        ndvi = ee.Image(image).normalizedDifference(ndvi_bands).rename('ndvi')
        ndvi = ndvi.where(ndvi.lt(0),0)
        average_ndvi = ndvi.reduceRegion(ee.Reducer.mean(),region).get('ndvi')
        return image.copyProperties(image).set({'NDVI':ee.Number(average_ndvi)})
    return addNDVI_child

def TerrainCorrection(scale,n_bands,smooth=5):
    '''
    Initializes a function that is passed to the map method in GEE
    The child function takes an image, applies the Teillet C smooth correction as describred in
    Riaño et al (2003) (https://ieeexplore.ieee.org/abstract/document/1206729), and returns the
    topographically corrected image.
    This function required the following parameters to initialize the child function:
        - Scale: scale of the final image
        - n_bands: Number of bands of the initial and final image
    '''

    def TerrainCorrection_child(img):
        degree2radian = 0.0174533
        #Region from footprint
        region = ee.Geometry.Polygon(ee.Geometry(img.get('system:footprint')).coordinates())
        #Extract solar zenith and calculate incidence angle (i)
        #Load USGS/SRTMGL1_003 DEM
        terrain = ee.call('Terrain', ee.Image('USGS/SRTMGL1_003')).clip(region)
        #Extract slope in radians for each pixel in the image
        p = terrain.select(['slope']).multiply(degree2radian).tan().divide(smooth).atan()
        #Extract solar zenith angle from the image
        z = ee.Image(ee.Number(img.get('SOLAR_ZENITH_ANGLE')).multiply(degree2radian))
        #Extract solar azimuth from the image
        az = ee.Image(ee.Number(img.get('SOLAR_AZIMUTH_ANGLE')).multiply(degree2radian))
        #Extract aspect in radians for each pixel in the image
        o = terrain.select(['aspect']).multiply(degree2radian)
        cosao = (az.subtract(o)).cos() #cos(ϕa−ϕo)

        #Calculate the cosine of the local solar incidence for every pixel in the image in radians (cosi=cosp*cosz+sinp*sinz*cos(ϕa−ϕo)
        cosi = img.expression('((cosp*cosz) + ((sinp*sinz)*(cosao)))',{'cosp': p.cos(),'cosz': z.cos(),'sinp': p.sin(),'sinz': z.sin(),'az' : az,'o' : o,'cosao': cosao})

        # Create the image to apply the linear regression.The first band is a constant, the second band the insidence angle cosi, and the next bands are the response variables (bands to which correction is being applied)
        # y = a + b*cosi
        reg_img = ee.Image.cat(ee.Image(1).rename('a'),cosi,img)
        #specify the linear regression reducer
        lr_reducer = ee.Reducer.linearRegression(**{'numX': 2,'numY': n_bands})
        #fit the model
        fit = reg_img.reduceRegion(**{'reducer': lr_reducer,'geometry': region,'scale': scale,'maxPixels': 1e10})

        # Calculate C corrector for each band: constant over slope
        coeff_array = ee.Array(fit.get('coefficients'))
        int = ee.Array(coeff_array.toList().get(0))
        slo = ee.Array(coeff_array.toList().get(1))
        C = int.divide(slo)
        Cimg = ee.Image.constant(C.toList())

        #Making the correction
        newimg = img.expression('((img * ((cosz) + C))/(cosi + C))',{'img': img,'cosz': z.cos(),'cosi': cosi,'C': Cimg})
        return newimg.copyProperties(img)
    return TerrainCorrection_child

def makeName(s):
    return s.strip().lower().replace(' ','_')

# Main class that create image from sensor
class downloadImagery():
    def __init__(self,folder,year,sensor,size,topocorrection=True):
        collections = {'L5':{'SR':['LANDSAT/LT05/C01/T1_SR','LANDSAT/LT05/C01/T2_SR'],'TOA':['LANDSAT/LT05/C01/T1_TOA','LANDSAT/LT05/C01/T2_TOA']},
                       'L7':{'SR':['LANDSAT/LE07/C01/T1_SR','LANDSAT/LE07/C01/T2_SR'],'TOA':['LANDSAT/LE07/C01/T1_TOA','LANDSAT/LE07/C01/T2_TOA']},
                       'L8':{'SR':['LANDSAT/LC08/C01/T1_SR','LANDSAT/LC08/C01/T2_SR'],'TOA':['LANDSAT/LC08/C01/T1_TOA','LANDSAT/LC08/C01/T2_TOA']},
                       }
        final_bands        = {'L5':[['B1','B2','B3','B4','B5','B6','B7'],
                                    [],
                                    ['B4','B3']],
                              'L7':[['B1','B2','B3','B4','B5','B6','B7'],
                                    ['B8'],
                                    ['B4','B3']],
                              'L8':[['B1','B2','B3','B4','B5','B6','B7','B10','B11'],
                                    ['B8'],
                                    ['B5','B4']]}
        self.folder  = folder
        self.year    = year
        self.sensor  = sensor
        self.topocorrection = topocorrection
        try:
            self.collection = collections[sensor]
            self.final_spec_bands = final_bands[sensor][0]
            self.final_pan_bands  = final_bands[sensor][1]
            self.ndvi_bands       = final_bands[sensor][2]
            self.pan              = bool(len(self.final_pan_bands))
        except:
            print("Sensor must be L5, L7, or L8")
        self.size     = size
        self.size_adj = int(size/(1+int(self.pan)))
        self.scale = 30
        self.scale_adj = int(self.scale/(1+int(self.pan)))
        self.d = 111319.49079327357

    def coords_to_box(self,coords):
        '''Return bounding box for coords'''
        ll_x,ll_y,ur_x,ur_y =  coords[0]-(0.5/(1+int(self.pan)))*self.size*self.scale/self.d,\
                               coords[1]-(0.5/(1+int(self.pan)))*self.size*self.scale/self.d,\
                               coords[0]+(0.5/(1+int(self.pan)))*self.size*self.scale/self.d,\
                               coords[1]+(0.5/(1+int(self.pan)))*self.size*self.scale/self.d
        return ee.Geometry.Rectangle(ll_x,ll_y,ur_x,ur_y)

    def collect_images(self):
        '''
        Collect imagery from region, adding TOA panchromatic band if available to SR multispectral bands.
        Filters imagery that has more than 95% of useful pixels in region of interest, and adds metadata of average NDVI.
        '''
        ######### Support functions ##########
        def toa_collection():
            ''' Returns TOA collection '''
            toa_collection = ee.ImageCollection(self.collection['TOA'][0])
            for c in self.collection['TOA'][1:]:
                toa_collection = toa_collection.merge(ee.ImageCollection(c))
            return toa_collection

        def addTOA(img):
            ''' add panchromatic band from TOA collection '''
            pan = toa_collection().filter(ee.Filter.eq('LANDSAT_PRODUCT_ID',img.get('LANDSAT_ID'))).select(['B8']).first()
            return ee.Algorithms.If(pan,img.addBands(pan),pan)
        #######################################

        # Collect SR imagery
        collection = ee.ImageCollection(self.collection['SR'][0])
        for c in self.collection['SR'][1:]:
            collection = collection.merge(ee.ImageCollection(c))
        collection = collection.filterDate('{0}-01-01'.format(self.year),
                                           '{0}-12-31'.format(self.year)).filterBounds(self.region)
        # Add TOA bands
        if self.pan:
            collection = collection.map(addTOA,True)

        # Keep only imagery where more than 95% of pixels within region of interest are valid
        collection = collection.map(keepClear(self.region,self.sensor)).filter(ee.Filter.gte('QI',0.95))
        # Save length of collection
        self.collection_length=collection.size().getInfo()
        return collection

    def prepare_batch(self,coords):
        '''
        Collect imagery in region of interest, apply topographic correction if option is selected, and save
        multispectral and panchromatic imagery to instance.
        '''
        # Define region of interest from coordinates of villages (union of boxes)
        self.region = ee.FeatureCollection([ee.Feature(self.coords_to_box(c)) for c in coords]).union().geometry()

        # Collect imagery within region of interest and sort on average NDVI (highest to lowest)
        image  = self.collect_images().map(addNDVI(self.region,self.ndvi_bands)).sort('NDVI',False)

        # Spectral bands: Topographic correction of option selected and reducer (first not null). Save to instance.
        image_spec = image.select(self.final_spec_bands)
        if self.topocorrection:
            image_spec     = image_spec.map(TerrainCorrection(self.scale,len(self.final_spec_bands)))
        image_spec = image_spec.reduce(ee.Reducer.firstNonNull()).multiply(ee.Image.constant(255/10000)).toUint8()
        self.batch_spec = ee.Image(image_spec).clip(self.region).rename(*[b for b in self.final_spec_bands])

        # If pansharpen, same proces for panchromatic band
        if self.pan:
            image_pan = image.select(self.final_pan_bands)
            if self.topocorrection:
                image_pan  = image_pan.map(TerrainCorrection(self.scale_adj,len(self.final_pan_bands)))
                image_pan  = image_pan.reduce(ee.Reducer.firstNonNull()).multiply(ee.Image.constant(255)).toUint8()
                self.batch_pan = ee.Image(image_pan).clip(self.region).rename(*[b for b in self.final_pan_bands])
        else:
            self.batch_pan = None

################################### Main function
def main():
    import argparse
    import pandas as pd
    import numpy as np
    from time import time, sleep
    import sys
    import os

    start_overall = time()
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Google Drive folder")
    parser.add_argument("year", help="Year for which satellite images are collected")
    parser.add_argument("sensor", help="Sensor. Can take values L5 and L7 for TM Landsat 5 and TME Landsat 7 respectively")
    parser.add_argument("batch", help="batch number")
    parser.add_argument("-s","--size", type=int,help="Size of output image, must be an even number. Default is 256.")
    parser.set_defaults(size=256)
    args = parser.parse_args()

    #import villages files
    try:
        villages = pd.read_csv(os.path.join(os.pardir,'local','villages_batch.csv'), dtype=str)
    except:
        print("Wrong file or file path to villages")
        sys.exit(1)
    if args.size % 2:
        print("Size of image must be a even")
        sys.exit(1)

    # Select batch
    batch       = args.batch
    villages   = villages.loc[villages.batch.eq(batch)]
    N = villages.shape[0]

    # Prepare batch
    print('Preparing batch {0} with {1} villages...'.format(batch,N))
    start= time()
    downloadBatch = downloadImagery(args.folder,int(args.year),args.sensor,args.size)
    downloadBatch.prepare_batch([[float(x.lon),float(x.lat)] for i,x in villages.iterrows()])
    time_preparing = np.round(time()-start,5)
    print('{0} sec'.format(time_preparing))

    # Download images
    export_image_toDrive = ee.batch.Export.image.toDrive(**{
                            'image': downloadBatch.batch_spec,
                            'folder':downloadBatch.folder,
                            'description': 'BMSP_{0}'.format(batch),
                            'scale': downloadBatch.scale,
                            'maxPixels':1e13})
    start= time()
    print('Downloading multispectral image to Drive...')
    export_image_toDrive.start()
    while export_image_toDrive.active():
        sleep(1)
    time_spc = np.round(time()-start,5)
    start= time()
    print('{0} sec'.format(time_spc))

    time_pan=0
    if downloadBatch.batch_pan:
        export_image_toDrive = ee.batch.Export.image.toDrive(**{
                                'image': downloadBatch.batch_pan,
                                'folder':downloadBatch.folder,
                                'description':'BPAN_{0}'.format(batch),
                                'scale': downloadBatch.scale_adj,
                                'maxPixels':1e13})
        start= time()
        print('Downloading panchromatic image to Drive...')
        export_image_toDrive.start()
        while export_image_toDrive.active():
            sleep(1)
        time_pan = np.round(time()-start,5)
        start= time()
        print('{0} sec'.format(time_pan))

    # Write batch of village in villages_loc file
    villages_loc = os.path.join(os.pardir,'local','villages_loc.csv')
    villages_loc=open(villages_loc,'a')
    for i,row in villages.iterrows():
        villages_loc.write('{0},{1}\n'.format(makeName(row.village),batch))
    villages_loc.close()
    log = os.path.join(os.pardir,'local','log.csv')

    # Write time in log file
    time_overall = np.round(time()-start_overall,5)
    log = os.path.join(os.pardir,'local','log.csv')
    log=open(log,'a')
    log.write('{0},Finished,{1},{2},{3},{4},{5},{6}\n'.format(batch,villages.n_tiles.tolist()[0],N,time_preparing,time_spc,time_pan,time_overall))
    log.close()
    print('Total time: {0} sec\n'.format(time_overall))

if __name__=="__main__":
  main()
