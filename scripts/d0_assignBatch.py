# -*- coding: utf-8 -*-
"""
Assign batch number to each image. Batches are created within groups of villages whose box intersect with the same group of WRS2 tiles.
Number of batches per tile is equal to1+(#Villages in tile//max_batch_size).

Positional argumentes:
    - wrs2:     Path to wrs2 shapefile relative to local folder in root dir of this repo.
    - villages: Path to file with village coords relative to local folder located in root dir of this repo. The file must have three columns: village, with the name of the village;
                lon, with the longitude of the village; and lat, with the latitude of the village.
Arguments:
    - max_batch_size --t: Threshold to create new cluster of villages within intersection of WRS2 tile. Default is 1000
    - s --size: Height and width of box around each village in pixels, must be an even number. Default is 256.
    - d --d: Size of pixels in meters. Default is 15.
Output:
   - local/villages_batch.csv: csv file with the following columns:
        * villages: name of village
        * lon    : longitude of village
        * lat    : latitude of village
        * tile   : ID that identifies the group of WRS2 tiles with which the village's box intersects
        * batch  : ID of batch
    - batch_summary.csv: csv file that shows the number of villages per batch

Author:        Felipe Jordan
Date created:  04/05/2020
Last modified: 04/08/2020
"""

import argparse
import os
import pandas as pd
import geopandas as gpd
import shapely as shp
import numpy as np
from sklearn.cluster import KMeans

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("wrs2", help="Path to wrs2 shapefile, relative to local dir")
parser.add_argument("villages", help="Path to villages.csv, file relative to local dir")
parser.add_argument("-t","--max_batch_size", type=int,help="Threshold to create new cluster of villages within intersection of WRS2 tile. Default is 1000.")
parser.set_defaults(max_batch_size=1000)
parser.add_argument("-s","--size", type=int,help="Height and width of box around each village in pixels, must be an even number. Default is 256.")
parser.set_defaults(size=256)
parser.add_argument("-d","--d", type=int,help="Size of pixels in meters")
parser.set_defaults(d=15)
args = parser.parse_args()
m2d            = 1/111319.49079327357
max_batch_size = int(args.max_batch_size)
size           = int(args.size)
d              = int(args.d)

# Import imput
wrs2 = gpd.read_file(os.path.join(os.pardir,'local',args.wrs2))
villages = pd.read_csv(os.path.join(os.pardir,'local',args.villages))
villages['geometry'] = [shp.geometry.box(x-.5*size*d*m2d,y-.5*size*d*m2d,x+.5*size*d*m2d,y+.5*size*d*m2d) for x,y in zip(villages.lon,villages.lat)]
villages = gpd.GeoDataFrame(villages)
villages.crs = wrs2.crs

# Join both geometries
villages_expanded = gpd.sjoin(villages,wrs2[['WRSPR','geometry']],how='left',op='intersects')

# List of PRID with which each village intersects
def listPRID(x):
    PRID = np.unique(x.WRSPR).tolist()
    PRID.sort()
    return pd.Series([x.lon.tolist()[0],x.lat.tolist()[0],'-'.join([str(s) for s in PRID]),len(PRID)])
villagesPRID = villages_expanded.groupby(['village']).apply(listPRID)
villagesPRID.columns=['lon','lat','PRIDLIST','n_tiles']
villagesPRID.reset_index(inplace=True)
#villagesPRID['PRGROUP'] = villagesPRID[['PRIDLIST']].groupby(['PRIDLIST']).grouper.group_info[0]
#del villagesPRID['PRIDLIST']

# Groups of max max_batch_size of villages within same group of PRID
def batchID(x):
    n = x.shape[0]
    points = np.array([[lon,lat] for lon,lat in zip(x.lon,x.lat)])
    nc = 1 + n // (max_batch_size)
    km = KMeans(n_clusters=nc).fit(points)
    x['g'] = km.labels_
    return x

villagesPRID2 = villagesPRID.groupby(['PRIDLIST']).apply(batchID)
villagesPRID2.reset_index(drop=[False,True],inplace=True)
villagesPRID2['tile']  = villagesPRID2[['PRIDLIST']].groupby(['PRIDLIST']).grouper.group_info[0]
villagesPRID2['batch'] = villagesPRID2[['PRIDLIST','g']].groupby(['PRIDLIST','g']).grouper.group_info[0]

# Save to file
villagesPRID2[['village','lon','lat','tile','batch','n_tiles']].to_csv(os.path.join(os.pardir,'local','villages_batch.csv'),index=False)

# Summary
villages_per_batch = villagesPRID2[['batch','village']].groupby('batch').count()
villages_per_batch.rename(columns={'village':'n'},inplace=True)
villages_per_batch.reset_index(inplace=True)
villages_per_batch.to_csv(os.path.join(os.pardir,'local','batch_summary.csv'),index=False)
print('Number of batches: {0}'.format(villages_per_batch.shape[0]))
