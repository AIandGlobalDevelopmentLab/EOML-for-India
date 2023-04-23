'''
This file calls the script download_batch.py for each batch of images from batch_from to batch_to (inclusive).
See that scripts for details on arguments.

Author:        Felipe Jordan
Date created:  04/05/2020
Last modified: 04/08/2020
'''

# Import the Earth Engine Python Package:
import os
import subprocess
import argparse
from time import time
import numpy as np

# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument("folder", help="Google Drive folder")
parser.add_argument("year", help="Year for which satellite images are collected")
parser.add_argument("sensor", help="Sensor. Can take values L5 and L7 for TM Landsat 5 and TME Landsat 7 respectively")
parser.add_argument("batch_from", help="batch to download")
parser.add_argument("batch_to",   help="batch to download")
parser.add_argument("-s","--size", type=int,help="Size of output image, must be an even number. Default is 256.")
parser.set_defaults(size=256)
args = parser.parse_args()

batch_from = int(args.batch_from)
batch_to   = int(args.batch_to)
i=batch_from

# Create log and villages_loc csv files
log_path = os.path.join(os.pardir,'local','log.csv')
if not os.path.exists(log_path):
    log=open(log_path,'w')
    log.write('batch,status,tiles,villages,preparation,multispectral,panchromatic,total\n')
    log.close()

villages_loc = os.path.join(os.pardir,'local','villages_loc.csv')
if not os.path.exists(villages_loc):
    villages_loc = open(villages_loc,'w')
    villages_loc.write('village,batch\n')
    villages_loc.close()

# Loop through batches
while i<=batch_to:
    start = time()
    p = subprocess.run('python download_batch.py {0} {1} {2} {3} {4} {5}'.format(args.folder,args.year,args.sensor,i,'-s',args.size),shell=True)
    if p.returncode:
        time_fail = np.round(time()-start,5)
        print('Batch {0} failed. {1} sec. Try again...'.format(i,time_fail))
        log=open(log_path,'a')
        log.write('{0},Failed,,,,,,{1}\n'.format(i,time_fail))
        log.close()
    else:
        i+=1
