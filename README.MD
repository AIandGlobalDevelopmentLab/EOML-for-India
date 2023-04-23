
# observatory-of-poverty
This repository contains scripts and data to reproduce the Forthcoming article in Social Indicators Research
## Required libraries
* Google Earth Engine API
* Geopandas
* Rasterio

## Usage

1. **Download imagery**: This section will download large tiles encompasing a batch of villages to a Google Drive folder of your choice. Each tile is a mozaic of the available imagery within that year, selecting the best images in terms of their average level of NDVI (hence, images captured in full growth season are prioratized).
    1. Create a local folder in the root of the repository. Git will ignore this folder.
    2. Create file local/villages.csv. Must include three columns: village, with the name or ID of villages; lon, with latitude in decimal form (e.g 6.176874); and lon, with longitude in decimal form (e.g -14.114580).
    3. Run the d0_assignBatch.py script. See script for details.
    4. Run the d1_downloadBatches.py script, using as input the villages_batch.csv file created by the previos script. See script for details.
    5. The previous script will download the imagery to a folder in your Google Drive. Download the folder and put the imagery in the local folder (e.g local/images). Note that the script will also generate a file local/villages_loc.csv, which links each village to its batch number.

2. **Retrieve villages' images from tiles**: The result of step 1 is a folder with large images that contain several villages. To create an image for a particular village, use the class villageImage defined in the village_image.py script. To initialize the class, you need to provide the name or ID of the village (same used in the village.csv file of step 1) and the longitude and latitude of the village. By default, the class will load the local/villages_loc.csv file to locate the tile that contains the village. The instance can be used to retrieve a subset of the original bands (which can optionally be pansharpened using the panchromatic band if available) for image size as a numpy array, or save it as a tif file. See the end of the village_image.py script for a demonstration of the class usage.

3. **Preparing input data for training models** :
1. The first step is to retrieve village images from tiles, to do that run generate_images.py, seting the correct arguments looking at the example in the code. This will create a folder of images in the local folder, which will be used to train the model.
2. The second step is to clean the census to extract the 16 dimensional feature vector we use for training the model. Use the code in dataprocessing folder to do so.
a) For the 2011 census :
    1. First run data_processing/census_2011/rawdata/download_commands.sh to download the rawdata from the indian govt. census website
    2. Then run data_processing/census_2011/preprocessing/HL_datacreation.py to create a census.csv, move to local/ folder
b) For the 2001 census :
    1. Download district wise xlsx files from the indian govt. website one by one (Not automatabile as each file has to be selected from a drop down menu to download, and no list of urls).
    2. Run the jupyter notebook data_processing/census_2001/census_2001_data_cleaning.ipynb end to end.


4. **Training models** :
    1. ***Training base model*** :
        1. Run scripts/main.py . Set the arguments as needed. The default arguments will train the census model on the 2011 census
    2: ***Training transfer learning model***:

5. **Data transformations** :
    1. Optimal transport
    2. Normalising distributions

6. **Generating tables for figures** :
