# Information
name = 'fldm2_map' # Inptu file name with topic, subtopic, and type
path = 'fracture-proof/version_2/fldm2/' # Input relative path to file 
directory = '/home/drewc/GitHub/' # Input working directory
title = 'FractureProof v2.1 - Diabetes Mortality in Florida' # Input descriptive title
author = 'Andrew S. Cistola, MPH' # Input Author

## Create FL Map from Zip Code Data

### Import python libraries
import os # Operating system navigation
from datetime import datetime
from datetime import date

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes

### Import Visualization Libraries
import matplotlib.pyplot as plt # Comprehensive graphing package in python
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import geopandas as gp # Simple mapping library for csv shape files with pandas like syntax for creating plots using matplotlib 
import descartes # Mapping library for use with geopandas
import mapclassify # Mapping library for use with geopandas

### Set Directory
os.chdir(directory) # Set wd to project repository

### Set Timestamps
day = str(date.today())
stamp = str(datetime.now())

### Preprocess First Dataset
df_d1 = pd.read_csv('fracture-proof/version_2/_data/FDOH_5Y2018_ZCTA.csv') # Import first dataset saved as csv in _data folder
df_d1 = df_d1[df_d1['POPULATION'] > 500] # Susbet numeric column by condition
df_d1 = df_d1.filter(['K00_K99_R1000', 'ZCTA']) # Drop or filter columns to keep only feature values and idenitifer
df_d1 = df_d1.rename(columns = {'ZCTA': 'ID', 'K00_K99_R1000': 'quant'}) # Apply standard name to identifier and quantitative outcome
df_d1.info() # Get class, memory, and column info: names, data types, obs

### Geojoin outcome table with polygons
gdf_d1 = gp.read_file('fracture-proof/version_2/_data/cb_2018_us_zcta510_500k/cb_2018_us_zcta510_500k.shp') # Import shape files from folder with all other files downloaded
gdf_d1['ID'] = gdf_d1['ZCTA5CE10'].astype('str') # Change data type of column in data frame
gdf_d1['ID'] = gdf_d1['ID'].str.rjust(5, '0') # add leading zeros of character column using rjust() function
gdf_d1['ID'] = 'ZCTA' + gdf_d1['ID'] # Combine string with column
gdf_d1 = gdf_d1.filter(['ID', 'geometry']) # Keep only selected columns
gdf_d1 = pd.merge(gdf_d1, df_d1, on = 'ID', how = 'inner') # Geojoins can use pandas merge as long as geo data is first passed in function
gdf_d1 = gdf_d1.filter(['quant', 'geometry']) # Keep only selected columns
gdf_d1.info() # Get class, memory, and column info: names, data types, obs

### Create choropleth for FL
map = gdf_d1.plot(column = 'quant', cmap = 'Blues', figsize = (10, 10), scheme = 'equal_interval', k = 9, legend = True, legend_kwds={'title': 'Deaths per 1000', 'loc': 'center left'})
map.set_title('Diabetes Mortality Rates by Zip Code in Florida 204-2018', fontdict = {'fontsize': 16}, loc = 'center')
map.set_axis_off()
map.annotate('', xy = (0.5, 0.1), xytext = (0.5, 0.0), arrowprops = dict(arrowstyle = 'simple'), fontsize = 10, xycoords= 'axes fraction')
map.annotate('N', xy = (0.51, 0.01), xycoords = 'axes fraction', fontsize = 16)
map.annotate('Andrew S. Cistola, MPH', xy = (0.0, 0.15), xycoords = 'axes fraction', fontsize = 10)
map.annotate(stamp, xy = (0.0, 0.1), xycoords = 'axes fraction', fontsize = 10)
map.add_artist(AnchoredSizeBar(map.transData, 3, '300 km', loc = 'lower left'))
plt.savefig(path + '_fig/' + name + '.png', dpi = 1000, bbox_inches = 'tight')





