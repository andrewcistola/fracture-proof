# Information
name = 'mrfpsctwc_fldm2_code' # Inptu file name with topic, subtopic, and type
path = 'fracture-proof/version_2/v2-1/' # Input relative path to file 
directory = '/home/drewc/GitHub/' # Input working directory
title = 'FractureProof v2.1 - Mr. Fracture Proofs Contemplative Woodcarving for Diabetes Mortality in Florida' # Input descriptive title
author = 'Andrew S. Cistola, MPH' # Input Author

## Setup Workspace

### Import python libraries
import os # Operating system navigation
from datetime import datetime
from datetime import date

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes

### Import statistics libraries
import statsmodels.api as sm # Statistics package best for regression models

### Import scikit-learn libraries
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest regression component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome
from sklearn.svm import LinearSVC # Linear Support Vector Classification from sklearn

### Import keras libraries
from keras.models import Sequential # Uses a simple method for building layers in MLPs
from keras.models import Model # Uses a more complex method for building layers in deeper networks
from keras.layers import Dense # Used for creating dense fully connected layers
from keras.layers import Input # Used for designating input layers

### Import PySAL Libraries
import libpysal as ps # Spatial data science modeling tools in python
from mgwr.gwr import GWR, MGWR # Geographic weighted regression modeling tools
from mgwr.sel_bw import Sel_BW # Bandwidth selection for GWR

### Import Visualization Libraries
import matplotlib.pyplot as plt # Comprehensive graphing package in python
import geopandas as gp # Simple mapping library for csv shape files with pandas like syntax for creating plots using matplotlib 

### Set Directory
os.chdir(directory) # Set wd to project repository

### Set Timestamps
day = str(date.today())
stamp = str(datetime.now())

### Append to Text File
text_file = open(path + name + '_' + day + '.txt', 'w') # Open text file and name with subproject, content, and result suffix
text_file.write('####################' + '\n\n')
text_file.write('Title: ' + title + '\n') # Script title
text_file.write('Author: ' + author + '\n') # Script Author
text_file.write('Filename: ' + name + '.py' + '\n') # Filename of script
text_file.write('Realtive Path: ' + path + '\n') # Relative path to script
text_file.write('Working Directory: ' + directory + '\n') # Directory used for script run
text_file.write('Time Run: ' + stamp + '\n') # Timestamp of script run
text_file.write('\n' + '####################' + '\n\n')
text_file.close() # Close file

# Step 1: Data Processing of Predictors and Outcomes
s1 = 'Step 1: Raw Data Processing and Feature Engineering' # Step 1 descriptive title
d1 = 'Florida Deaprtment of Health Vital Statistics 113 Leading Mortality Causes 2014-2018 Zip Code 5-year Average' # Input descriptive title for 1st dataset
d2 = 'US Census American Community Survey 2014-2018 Zip Code 5-year Average' # Input descriptive title for 2nd dataset

## Preprocess First Dataset
df_d1 = pd.read_csv('fracture-proof/version_2/_data/FDOH_5Y2018_ZCTA.csv') # Import first dataset saved as csv in _data folder
df_d1 = df_d1[df_d1["POPULATION"] > 500] # Susbet numeric column by condition
df_d1 = df_d1.filter(['K00_K99_R1000', 'ZCTA']) # Drop or filter columns to keep only feature values and idenitifer
df_d1.info() # Get class, memory, and column info: names, data types, obs

### Preprocess Second Data
df_d2 = pd.read_csv('fracture-proof/version_2/_data/ACS_5Y2018_ZCTA.csv') # Import dataset saved as csv in _data folder
df_d2 = df_d2.drop(columns = ['ST', 'FIPS']) # Drop or filter columns to keep only feature values and idenitifer
df_d2 = df_d2.select_dtypes(exclude = ['int64']) # Drop all unwanted data types
df_d2.info() # Get class, memory, and column info: names, data types, obs.

### Join Datasets by ID and define targets
df_full = pd.merge(df_d1, df_d2, on = 'ZCTA', how = 'inner') # Join datasets to create table with predictors and outcome
df_full = df_full.dropna(subset = ['K00_K99_R1000']) # Drop all outcome rows with NA values
df_full.info() # Get class, memory, and column info: names, data types, obs.

### Create outcome table
df_XY = df_full.rename(columns = {'ZCTA': 'ID'}) # Apply standard name to identifier used for joining datasets
df_XY = df_XY.rename(columns = {'K00_K99_R1000': 'quant'}) # Apply standard name to identifier used for joining datasets
df_Y = df_XY.filter(['quant', 'ID']) # Create Outcome table
df_Y = df_Y.set_index('ID') # Set identifier as index
df_Y.info() # Get class, memory, and column info: names, data types, obs.

### Create standard scaled predictor table
df_X = df_XY.drop(columns = ['quant', 'ID']) # Drop Unwanted Columns
df_X = df_X.replace([np.inf, -np.inf], np.nan) # Replace infitite values with NA
df_X = df_X.dropna(axis = 1, thresh = 0.75*len(df_X)) # Drop features less than 75% non-NA count for all columns
df_X = pd.DataFrame(SimpleImputer(strategy = 'median').fit_transform(df_X), columns = df_X.columns) # Impute missing data
df_X = pd.DataFrame(StandardScaler().fit_transform(df_X.values), columns = df_X.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_X['ID'] = df_XY['ID'] # Save ID as column in predictor table
df_X = df_X.set_index('ID') # Set identifier as index
df_X.info() # Get class, memory, and column info: names, data types, obs.

## Append to Text File
text_file = open(path + name + '_' + day + '.txt', 'a') # Open text file and name with subproject, content, and result suffix
text_file.write(s1 + '\n\n') # Step 1 descriptive title
text_file.write(d1 + '\n') # First dataset descriptive title
text_file.write(d2 + '\n\n') # Second dataset descriptive title
text_file.write('Target labels: quant = Diabetes Related (K00-K99) Raw Mortality Rate per 1000k' + '\n') # Target labels
text_file.write('Target processing: None' + '\n\n') # Target processing
text_file.write(str(df_Y.describe())  + '\n\n') # Descriptive statistics for target
text_file.write('Features labels: ACS Percent Estimates' + '\n') # Number of observations and variables
text_file.write('Feature processing: 75% nonNA, Median Imputed NA, Standard Scaled' + '\n\n') # Feature processing
text_file.write('Rows, Columns: ' + str(df_X.shape) + '\n\n') # Number of observations and variables
text_file.write('####################' + '\n\n')
text_file.close() # Close file

# Step 2: Identify Predictors with Open Box Models
s2 = "Step 2: Identify Predictors with Open Models" # Step 1 descriptive title
m1 = "Principal Component Analysis" # Model 1 descriptive title
m2 = "Random Forests" # Model 2 descriptive title
m3 = "Recursive feature Elimination" # Model 3 descriptive title

## Principal Component Analysis
degree = len(df_X.columns) - 1  # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(df_X) # Fit initial PCA model
df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
components = len(df_comp.index) - 1 # Save count of components for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(df_X) # finally call fit_transform on the aggregate data to create PCA results object
df_pc = pd.DataFrame(pca.components_, columns = df_X.columns) # Export eigenvectors to data frame with column names from original data
df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pc = df_pc.abs() # Get absolute value of eigenvalues
df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
df_p = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_p = df_p[df_p.MaxEV > df_p.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_p = df_p.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
df_pca = df_p.rename(columns = {"index": "Feature"}) # Rename former index as features
df_pca = df_pca.sort_values(by = ["MaxEV"], ascending = False) # Sort Columns by Value
df_pca.info() # Get class, memory, and column info: names, data types, obs.

### Random Forest Regressor
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(df_X, df_Y["quant"]) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(df_X, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Feature", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean
df_rf = df_rf.sort_values(by = ["Gini"], ascending = False) # Sort Columns by Value
df_rf.info() # Get class, memory, and column info: names, data types, obs.

### Fracture: Join RF and PCA 
df_fr = pd.merge(df_pca, df_rf, on = "Feature", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
fracture = df_fr["Feature"].tolist() # Save features from data frame
df_fr.info() # Get class, memory, and column info: names, data types, obs.

### Recursive Feature Elimination
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(df_X[fracture], df_Y['quant']) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(df_X[fracture], rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Feature", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe.sort_values(by = ["RFE"], ascending = True) # Sort Columns by Value
df_rfe = df_rfe[df_rfe["RFE"] == True] # Select Variables that were True
df_rfe.info() # Get class, memory, and column info: names, data types, obs.

### FractureProof: Join RFE with Fracture
df_fp = pd.merge(df_fr, df_rfe, on = "Feature", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
fractureproof = df_fp["Feature"].tolist() # Save chosen featres as list
df_fp.info() # Get class, memory, and column info: names, data types, obs.

### Add feature labels
df_l1 = pd.read_csv('fracture-proof/version_2/_data/ACS_5Y2018_labels.csv') # Import dataset saved as csv in _data folder
df_l2 = pd.read_csv('fracture-proof/version_2/_data/FDOH_5Y2018_labels.csv')
df_label = pd.concat([df_l1, df_l2]) # Combine rows with same columns
df_label = df_label.filter(["Feature", "Label"]) # Keep only selected columns
df_label = df_label.set_index("Feature") # Set column as index
df_label = df_label.transpose() # Switch rows and columns
df_label = df_label[fractureproof] # Save chosen featres as list
df_label = df_label.transpose() # Switch rows and columns
df_label = df_label.reset_index() # Reset index
l_label = list(zip(df_label["Feature"], df_label["Label"])) # Create list of variables alongside RFE value 
df_label.info() # Get class, memory, and column info: names, data types, obs.

### Append to Text File
text_file = open(path + name + "_" + day + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(s2 + "\n\n") # Line of text with space after
text_file.write("Models: " + m1 + ", " + m2 + ", " + m3 + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Values: Eigenvectors, Gini Impurity, Boolean" + "\n") # Add two lines of blank text at end of every section text
text_file.write("Thresholds: Mean, Mean, Cross Validation" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(df_fp)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(l_label)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

## Step 3: Create Informative Prediction Model
s3 = 'Step 3: Create Informative Preidction Model' # Step 1 descriptive title
m4 = 'Multiple Linear Regression Model' # Model 3 descriptive title

### Principal Component Analysis
mrfractureproof = df_X[fractureproof].columns.to_list() 

### Add confounders to multiple regression model
mrfractureproof.append("quant") # Add outcome to regression dataset

### Create Multiple Regression Model
df_mrfp = df_XY[mrfractureproof]
df_mrfp = df_mrfp.dropna()
X = df_mrfp.drop(columns = ['quant'])
Y = df_mrfp['quant']
mod = sm.OLS(Y, X) # Create linear model
res = mod.fit() # Fit model to create result
res.summary() # Print results of regression model

### Add feature labels
df_l1 = pd.read_csv('fracture-proof/version_2/_data/ACS_5Y2018_labels.csv') # Import dataset saved as csv in _data folder
df_l2 = pd.read_csv('fracture-proof/version_2/_data/FDOH_5Y2018_labels.csv')
df_label = pd.concat([df_l1, df_l2]) # Combine rows with same columns
df_label = df_label.filter(["Feature", "Label"]) # Keep only selected columns
df_label = df_label.set_index("Feature") # Set column as index
df_label = df_label.transpose() # Switch rows and columns
mrfractureproof.remove("quant") # Add outcome to regression dataset
df_label = df_label[mrfractureproof] # Save chosen featres as list
df_label = df_label.transpose() # Switch rows and columns
df_label = df_label.reset_index() # Reset index
l_label = list(zip(df_label["Feature"], df_label["Label"])) # Create list of variables alongside RFE value 
df_label.info() # Get class, memory, and column info: names, data types, obs.

### Append to Text File
text_file = open(path + name + "_" + day + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(s3 + "\n\n") # Line of text with space after
text_file.write("Models: " + m4 + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(res.summary())  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(l_label)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

## Step 3: Geographic Weighted Regression
s3 = "Step 3: Geographic Weighted Regression"
m5 = "Multi-scale Geographic Weighted Regression"

### Geojoin Susbet Table with Polygons, get centroid with coordinates
gdf_XY = gp.read_file('fracture-proof/version_2/_data/cb_2018_us_zcta510_500k/cb_2018_us_zcta510_500k.shp') # Import shape files from folder with all other files downloaded
gdf_XY['ID'] = gdf_XY['ZCTA5CE10'].astype('str') # Change data type of column in data frame
gdf_XY['ID'] = gdf_XY['ID'].str.rjust(5, '0') # add leading zeros of character column using rjust() function
gdf_XY['ID'] = 'ZCTA' + gdf_XY['ID'] # Combine string with column
gdf_XY = gdf_XY.filter(['ID', "geometry"]) # Keep only selected columns
gdf_XY = pd.merge(gdf_XY, df_XY, on = 'ID', how = 'inner') # Geojoins can use pandas merge as long as geo data is first passed in function
gdf_XY['x'] = gdf_XY['geometry'].centroid.x # Save centroid coordinates as separate column
gdf_XY['y'] = gdf_XY['geometry'].centroid.y # Save centroid coordinates as separate column
gdf_XY['coordinates'] = list(zip(gdf_XY['x'], gdf_XY['y'])) # Save individual coordinates as column of paired list
gdf_XY = gdf_XY.drop(columns = ['x', 'y', 'geometry']) # Drop Unwanted Columns
gdf_XY.info() # Get class, memory, and column info: names, data types, obs.

### Setup GWR table
gdf_gwr = gdf_XY.set_index("ID") # Set column as index
woodcarving = gdf_gwr[mrfractureproof].columns.to_list() # Save sleetced variables as list
woodcarving.append('quant') # Add outcome to regression dataset
woodcarving.append('coordinates') # Add outcome to regression dataset
gdf_gwr = gdf_gwr[woodcarving] # Subset dataframe by sleetced variables
gdf_gwr = gdf_gwr.dropna() # Drop all rows with NA values
c = list(gdf_gwr["coordinates"]) # save coordinates column as list
x = gdf_gwr.drop(columns = ['quant', 'coordinates']).values # save selected features as numpy array
y = gdf_gwr["quant"].values # save target as numpy array
y = np.transpose([y]) # Transpose numpy array to fit GWR input
gdf_gwr.info() # Get class, memory, and column info: names, data types, obs.

### Create GWR model
mgwr_selector = Sel_BW(c, y, x, multi = True) # create model to calibrate selector
mgwr_bw = mgwr_selector.search(multi_bw_min = [2]) # search for selectors with minimum of 2 bandwidths this may take a while
mgwr_results = MGWR(c, y, x, mgwr_selector).fit() # fit MGWR model
mgwr_results.summary() # Show MGWR summary

### Export GWR results to new table
woodcarving.remove('quant') # Add outcome to regression dataset
woodcarving.remove('coordinates') # Add outcome to regression dataset
woodcarving = ['Intercept'] + woodcarving # Insert intercept label at front of list
df_gwr = pd.DataFrame(mgwr_results.params, columns = [woodcarving]) # Create data frame of importances with variables and gini column names
gdf_ID = gdf_gwr.reset_index() # Reset index on GWR inputs
df_gwr['ID'] = gdf_ID['ID'] # Ad ID column from GWR inputs table
df_gwr.info()  # Get class, memory, and column info: names, data types, obs.

### Join ZCTA to FIPS Data
df_FIPS = pd.read_csv('fracture-proof/version_2/_data/FIPS_ZCTA_key.csv') # Import first dataset saved as csv in _data folder
df_FIPS = df_FIPS.filter(['FIPS', 'ZCTA']) # Keep only selected columns
df_FIPS = df_FIPS.rename(columns = {'ZCTA': 'ID', 'FIPS': 'ID_2'}) # Rename multiple columns in place
gdf_ID_2 = gdf_gwr.reset_index() # Reset Index
df_gwr = pd.merge(gdf_gwr, df_FIPS, on = 'ID', how = 'left') # Join zip code geo weighted coefficients to county labels
df_gwr = df_gwr.dropna() # Drop all rows with NA values
df_gwr = df_gwr.set_index('ID') # Set column as index
df_gwr = df_gwr.drop(columns = ['coordinates', 'quant']) # Drop Unwanted Columns
df_gwr = df_gwr.groupby(['ID_2'], as_index = False).max() # Group data by columns and maximum value
df_gwr.info() # Get class, memory, and column info: names, data types, obs.

### Append to Text File
text_file = open(path + name + "_" + day + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(s3 + "\n\n") # Line of text with space after
text_file.write("Models: " + m5 + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Bandwidths: " + str(mgwr_bw) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Mean Coefficients by County " + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(df_gwr.describe()) + '\n\n') # Descriptive statistics for target
text_file.write("####################" + "\n\n")
text_file.close() # Close file

# Step 4: Data Processing of 2nd Geographic Layer
s4 = 'Step 4: Raw Data Processing and Feature Engineering (2nd Geographic Layer)' # Step 1 descriptive title
d3 = "Beureau for Economic Analysis GDP Measures by County 2014-2018 5-year Average"
d4 = "Health Resources and Servcies Administration Area Heath Resource File Populaton Rates by County 2014-2018 5-year Average"

## Preprocess First Dataset
df_d3 = pd.read_csv('fracture-proof/version_2/_data/.csv') # Import first dataset saved as csv in _data folder
df_d3 = df_d3.filter([]) # Drop or filter columns to keep only feature values and idenitifer
df_d3.info() # Get class, memory, and column info: names, data types, obs

### Preprocess Second Data
df_d4 = pd.read_csv('fracture-proof/version_2/_data/.csv') # Import dataset saved as csv in _data folder
df_d4 = df_d4.drop(columns = []) # Drop or filter columns to keep only feature values and idenitifer
df_d4.info() # Get class, memory, and column info: names, data types, obs.

### Join Datasets by ID and define targets
df_full_2 = pd.merge(df_d3, df_d4, on = 'FIPS', how = 'inner') # Join datasets to create table with predictors and outcome
df_full_2 = df_full_2.dropna(subset = ['']) # Drop all outcome rows with NA values
df_full_2.info() # Get class, memory, and column info: names, data types, obs.

### Create outcome table
df_XY_2 = df_full_2.rename(columns = {'FIPS': 'ID_2'}) # Apply standard name to identifier used for joining datasets
df_XY_2 = df_XY_2.rename(columns = {'': 'quant'}) # Apply standard name to identifier used for joining datasets
df_Y_2 = df_XY_2.filter(['quant', 'ID_2']) # Create Outcome table
df_Y_2 = df_Y_2.set_index('ID_2') # Set identifier as index
df_Y_2.info() # Get class, memory, and column info: names, data types, obs.

### Create standard scaled predictor table
df_X_2 = df_XY_2.drop(columns = ['quant', 'ID_2']) # Drop Unwanted Columns
df_X_2 = df_X_2.replace([np.inf, -np.inf], np.nan) # Replace infitite values with NA
df_X_2 = df_X_2.dropna(axis = 1, thresh = 0.75*len(df_X_2)) # Drop features less than 75% non-NA count for all columns
df_X_2 = pd.DataFrame(SimpleImputer(strategy = 'median').fit_transform(df_X_2), columns = df_X_2.columns) # Impute missing data
df_X_2 = pd.DataFrame(StandardScaler().fit_transform(df_X_2.values), columns = df_X_2.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_X_2['ID'] = df_XY_2['ID'] # Save ID as column in predictor table
df_X_2 = df_X_2.set_index('ID') # Set identifier as index
df_X_2.info() # Get class, memory, and column info: names, data types, obs.

## Append to Text File
text_file = open(path + name + '_' + day + '.txt', 'a') # Open text file and name with subproject, content, and result suffix
text_file.write(s4 + '\n\n') # Step 1 descriptive title
text_file.write(d3 + '\n') # First dataset descriptive title
text_file.write(d4 + '\n\n') # Second dataset descriptive title
text_file.write('Target labels: quant = ' + '\n') # Target labels
text_file.write('Target processing: None' + '\n\n') # Target processing
text_file.write(str(df_Y_2.describe())  + '\n\n') # Descriptive statistics for target
text_file.write('Features labels: ACS Percent Estimates' + '\n') # Number of observations and variables
text_file.write('Feature processing: 75% nonNA, Median Imputed NA, Standard Scaled' + '\n\n') # Feature processing
text_file.write('Rows, Columns: ' + str(df_X_2.shape) + '\n\n') # Number of observations and variables
text_file.write('####################' + '\n\n')
text_file.close() # Close file

## Step 5: Identify 2nd Layer Predictors
s5 = "Step 5: Identify 2nd Layer Predictors" # Step 1 descriptive title
m6 = "Support Vector Machines"

### Support Vector Machines
vector = LinearSVC() 
vector.fit(df_X_2, df_Y_2)
svm = vector.coef_
l_svm = list(zip(df_X_2, svm)) # Create list of variables alongside importance scores 
df_svm = pd.DataFrame(l_svm, columns = ["Feature", "Coefficient"]) # Create data frame of importances with variables and gini column names
df_svm = df_svm[(df_svm["Coefficient"] > df_svm["Coefficient"].mean())] # Subset by Gini values higher than mean
df_svm = df_svm.sort_values(by = ["Coefficients"], ascending = False) # Sort Columns by Value
wood = df_svm["Feature"].tolist() # Save features from data frame
df_svm.info() # Get class, memory, and column info: names, data types, obs.

### Principal Component Analysis
degree = len(df_X_2[wood].columns) - 1  # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(df_X_2[wood]) # Fit initial PCA model

### Export Variance Ratios
cvr = pca.explained_variance_ratio_.cumsum() # Save cumulative variance ratio
comps = np.count_nonzero(cvr) - np.count_nonzero(cvr > 0.95) + 1 # Save number of components above threshold value

### Component Loadings
load = pca.components_.T * np.sqrt(pca.explained_variance_) # Export component loadings
df_load = pd.DataFrame(load, index = df_X[fractureproof].columns) # Create data frame of component loading
df_load = df_load.iloc[:, 0:comps] # Save columns by components above threshold
df_load = df_load.abs() # get absolute value for column or data frame
df_load = df_load[df_load > 0.5] # Subset by character
df_load = df_load.dropna(thresh = 1) # Drop all rows without 1 non-NA value
df_load = df_load.dropna(axis = 'columns', thresh = 1) # Drop all rows without 1 non-NA value
woodcarving = df_load.index.to_list() # Save variables to list

### Add feature labels
df_l3 = pd.read_csv('fracture-proof/version_2/_data/ACS_5Y2018_labels.csv') # Import dataset saved as csv in _data folder
df_l4 = pd.read_csv('fracture-proof/version_2/_data/FDOH_5Y2018_labels.csv')
df_label = pd.concat([df_l3, df_l4]) # Combine rows with same columns
df_label = df_label.filter(["Feature", "Label"]) # Keep only selected columns
df_label = df_label.set_index("Feature") # Set column as index
df_label = df_label.transpose() # Switch rows and columns
df_label = df_label[woodcarving] # Save chosen featres as list
df_label = df_label.transpose() # Switch rows and columns
df_label = df_label.reset_index() # Reset index
l_label = list(zip(df_label["Feature"], df_label["Label"])) # Create list of variables alongside RFE value 
df_label.info() # Get class, memory, and column info: names, data types, obs.

### Append to Text File
text_file = open(path + name + "_" + day + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(s5 + "\n\n") # Line of text with space after
text_file.write("Models: " + m5 + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Values: Coefficients" + "\n") # Add two lines of blank text at end of every section text
text_file.write("Thresholds: Mean" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(df_svm)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Models: " + m1 + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Cumulative Variance: Threshold = 95%" + "\n") # Add two lines of blank text at end of every section text
text_file.write(str(a_cvr) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Component Loadings" + "\n") # Add two lines of blank text at end of every section text
text_file.write(str(df_load)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(l_label)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

## Step 6: Create Informative Prediction Model with both geographic layers
s6 = 'Step 3: Create Informative Preidction Model with both geographic layers' # Step 1 descriptive title

### Join Datasets by ID and define targets
df_full_f = pd.merge(df_XY, df_XY_2, on = 'FIPS', how = 'right') # Join datasets to create table with predictors and outcome
df_full_f = df_full_f.dropna(subset = ['quant']) # Drop all outcome rows with NA values
df_full_f["train"] = np.where(df_full_f["quant"] > df_full_f["quant"].quintile(0.95), 1, 0) # Create categorical test target outcome based on conditions
df_full_f["test"] = np.where(df_full_f["quant"] > df_full_f["quant"].quintile(0.80), 1, 0) # Create categorical test target outcome based on conditions
df_full_f.info() # Get class, memory, and column info: names, data types, obs.

### Create outcome table
df_XY_f = df_full_f.rename(columns = {'FIPS': 'ID'}) # Apply standard name to identifier used for joining datasets
df_Y_f = df_XY_f.filter(['quant', 'ID']) # Create Outcome table
df_Y_f = df_Y_f.set_index('ID') # Set identifier as index
df_Y_f.info() # Get class, memory, and column info: names, data types, obs.

### Create standard scaled predictor table
df_X_2 = df_XY_2.drop(columns = ['quant', 'ID']) # Drop Unwanted Columns
df_X_2 = df_X_2.replace([np.inf, -np.inf], np.nan) # Replace infitite values with NA
df_X_2 = df_X_2.dropna(axis = 1, thresh = 0.75*len(df_X_2)) # Drop features less than 75% non-NA count for all columns
df_X_2 = pd.DataFrame(SimpleImputer(strategy = 'median').fit_transform(df_X_2), columns = df_X_2.columns) # Impute missing data
df_X_2 = pd.DataFrame(StandardScaler().fit_transform(df_X_2.values), columns = df_X_2.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_X_2['ID'] = df_XY_2['ID'] # Save ID as column in predictor table
df_X_2 = df_X_2.set_index('ID') # Set identifier as index
df_X_2.info() # Get class, memory, and column info: names, data types, obs.

### Save second geographic layer features as list
mrfractureproofswoodcarvings = df_XY_f[mrfractureproof].columns.to_list() #
mrfractureproofswoodcarvings.append(mrfractureproof) #
mrfractureproofswoodcarvings.append("quant") #

### Create Multiple Regression Model
df_mrfpwc = df_XY_f[mrfractureproofswoodcarvings]
df_mrfpwc = df_mrfpwc.dropna()
X = df_mrfpwc.drop(columns = ['quant'])
Y = df_mrfpwc['quant']
mod = sm.OLS(Y, X) # Create linear model
res = mod.fit() # Fit model to create result
res.summary() # Print results of regression model

### Add feature labels
df_label = pd.concat([df_l1, df_l2, df_l3, df_l4]) # Combine rows with same columns
df_label = df_label.filter(["Feature", "Label"]) # Keep only selected columns
df_label = df_label.set_index("Feature") # Set column as index
df_label = df_label.transpose() # Switch rows and columns
mrfractureproofswoodcarvings.remove("quant") # Add outcome to regression dataset
df_label = df_label[mrfractureproofswoodcarvings] # Save chosen featres as list
df_label = df_label.transpose() # Switch rows and columns
df_label = df_label.reset_index() # Reset index
l_label = list(zip(df_label["Feature"], df_label["Label"])) # Create list of variables alongside RFE value 
df_label.info() # Get class, memory, and column info: names, data types, obs.

### Append to Text File
text_file = open(path + name + "_" + day + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(s6 + "\n\n") # Line of text with space after
text_file.write("Models: " + m4 + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(res.summary())  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(l_label)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

## Step 7: Predict Categorical targets with Artificial Neural Networks
s7 = 'Step 7: Predict Categorical targets with Artificial Neural Networks'
m6 = 'Multi-Layer Perceptron with Stacked Convolutional Autoencoder'

### Build Network with keras Sequential API for all features from all layers
# Prep Inputs
Y_train = df_X_f.filter(["train"])
Y_test = df_X_f.filter(["test"])
X = df_X_f.drop(columns = ["quant", "train", "test"])
input = X.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
network = Sequential()
# Dense Layers
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
# Activation Layer
network.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
final = network.fit(X, Y_train, batch_size = 10, epochs = 200) # Fitting the data to the train outcome
# Predict
Y_f = network.predict(X) # Predict values from testing model
# AUC Score
Y_pred = (Y_a > 0.5)
Y_train = (Y_train > 0)
Y_test = (Y_test > 0)
fpr, tpr, threshold = roc_curve(Y_train, Y_pred) # Create ROC outputs, true positive rate and false positive rate
a_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
a_test = auc(fpr, tpr) # Plot ROC and get AUC score

### Build Network with keras Sequential API for selected features from all layers
# Prep Inputs
Y_train = df_X_f.filter(["train"])
Y_test = df_X_f.filter(["test"])
X = df_X_f[mrfractureproofswoodcarvings]
input = X.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
network = Sequential()
# Dense Layers
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
# Activation Layer
network.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
final = network.fit(X, Y_train, batch_size = 10, epochs = 200) # Fitting the data to the train outcome
# Predict
Y_s = network.predict(X) # Predict values from testing model
# AUC Score
Y_pred = (Y_s > 0.5)
Y_train = (Y_train > 0)
Y_test = (Y_test > 0)
fpr, tpr, threshold = roc_curve(Y_train, Y_pred) # Create ROC outputs, true positive rate and false positive rate
s_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
s_test = auc(fpr, tpr) # Plot ROC and get AUC score

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(s7 + "\n\n") # Line of text with space after
text_file.write(m6 + "\n") # Add two lines of blank text at end of every section text
text_file.write("Layers = CVN, Dense, Dense, Activation" + "\n") # Add two lines of blank text at end of every section text
text_file.write("Functions = ReLU, ReLU, Sigmoid" + "\n") # Add two lines of blank text at end of every section text
text_file.write("Epochs = 200" + "\n") # Add two lines of blank text at end of every section text
text_file.write("Targets = (train, test), (95th percentile, 80th percentile)" + "\n\n")
text_file.write("AUC Scores (selected features, all layers)" + "\n") # Add two lines of blank text at end of every section text
text_file.write("train = " + str(a_train) + "\n") # Add two lines of blank text at end of every section text
text_file.write("test = " + str(a_test) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("AUC Scores (all features, all layers)" + "\n") # Add two lines of blank text at end of every section text
text_file.write("train = " + str(s_train) + "\n") # Add two lines of blank text at end of every section text
text_file.write("test = " + str(s_test) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file