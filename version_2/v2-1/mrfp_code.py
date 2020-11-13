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
import scipy.stats as st # Statistics package best for t-test, ChiSq, correlation
import statsmodels.api as sm # Statistics package best for regression models

### Import Visualization Libraries
import matplotlib.pyplot as plt # Comprehensive graphing package in python
import geopandas as gp # Simple mapping library for csv shape files with pandas like syntax for creating plots using matplotlib 

### Import scikit-learn libraries
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest regression component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.svm import LinearSVC # Linear Support Vector Classification from sklearn
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome
from sklearn.linear_model import LogisticRegression # Used for machine learning with quantitative outcome
from sklearn.model_selection import train_test_split # train test split function for validation
from sklearn.metrics import roc_curve # Reciever operator curve
from sklearn.metrics import auc # Area under the curve 

### Import PySAL Libraries
import libpysal as ps # Spatial data science modeling tools in python
from mgwr.gwr import GWR, MGWR # Geographic weighted regression modeling tools
from mgwr.sel_bw import Sel_BW # Bandwidth selection for GWR

### Import keras libraries
from keras.models import Sequential # Uses a simple method for building layers in MLPs
from keras.models import Model # Uses a more complex method for building layers in deeper networks
from keras.layers import Dense # Used for creating dense fully connected layers
from keras.layers import Input # Used for designating input layers

### Set Directory
os.chdir(directory) # Set wd to project repository

### Set Timestamps
day = str(date.today())
stamp = str(datetime.now())

### Write corresponding text file for collecting results
text_file = open(path + name + '_' + day + '.txt', 'w') # Write new corresponding text file
text_file.write('####################' + '\n\n') # Add section break for end of step
text_file.write('Title: ' + title + '\n') # Script title
text_file.write('Author: ' + author + '\n') # Script Author
text_file.write('Filename: ' + name + '.py' + '\n') # Filename of script
text_file.write('Realtive Path: ' + path + '\n') # Relative path to script
text_file.write('Working Directory: ' + directory + '\n') # Directory used for script run
text_file.write('Time Run: ' + stamp + '\n') # Timestamp of script run
text_file.write('\n' + '####################' + '\n\n') # Add section break for end of step
text_file.close() # Close file

## Step 1: Data Processing of Predictors and Outcomes
s1 = 'Step 1: Raw Data Processing and Feature Engineering' # Step 1 descriptive title
d1 = 'Florida Deaprtment of Health Vital Statistics 113 Leading Mortality Causes 2014-2018 Zip Code 5-year Average' # Dataset 1 descriptive title
d2 = 'US Census American Community Survey 2014-2018 Zip Code 5-year Average' # Dataset 2 descriptive title

### Preprocess First Dataset
df_d1 = pd.read_csv('fracture-proof/version_2/_data/FDOH_5Y2018_ZCTA.csv') # Import first dataset saved as csv in _data folder
df_d1 = df_d1[df_d1['POPULATION'] > 500] # Susbet numeric column by condition
df_d1 = df_d1.filter(['K00_K99_R1000', 'ZCTA']) # Drop or filter columns to keep only feature values and idenitifer
df_d1 = df_d1.rename(columns = {'ZCTA': 'ID', 'K00_K99_R1000': 'quant'}) # Apply standard name to identifier and quantitative outcome
df_d1.info() # Get class, memory, and column info: names, data types, obs

### Preprocess Second Dataset
df_d2 = pd.read_csv('fracture-proof/version_2/_data/ACS_5Y2018_ZCTA.csv') # Import second dataset saved as csv in _data folder
df_d2 = df_d2.drop(columns = ['ST', 'FIPS']) # Drop or filter columns to keep only feature values and idenitifer
df_d2 = df_d2.select_dtypes(exclude = ['int64']) # Drop all unwanted data types
df_d2 = df_d2.rename(columns = {'ZCTA': 'ID'}) # Apply standard name to identifier used for joining datasets
df_d2.info() # Get class, memory, and column info: names, data types, obs.

### Join Datasets by ID and define targets
df_XY = pd.merge(df_d1, df_d2, on = 'ID', how = 'inner') # Join datasets to create table with predictors and outcome
df_XY = df_XY.dropna(subset = ['quant']) # Drop all outcome rows with NA values
df_XY.info() # Get class, memory, and column info: names, data types, obs.

### Create outcome table
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

### Add feature labels
df_l1 = pd.read_csv('fracture-proof/version_2/_data/ACS_5Y2018_labels.csv') # Import feature lables for first dataset saved as csv in _data folder
df_l2 = pd.read_csv('fracture-proof/version_2/_data/FDOH_5Y2018_labels.csv') # Import feature lables for second dataset saved as csv in _data folder
df_l1_l2 = pd.concat([df_l1, df_l2]) # Combine rows with same columns
df_l1_l2 = df_l1_l2.filter(['Feature', 'Label']) # Keep only selected columns
df_l1_l2 = df_l1_l2.set_index('Feature') # Set column as index
df_l1_l2 = df_l1_l2.transpose() # Switch rows and columns
df_l1_l2.info # Get class, memory, and column info: names, data types, obs.

### Append step 1 results to corresponding text file
text_file = open(path + name + '_' + day + '.txt', 'a') # Open corresponding text file
text_file.write(s1 + '\n\n') # Step description
text_file.write(d1 + '\n') # Dataset description
text_file.write(d2 + '\n\n') # Dataset description
text_file.write('Target labels: quant = Diabetes Related (K00-K99) Raw Mortality Rate per 1000k' + '\n') # Dataset methods description
text_file.write('Target processing: None' + '\n\n') # Dataset methods description
text_file.write(str(df_Y.describe())  + '\n\n') # Result descriptive statistics for target
text_file.write('Features labels: ACS Percent Estimates' + '\n') # Result description
text_file.write('Feature processing: 75% nonNA, Median Imputed NA, Standard Scaled' + '\n\n') # Dataset methods description
text_file.write('Rows, Columns: ' + str(df_X.shape) + '\n\n') # Result description and result dataframe
text_file.write('####################' + '\n\n') # Add section break for end of step
text_file.close() # Close file

## Step 2: Identify Predictors
s2 = "Step 2: Identify Predictors with Open Models" # Step 2 descriptive title
m1 = "Principal Component Analysis" # Model 1 descriptive title
m2 = "Random Forests" # Model 2 descriptive title
m3 = "Recursive feature Elimination" # Model 3 descriptive title

### Principal Component Analysis
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
forest.fit(df_X, df_Y['quant']) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(df_X, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ['Feature', 'Gini']) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf['Gini'] > df_rf['Gini'].mean())] # Subset by Gini values higher than mean
df_rf = df_rf.sort_values(by = ['Gini'], ascending = False) # Sort Columns by Value
df_rf.info() # Get class, memory, and column info: names, data types, obs.

### Fracture: Join RF and PCA 
df_fr = pd.merge(df_pca, df_rf, on = 'Feature', how = 'inner') # Join by column while keeping only items that exist in both, select outer or left for other options
fracture = df_fr['Feature'].tolist() # Save features from data frame
df_fr.info() # Get class, memory, and column info: names, data types, obs.

### Recursive Feature Elimination
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(df_X[fracture], df_Y['quant']) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(df_X[fracture], rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ['Feature', 'RFE']) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe.sort_values(by = ['RFE'], ascending = True) # Sort Columns by Value
df_rfe = df_rfe[df_rfe['RFE'] == True] # Select Variables that were True
df_rfe.info() # Get class, memory, and column info: names, data types, obs.

### FractureProof: Join RFE with Fracture
df_fp = pd.merge(df_fr, df_rfe, on = 'Feature', how = 'inner') # Join by column while keeping only items that exist in both, select outer or left for other options
fractureproof = df_fp['Feature'].tolist() # Save chosen featres as list
df_fp.info() # Get class, memory, and column info: names, data types, obs.

### Get FractureProof feature labels
df_lfp = df_l1_l2[fractureproof] # Save chosen featres as list
df_lfp = df_lfp.transpose() # Switch rows and columns
df_lfp = df_lfp.reset_index() # Reset index
l_lfp = list(zip(df_lfp['Feature'], df_lfp['Label'])) # Create list of variables alongside RFE value 
df_lfp.info() # Get class, memory, and column info: names, data types, obs.

### Append step 2 results to corresponding text file
text_file = open(path + name + '_' + day + '.txt', 'a') # Open corresponding text file
text_file.write(s2 + '\n\n') # Step description
text_file.write('Models: ' + m1 + ', ' + m2 + ', ' + m3 + '\n\n') # Model description
text_file.write('Values: Eigenvectors, Gini Impurity, Boolean' + '\n') # Model methods description
text_file.write('Thresholds: Mean, Mean, Cross Validation' + '\n\n') # Model methods description
text_file.write(str(df_fp)  + '\n\n') # Result dataframe
text_file.write("Final list of selected features" + "\n") # Result description
text_file.write(str(l_lfp)  + '\n\n') # Result list
text_file.write('####################' + '\n\n') # Add section break for end of step
text_file.close() # Close file

## Step 3: Create Informative Prediction Model
s3 = 'Step 3: Create Informative Preidction Model' # Step 3 descriptive title
m4 = 'Multiple Linear Regression Model' # Model 4 descriptive title

### Add confounders to multiple regression model
mrfractureproof = df_X[fractureproof].columns.to_list() # Save list of selected variables for multiple regression model
mrfractureproof.append('quant') # Add outcome to list of selected variables for multiple regression model
mrfractureproof.append('DP05_0024PE') # Add confounder (Over 65) to list of selected variables for multiple regression model

### Create Multiple Regression Model
df_mrfp = df_XY[mrfractureproof] # Subset original nonscaled data for regression
df_mrfp = df_mrfp.dropna() # Drop all rows with NA values
X = df_mrfp.drop(columns = ['quant']) # Susbet predictors for regression
Y = df_mrfp['quant'] # Subset quantitative outcome for regression
mod = sm.OLS(Y, X) # Create linear regression model
res = mod.fit() # Fit model to create result
res.summary() # Print results of regression model

### Add feature labels
mrfractureproof.remove('quant') # Remove outcome to list of features used for collecting lables
df_lmrfp = df_l1_l2[mrfractureproof] # Save selected features as list for collecting labels
mrfractureproof.append('quant') # Add outcome to to list of selected variables for multiple regression model
df_lmrfp = df_lmrfp.transpose() # Switch rows and columns
df_lmrfp = df_lmrfp.reset_index() # Reset index
l_lmrfp = list(zip(df_lmrfp['Feature'], df_lmrfp['Label'])) # Create list of variables alongside RFE value 
df_lmrfp.info() # Get class, memory, and column info: names, data types, obs.

### Append step 3 results to corresponding text file
text_file = open(path + name + '_' + day + '.txt', 'a') # Open corresponding text file 
text_file.write(s3 + '\n\n') # Step title
text_file.write('Models: ' + m4 + '\n\n') # Model description
text_file.write(str(res.summary())  + '\n\n') # Result summary
text_file.write(str(l_lmrfp)  + '\n\n') # Result list
text_file.write('####################' + '\n\n') # Add section break for end of step
text_file.close() # Close file