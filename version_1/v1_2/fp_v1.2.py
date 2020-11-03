# FractureProof
## Regression code template
## Version: Alpha

### Set working directory to local folder
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

title = "fp_REGR_alpha"
path = "fp/REGR/"

## Section A: Collect Possible Predictors from Public Access Data

### Import Python Libraries
import os # Operating system navigation
import sqlite3 # SQLite database manager

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes

### Import scikit-learn libraries: data preparation 
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data

### Step 1: Import and Join Data

### Import CMS Data
df_cms = pd.read_csv("hnb/.csv", low_memory = 'false') # Import dataset saved as csv in _data folder

### Rename and Verify
df_step1 = df_cms
df_cms = 0
df_step1.info() # Get class, memory, and column info: names, data types, obs.
df_step1.head() # Print first 5 observations

### Step 2: Data Manipulation

### Drop ID variables
df_man = df_step1.drop(columns = ["FIPS"]) # Drop Unwanted Columns

### Drop proximity variables
df_man = df_man.drop(columns = [""]) # Drop Unwanted Columns

### Rename outcome and test
df_man = df_man.rename(columns = {"": "outcome"}) # Rename multiple columns in place

### Rename and Verify
df_step2 = df_man
df_man = 0
df_step2.info() # Get class, memory, and column info: names, data types, obs.
df_step2.head() # Print first 5 observations

## Step 3: Data Standardization

### Remove outcome and test
df_NA = df_step2
outcome = df_NA.pop("outcome") # 'pop' column from df

### Drop features with less than 75% data
df_NA = df_NA.dropna(axis = 1, thresh = 0.75*len(df_NA)) # Drop features less than 75% non-NA count for all columns

### Impute missing values
df_NA = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_NA), columns = df_NA.columns) # Impute missing data

### Standard Scale Values
df_NA = pd.DataFrame(StandardScaler().fit_transform(df_NA.values), columns = df_NA.columns) # convert the normalized features into a tabular format with the help of DataFrame.

### Reattach outcome
df_NA.insert(0, "outcome", outcome) # reinsert in index

### Drop all remaining rows (should be none)
df_NA = df_NA.dropna() # Drop all rows with NA values

### Rename and Verify
df_step3 = df_NA
df_NA = 0
df_step3.info() # Get class, memory, and column info: names, data types, obs.
df_step3.head() # Print first 5 observations

## Section B: Identify Significant Predictors with Reduction Algorithms

### Import scikit-learn: machine learning
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
from sklearn.ensemble import RandomForestRegressor # Random Forest classification component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome

### Step 4: Principal Component Analysis

### Setup initial PCA model
df_pca = df_step3.drop(columns = ["outcome"]) # Drop outcome variable
degree = len(df_step3.columns) - 2 # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # you will pass the number of components to make PCA model based on degrees of freedom

### Fit initial PCA model
pca.fit(df_pca) # fit to data

### Setup final PCA model
df_ev = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_ev = df_ev[(df_ev[0] > 1)] # Save eigenvalues above 1
components = len(df_ev.index) # Save count of values for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model

### Fit final PCA model
pca.fit_transform(df_pca) # finally call fit_transform on the aggregate data to create PCA results object

### Collect feature list from PCA
df_pca2 = pd.DataFrame(pca.components_, columns = df_pca.columns) # Export eigenvectors to data frame
df_pca2["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pca2 = df_pca2[df_pca2.Variance > df_pca2.Variance.mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pca2 = df_pca2.abs() # get absolute value for column or data frame
df_pca3 = pd.DataFrame(df_pca2.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_pc = df_pca3[df_pca3.MaxEV > df_pca3.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_pc = df_pc.reset_index() # Add a new index of ascending values, existing index becomes column named "index"
df_pc =  df_pc.rename(columns = {"index": "Features"}) # Rename multiple columns in place

### Rename and Verify
df_step4 = df_pc
df_step4.info() # Get class, memory, and column info: names, data types, obs.
df_step4.head() # Print first 5 observations

### Step 5: Random Forest Regressor

### Setup RF model
Y = df_step3["outcome"] # Isolate Outcome variable
X = df_step3.drop(columns = ["outcome"]) # Drop Unwanted Columns # Save features columns as predictor data frame
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 

### Fit Forest model
forest.fit(X, Y) # This will take time

### Collect features from RF
gini = forest.feature_importances_ # Output importances of features
l_gini = list(zip(X, gini)) # Create list of variables alongside importance scores 
df_gini = pd.DataFrame(l_gini, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_gini = df_gini.sort_values(by = ["Gini"], ascending = False) # Sort data frame by gini value in desceding order
df_gini = df_gini[(df_gini["Gini"] > df_gini["Gini"].mean())] # Subset by Gini values higher than mean

### Rename and Verify
df_step5 = df_gini
df_step5.info() # Get class, memory, and column info: names, data types, obs.
df_step5.head() # Print first 5 observations

### Step 6: Recursive Feature Elimination

### Collect features from RF and PC
df_pc_gini = pd.merge(df_pc, df_gini, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
pc_gini_features = df_pc_gini["Features"].tolist() # Save features from data frame
df_rfecv = df_step3[pc_gini_features] # Add selected features to df

### Setup RFE model
X = df_rfecv # Save features columns as predictor data frame
Y = df_step3["outcome"] # Use outcome data frame 
RFE = LinearRegression() # Use regression coefficient as estimator
selector = RFECV(estimator = RFE, min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo

### Fit RFE model
selected = selector.fit(X, Y) # This will take time

### Collect features from RFE model
ar_rfe = selected.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, ar_rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
df_rfe = df_rfe.reset_index() # Reset Index
df_rfe = df_rfe.filter(["Features"]) # Keep only selected columns

### Rename and Verify
df_step6 = df_rfe
df_step6.info() # Get class, memory, and column info: names, data types, obs.
df_step6.head() # Print first 5 observations

## Section C: Evaluate Significant Features with Modeling and Prediction

### Import scikit-learn libraries: regression
from sklearn.linear_model import LogisticRegression # Used for machine learning with categorical outcome
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome

### Step 7: Multiple Regression

### Setup MR Model
features = list(df_step6["Features"]) # Save chosen featres as list
x = df_step3.filter(features) # Keep only selected columns from rfe
y = df_step3["outcome"] # Add outcome variable
LR = LinearRegression() # Linear Regression in scikit learn

### Fit MR model
regression = LR.fit(x, y) # Fit model

### Collect features from MR model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(x, coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names

### Export feature attributes
df_pc_gini_reg = pd.merge(df_pc_gini, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_pc_gini_reg.to_csv(r"fp/.csv") # Export df as csv
print(df_pc_gini_reg)

### Collect prediction results
determination = regression.score(x, y) # rsq value, ceofficient of determination
print(determination)

### Rename and Verify
df_step7 = df_pc_gini_reg
df_step7.info() # Get class, memory, and column info: names, data types, obs.
df_step7.head() # Print first 5 observations

