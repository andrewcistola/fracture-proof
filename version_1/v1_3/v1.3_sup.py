File = "CMS_FL_v1.3_sup"
path = "fp/VBHC/ADJ/delta/"
title = "FractureProof Final Payment Adjustments from CMS with Hopsital Predictorsamong Florida Hospitals"
author = "DrewC!"

### Import FractureProof Libraries
import os # Operating system navigation
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes
import statsmodels.api as sm # Statistics package best for regression models for statistical tests
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
from sklearn.cluster import KMeans # clusters data by trying to separate samples in n groups of equal variance
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest regression component
from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome
from sklearn.linear_model import LogisticRegression # Used for machine learning with quantitative outcome
from sklearn.metrics import roc_curve # Reciever operator curve
from sklearn.metrics import auc # Area under the curve 
from keras import Sequential # Sequential neural network modeling
from keras.layers import Dense # Used for creating layers within neural network

### Set Directory
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

### Import Data and Rename
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_key = pd.read_csv("hnb/FIPS/FIPS_ZCTA_key.csv") # Import dataset saved as csv in _data folder
df_key = df_key.filter(["FIPS", "ST"]) # Keep only selected columns
df_key = df_key.drop_duplicates(keep = "first", inplace = False) # Drop all dupliacted values
df_cms = pd.merge(df_key, df_cms, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_cms = df_cms[df_cms["ST"] == "FL"] # Susbet numeric column by condition
df_cms = df_cms.drop(columns = ["ST"]) # Drop Unwanted Columns
df_cms = df_cms.rename(columns = {"2020 VBP Adjustment Factor": "quant"}) # Rename quantitative outcome
df_cms = df_cms.drop(columns = ["Facility ID", "FIPS"]) # Drop ID variables
df_cms = df_cms.drop(columns = ["2018 VBP Adjustment Factor", "2019 VBP Adjustment Factor"]) # Drop proximity features: Adjustment factor scores
df_cms = df_cms.drop(columns = ["Total Performance Score", "Weighted Normalized Clinical Outcomes Domain Score", "Weighted Safety Domain Score", "Weighted Person and Community Engagement Domain Score", "Weighted Efficiency and Cost Reduction Domain Score"]) # Drop proximity features: Adjustment factor scores
df_cms.info() # Get class, memory, and column info: names, data types, obs.

### Data Pre-processing
df_prep = df_cms.dropna(subset = ["quant"])
df_prep = df_prep.reset_index() # Reset Index
df_prep = df_prep.drop(columns = ["index"]) # Drop Unwanted Columns
df_hold = df_prep.filter(["quant"])
df_prep = df_prep.drop(columns = ["quant"]) # Drop ID variables
df_prep = df_prep.dropna(axis = 1, thresh = 0.75*len(df_prep)) # Drop features less than 75% non-NA count for all columns
df_prep = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_prep), columns = df_prep.columns) # Impute missing data
df_prep = pd.DataFrame(StandardScaler().fit_transform(df_prep.values), columns = df_prep.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_prep["quant"] = df_hold["quant"]
df_prep = df_prep.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_prep.info() # Get class, memory, and column info: names, data types, obs.

### Principal Component Analysis
df_pca = df_prep.drop(columns = ["quant"]) # Drop outcomes and targets
degree = len(df_pca.columns) - 2 # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(df_pca) # Fit initial PCA model
df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
components = len(df_comp.index) - 3 # Save count of components for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(df_pca) # finally call fit_transform on the aggregate data to create PCA results object
df_pc = pd.DataFrame(pca.components_, columns = df_pca.columns) # Export eigenvectors to data frame with column names from original data
df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pc = df_pc.abs() # Get absolute value of eigenvalues
df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
df_p = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_p = df_p[df_p.MaxEV > df_p.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_p = df_p.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
df_pca = df_p.rename(columns = {"index": "Features"}) # Rename former index as features
df_pca = df_pca.sort_values(by = ["MaxEV"], ascending = False) # Sort Columns by Value
print(df_pca)

### Random Forest Regressor
X = df_prep.drop(columns = ["quant"]) # Drop outcomes and targets
Y = df_prep["quant"] # Isolate Outcome variable
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X, Y) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean
df_rf = df_rf.sort_values(by = ["Gini"], ascending = False) # Sort Columns by Value
print(df_rf)

### Recursive Feature Elimination
df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
X = df_prep[pca_rf] # Save features columns as predictor data frame
Y = df_prep["quant"] # Selected quantitative outcome from original data frame
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X, Y) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
print(df_rfe)

### Multiple Regression
pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
X = df_prep.filter(pca_rf_rfe) # Keep only selected columns from rfe
Y = df_prep["quant"] # Add outcome variable
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X, Y) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X, coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names
df_reg = df_reg.sort_values(by = ["Coefficients"], ascending = False) # Sort Columns by Value
print(df_reg)

### Export Results to Text File
df_final = pd.merge(df_pca_rf, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
text_file = open(path + File + "_results.txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write(title) # Line of text with space after
text_file.write("\nCMS Quality Metrics by Hospital in FLorida\n") # Line of text with space after
text_file.write(str(df_cms.shape)) # Line of text with space after
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.write("\nSignificant Quality Metrics\n") # Line of text with space after
text_file.write(str(df_final["Features"].tolist())) # Line of text with space after
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.write("\n") # Line of text with space after
text_file.write(str(df_final)) # Line of text with space after
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file