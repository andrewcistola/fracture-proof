File = "ACS_ZCTA_FL_v1.3_un"
path = "fp/VBHC/ADJ/delta/"
title = "FractureProof Final Payment Adjustments from CMS with ACS Zip Code Data in FLorida"
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

## Section A: Collect Possible Predictors from Public Access Data

### Import ACS Data and FIPS to join by ZCTA
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_doh = pd.read_csv("hnb/DOH/FL/113_5Y2018/FL_113_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_fl = pd.merge(df_acs, df_doh, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_fl.info() # Get class, memory, and column info: names, data types, obs.

### Process Data for KMeans
df_NA = df_fl.dropna(subset = ["FIPS"])
df_FIPS = df_NA.filter(["FIPS"])
df_NA = df_NA.drop(columns = ["FIPS"]) # Drop Unwanted Columns
df_NA = df_NA.dropna(axis = 1, thresh = 0.75*len(df_NA)) # Drop features less than 75% non-NA count for all columns
df_NA = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_NA), columns = df_NA.columns) # Impute missing data
df_NA = pd.DataFrame(StandardScaler().fit_transform(df_NA.values), columns = df_NA.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_NA["FIPS"] = df_FIPS["FIPS"]
df_NA = df_NA.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_NA.info() # Get class, memory, and column info: names, data types, obs.

### PCA to determine cluster count
df_comp = df_NA.drop(columns = ["FIPS"]) # Drop Unwanted Columns 
degree = len(df_comp.index) - 2 # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(df_comp) # Fit initial PCA model
df_pca = pd.DataFrame(pca.explained_variance_, columns = ["Eigenvalues"]) # Print explained variance of components
df_pca = df_pca.sort_values(by = ["Eigenvalues"], ascending = False) # Sort Columns by Value
print(df_pca) # Print value, select "elbow" to determine number of components

# K-Means Unsupervised Clustering
kmeans = KMeans(n_clusters = 3, random_state = 0) # Setup Kmeans model, pre-select number of clusters
kmeans.fit(df_comp) # Fit Kmeans
km = kmeans.labels_ # Output importances of features
l_km = list(zip(df_NA["FIPS"], km)) # Create list of variables alongside importance scores 
df_km = pd.DataFrame(l_km, columns = ["FIPS", "Cluster"]) # Create data frame of importances with variables and gini column names
df_km["Cluster"] = df_km["Cluster"] + 1 # Add 1 to cluster array since numpy array starts at zero
df_km = pd.merge(df_km, df_NA, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_km.info() # Get class, memory, and column info: names, data types, obs.

### Import CMS VBHC Outcomes
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_cms = df_cms.rename(columns = {"2020 VBP Adjustment Factor": "quant"}) # Rename quantitative outcome
df_cms["test"] = np.where(df_cms["quant"] < 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms = df_cms.filter(["FIPS", "quant", "test"]) # Keep only selected columns
df_cms = pd.merge(df_cms, df_km, on = "FIPS", how = "right") # Join by column while keeping only items that exist in both, select outer or left for other options
df_cms = df_cms.dropna() # Drop all rows with NA values
df_cms.info() # Get class, memory, and column info: names, data types, obs.

### Prep Clusters
df_cms = df_cms[df_cms["Cluster"] == 1] # Susbet numeric column by condition
Y = df_cms["quant"] # Isolate Outcome variable
X = df_cms.drop(columns = ["Cluster", "FIPS", "quant", "test"]) # Drop outcomes and targets
df_cms.info() # Get class, memory, and column info: names, data types, obs.

df_cms = df_cms[df_cms["Cluster"] == 2] # Susbet numeric column by condition
Y = df_cms["quant"] # Isolate Outcome variable
X = df_cms.drop(columns = ["Cluster", "FIPS", "quant", "test"]) # Drop outcomes and targets
df_cms.info() # Get class, memory, and column info: names, data types, obs.

df_cms = df_cms[df_cms["Cluster"] == 3] # Susbet numeric column by condition
Y = df_cms["quant"] # Isolate Outcome variable
X = df_cms.drop(columns = ["Cluster", "FIPS", "quant", "test"]) # Drop outcomes and targets
df_cms.info() # Get class, memory, and column info: names, data types, obs.

### Run Random Forest
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X, Y) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].quantile(0.9))] # Subset by Gini values higher than mean
df_rf.info() # Get class, memory, and column info: names, data types, obs.

### Recursive Feature Elimination
features = df_rf["Features"].tolist() # Save features from data frame
X = df_cms[features] # Selected quantitative outcome from original data frame
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5, cv = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X, Y) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
df_rfe.info() # Get class, memory, and column info: names, data types, obs.

### Multiple Regression
features = df_rfe["Features"].tolist() # Save chosen featres as list
X = df_cms.filter(features) # Keep only selected columns from rfe
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X, Y) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X, coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names
df_reg = df_reg.sort_values(by = ["Coefficients"], ascending = False) # Sort Columns by Value
df_reg.info() # Get class, memory, and column info: names, data types, obs.

### Examing Clusters
F1 = df_reg # Rename df
print(F1) # Print value, select "elbow" to determine number of components

F2 = df_reg # Rename df
print(F2) # Print value, select "elbow" to determine number of components

F3 = df_reg # Rename df
print(F3) # Print value, select "elbow" to determine number of components

### Export Results to Text File
text_file = open(path + File + "_results.txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write(title) # Line of text with space after
text_file.write("\nACS Predictors by Zip Code in FL\n") # Line of text with space after
text_file.write(str(df_fl.shape)) # Line of text with space after
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.write("\nIdentified Clusters\n") # Line of text with space after
text_file.write(str(df_km["Cluster"].describe())) # Line of text with space after
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.write("\nCluster 1 - Significant Predictors\n") # Line of text with space after
text_file.write(str(F1)) # Line of text with space after
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.write("\nCluster 2 - Significant Predictors\n") # Line of text with space after
text_file.write(str(F2)) # Line of text with space after
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.write("\nCluster 3 - Significant Predictors\n") # Line of text with space after
text_file.write(str(F3)) # Line of text with space after
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file
















## FractureProof v1.3 - Categorical 

### Principal Component Analysis
df_pca = df_prep.drop(columns = ["cat"]) # Drop outcomes and targets
degree = len(df_pca.index) - 2 # Save number of features -1 to get degrees of freedom
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

### Random Forest Regressor
X = df_prep.drop(columns = ["cat"]) # Drop outcomes and targets
Y = df_prep["cat"] # Isolate Outcome variable
forest = RandomForestClassifier(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X, Y) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean

### Recursive Feature Elimination
df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
X = df_prep[pca_rf] # Save features columns as predictor data frame
Y = df_prep["cat"] # Selected quantitative outcome from original data frame
recursive = RFECV(estimator = LogisticRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X, Y) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True

### Multiple Regression
pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
X = df_prep[pca_rf_rfe] # Keep only selected columns from rfe
Y = df_prep["cat"] # Add outcome variable
regression = LogisticRegression() # Linear Regression in scikit learn
regression.fit(X, Y) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X, coef.T)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names

