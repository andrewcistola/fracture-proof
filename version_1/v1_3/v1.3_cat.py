# FractureProof v1.3
## Value Based Healthcare Reimbursements

### Import Python Libraries
import os # Operating system navigation
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
from sklearn.cluster import KMeans # clusters data by trying to separate samples in n groups of equal variance
from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.linear_model import LogisticRegression # Used for machine learning with categorical outcome
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome

### Setup Directory and Title
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository
title = "ACS_v1.3_un"
path = "fp/VBHC/gamma/"

## Section A: Collect Possible Predictors from Public Access Data

### Import Data
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_cms = df_cms.filter(["Facility ID", "FIPS"]) # Keep only selected columns
df_join = pd.merge(df_cms, df_acs, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_join = df_join.drop(columns = ["FIPS"]) # Drop Unwanted Columns
df_km = df_join
df_cms = 0 # Clear variable
df_acs = 0 # Clear variable

### K-Means Unsupervised Clustering
ID = df_km.pop("Facility ID") # Remove quantitative outcome
df_km = df_km.dropna(axis = 1, thresh = 0.75*len(df_km)) # Drop features less than 75% non-NA count for all columns
df_km = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_km), columns = df_km.columns) # Impute missing data
df_km = pd.DataFrame(StandardScaler().fit_transform(df_km.values), columns = df_km.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_km = df_km.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
kmeans = KMeans(n_clusters = 5, random_state = 0) # Setup Kmeans model, pre-select number of clusters
kmeans.fit(df_km) # Fit Kmeans
km = kmeans.labels_ # Output importances of features
l_km = list(zip(ID, km)) # Create list of variables alongside importance scores 
df_cl = pd.DataFrame(l_km, columns = ["Facility ID", "Cluster"]) # Create data frame of importances with variables and gini column names
df_km.insert(0, "Facility ID", ID) # reinsert in index
df_km = pd.merge(df_cl, df_km, on = "Facility ID", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options

### Create Dummy Variables
df_km["Cluster1"] = np.where(df_km["Cluster"] == 1, 1, 0) # Create New Column Based on Conditions
df_km["Cluster2"] = np.where(df_km["Cluster"] == 2, 1, 0) # Create New Column Based on Conditions
df_km["Cluster3"] = np.where(df_km["Cluster"] == 3, 1, 0) # Create New Column Based on Conditions
df_km["Cluster4"] = np.where(df_km["Cluster"] == 4, 1, 0) # Create New Column Based on Conditions

### Rename
df_clusters = df_km

## Cluster 1

### Cluster 1: Prepare Data - Run before FP
df_c1 = df_clusters.drop(columns = ["Facility ID", "Cluster", "Cluster2", "Cluster3", "Cluster4"]) # Drop outcomes and targets
df_c1 = df_c1.rename(columns = {"Cluster1": "cat"}) # Rename multiple columns in place
df_c1 = df_c1.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_prep = df_c1

### Cluster 1: Export Results - Run After FP
df_final = pd.merge(df_pca_rf, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
final = df_final["Features"].tolist() # Save chosen featres as list
print(df_final) # Show in terminal
df_final.to_csv(path + title + "_c1.csv") # Export df as csv

## Cluster 2

### Cluster 2: Prepare Data - Run before FP
df_c2 = df_clusters.drop(columns = ["Facility ID", "Cluster1", "Cluster", "Cluster3", "Cluster4"]) # Drop outcomes and targets
df_c2 = df_c2.rename(columns = {"Cluster2": "cat"}) # Rename multiple columns in place
df_c2 = df_c2.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_prep = df_c2

### Cluster 2: Export Results - Run After FP
df_final = pd.merge(df_pca_rf, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
final = df_final["Features"].tolist() # Save chosen featres as list
print(df_final) # Show in terminal
df_final.to_csv(path + title + "_c2.csv") # Export df as csv

## Cluster 3

### Cluster 3: Prepare Data - Run before FP
df_c3 = df_clusters.drop(columns = ["Facility ID", "Cluster1", "Cluster2", "Cluster", "Cluster4"]) # Drop outcomes and targets
df_c3 = df_c3.rename(columns = {"Cluster3": "cat"}) # Rename multiple columns in place
df_c3 = df_c3.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_prep = df_c3

### Cluster 3: Export Results - Run After FP
df_final = pd.merge(df_pca_rf, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
final = df_final["Features"].tolist() # Save chosen featres as list
print(df_final) # Show in terminal
df_final.to_csv(path + title + "_c3.csv") # Export df as csv

## Cluster 4

### Cluster 4: Prepare Data - Run before FP
df_c4 = df_clusters.drop(columns = ["Facility ID", "Cluster1", "Cluster2", "Cluster3", "Cluster"]) # Drop outcomes and targets
df_c4 = df_c4.rename(columns = {"Cluster4": "cat"}) # Rename multiple columns in place
df_c4 = df_c4.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_prep = df_c4

### Cluster 4: Export Results - Run After FP
df_final = pd.merge(df_pca_rf, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
final = df_final["Features"].tolist() # Save chosen featres as list
print(df_final) # Show in terminal
df_final.to_csv(path + title + "_c4.csv") # Export df as csv

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

