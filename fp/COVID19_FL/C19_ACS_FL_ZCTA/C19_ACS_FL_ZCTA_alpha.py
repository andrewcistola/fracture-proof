# FractureProof
# ACS 2014-2018 Zip Code Percent Estimates for 50 States
# Florida DOH 2014-2018 Zip Code 113 Causes of Death

# Section A: Generate Hypothesis with Machine Learning Algorithms

## Step 1-2: Import Libraries and Import Dataset

### Import Python Libraries
import os # Operating system navigation

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes

### Import scikit-learn libraries: data preparation 
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
from sklearn.model_selection import train_test_split # Randomized test-train splitter for evaluating prediction

### Import scikit-learn: machine learning
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
from sklearn.ensemble import RandomForestRegressor # Random Forest classification component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation

### Import scikit-learn: neural network
from sklearn.neural_network import MLPRegressor

### Import scikit-learn libraries: regression
from sklearn.linear_model import LogisticRegression # Used for machine learning with categorical outcome
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome

### Import statistics libraries
import statsmodels.api as sm # Statistics package best for regression models for statistical tests
import scipy.stats as st # Statistics package best for t-test, ANOVA, ChiSq

### Import visualization libraries
import matplotlib.pyplot as plt # Comprehensive graphing package in python
import geopandas as gp # Simple mapping library for csv shape files with pandas like syntax for creating plots using matplotlib 

### Set working directory to subproject folder
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

### Import Datasets
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_PE_US_NA_stage.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository
df_fl = pd.read_csv("hnb/DOH/FL/DeathsReport_5Y2018/FL_113_rates_stage.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository
df_c19 = pd.read_csv("hnb/DOH/FL/COVID_ZCTA/July 20 2020/Florida_Cases_Zips_COVID19.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository
df_pop = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_pop.csv") # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository
df_label = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_label.csv") # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository
df_label2 = pd.read_csv("hnb/DOH/FL/DeathsReport_5Y2018/FL_113_labels.csv") # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository

### Clean C19 data and calculate population adjusted rate
df_c19['ZCTA'] = df_c19['ZIP'].astype("str") # Change data type of column in data frame
df_c19['ZCTA'] = df_c19['ZCTA'].str.rjust(5, "0") # add leading zeros of character column using rjust() function
df_c19['ZCTA'] = "ZCTA"+ df_c19['ZCTA'] # Combine string with column
df_c19['Cases_1'] = df_c19['Cases_1'].str.replace('<', '')
df_c19['C19_Cases'] = df_c19['Cases_1'].astype("int64") # Change data type of column in data frame
df_c19 = df_c19.filter(["ZCTA", "C19_Cases"]) # Keep only selected columns
df_c19 = df_c19.groupby(["ZCTA"], as_index = False).sum() # Group data by columns and sum
df_c19 = pd.merge(df_pop, df_c19, on = "ZCTA", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_c19["C19_Cases_R1000"] = df_c19["C19_Cases"] / df_c19["POPULATION"] * 1000
df_c19 = df_c19.filter(["ZCTA", "C19_Cases_R1000"]) # Keep only selected columns

### Join, reindex, drop columns
df_join = pd.merge(df_fl, df_acs, on = "ZCTA", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_join2 = pd.merge(df_join, df_c19, on = "ZCTA", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_index = df_join2.set_index("ZCTA") # Set column as index
df_drop = df_index.drop(columns = ["PERCENT_TOTAL", "POPULATION"]) # Drop Unwanted Columns

### Drop features with less than 75% data
df_NA = df_drop # Rename data for missing values
df_NA = df_NA.dropna(axis = 1, thresh = 0.75*len(df_NA)) # Drop features less than 75% non-NA count for all columns
NAtot = df_NA.isnull().sum().sum() # Get count of all NA values
NAnot = df_NA.count().sum().sum() # Get count of all nonNA values
NAratio = NAtot / (NAtot + NAnot) # Percent of values with values
print(NAratio) # Print value
df_NA.info() # Get info

### Impute missing values
imp = SimpleImputer(strategy = "median") # Build Imputer model. strategy = "mean" or " median" or "most_frequent" or "constant"
df_imp = pd.DataFrame(imp.fit_transform(df_NA)) # Impute missing data
df_imp.columns = df_NA.columns # Rename columns from new dataset

### Create outcome value
df_long = df_imp.rename(columns = {"C19_Cases_R1000": "outcome"}) # Rename multiple columns in place

### Standard Scale Values
df_stsc = df_long.drop(columns = ["outcome"])  # Remove outcome variable
x = df_stsc.values # Save feature values as x
x = StandardScaler().fit_transform(x) # While applying StandardScaler, each feature of your data should be normally distributed such that it will scale the distribution to a mean of zero and a standard deviation of one.
x.shape # Verify that dimensions are same length
np.mean(x),np.std(x) # whether the normalized data has a mean of zero and a standard deviation of one.
df_stsc2 = pd.DataFrame(x, columns = df_stsc.columns) # convert the normalized features into a tabular format with the help of DataFrame.
df_stsc2["outcome"] = df_long["outcome"]

### Rename as Neville
df_nev = df_stsc2 # Rename Data

### Verify
df_nev.info() # Get class, memory, and column info: names, data types, obs.
df_nev.head() # Print first 5 observations

### Create Results Text File
text_file = open("fp/ACS_FL_C19_ZCTA/ACS_FL_C19_ZCTA_README.txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write("allocativ\n") # Line of text with space after
text_file.write("FractureProof\n") # Line of text with space after
text_file.write("ACS 2014-2018 Zip Code Percent Estimates for 50 States\n") # Line of text with space after
text_file.write("Florida DOH 2014-2018 Zip Code 113 Causes of Death\n") # Line of text with space after
text_file.write("COVID 19 Cases by Zip Code 20 July 2020\n") # Line of text with space after
text_file.write("\nFull Dataset\n") # Line of text with space after
text_file.write("(Observations, Features)\n") # Line of text with space after
text_file.write(str(df_nev.shape)) # write string version of variable above
text_file.write("\nDataset with features <25% NA\n") # Line of text with space after
text_file.write(str(df_NA.shape)) # write string version of variable above
text_file.write("\nNA ratio (median values imputed)\n") # Line of text with space after
text_file.write(str(NAratio)) # write string version of variable above
text_file.write("\nDemographics\n") # Line of text with space after
text_file.write(str(df_nev["outcome"].describe())) # write string version of variable above
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

## Step 3: Principal Component Analysis

### Isolate Data frame for Outcome
df_pca = df_nev.drop(columns = ["outcome"]) # Drop outcome variable
degree = len(df_nev.columns) - 2 # Save number of features -1 to get degrees of freedom

### Create PCA model to determine Components
pca = PCA(n_components = degree) # you will pass the number of components to make PCA model based on degrees of freedom
pca.fit(df_pca) # fit to data
df_ev = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_ev = df_ev[(df_ev[0] > 1)] # Save eigenvalues above 1
components = len(df_ev.index) # Save count of values for Variable reduction

### PCA to reduce variables
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(df_pca) # finally call fit_transform on the aggregate data to create PCA results object
df_pca2 = pd.DataFrame(pca.components_, columns = df_pca.columns) # Export eigenvectors to data frame

### Collect list important features
df_pca2["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pca2 = df_pca2[df_pca2.Variance > df_pca2.Variance.mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pca2 = df_pca2.abs() # get absolute value for column or data frame
df_pca3 = pd.DataFrame(df_pca2.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_pc = df_pca3[df_pca3.MaxEV > df_pca3.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_pc = df_pc.reset_index() # Add a new index of ascending values, existing index becomes column named "index"
df_pc =  df_pc.rename(columns = {"index": "Features"}) # Rename multiple columns in place

### Verify
df_pc.info() # Get class, memory, and column info: names, data types, obs.
df_pc.head() # Print first 5 observations

## Write Summary to Text File
text_file = open("fp/ACS_FL_C19_ZCTA/ACS_FL_C19_ZCTA_README.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nPrincipal Components Analysis\n") # Title of section with double space after
text_file.write("\nFeatures with above average absolute value of Eigenvectors on components with above average Explained Variance Ratios \n") # Line of text with space after
text_file.write(str(df_pc.shape)) # write string version of variable above
text_file.write("\nTop 10 Features by Eigenvector\n") # Line of text with space after
text_file.write(df_pc.head(10).to_string()) # write string version of variable above
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

## Step 3: Random Forest Classifier

### Modify for RFC
Y = df_nev["outcome"] # Isolate Outcome variable
features = df_nev.columns.drop(["outcome"]) # Drop outcome variable and Geo to isolate all predictor variable names as features
X = df_nev[features] # Save features columns as predictor data frame
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 

### Fit Forest 
forest.fit(X, Y) # This will take time

### Output importances
gini = forest.feature_importances_ # Output importances of features
l_gini = list(zip(X, gini)) # Create list of variables alongside importance scores 
df_gini = pd.DataFrame(l_gini, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_gini = df_gini.sort_values(by = ["Gini"], ascending = False) # Sort data frame by gini value in desceding order
df_gini = df_gini[(df_gini["Gini"] > df_gini["Gini"].mean())] # Subset by Gini values higher than mean

### Verify
df_gini.info() # Get class, memory, and column info: names, data types, obs.
df_gini.head() # Print first 5 observations

### Write Summary to Text File
text_file = open("fp/ACS_FL_C19_ZCTA/ACS_FL_C19_ZCTA_README.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nRandom Forest\n") # Line of text with space after
text_file.write("\nFeatures with above average Gini Impurity Values\n") # Line of text with space after
text_file.write(str(df_gini.shape)) # write string version of variable above
text_file.write("\nTop 10 Variables by Gini Impurity\n") # Line of text with space after
text_file.write(df_gini.head(10).to_string()) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

## Step 4: Run Recursive Feature Elimination

### Join Forest and Fator Analysis
df_join = pd.merge(df_gini, df_pc, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_features = df_join["Features"] # Save features from data frame
features = df_features.tolist() # Convert to list

### Setup features and Outcome
df_rfecv = df_nev[features] # Add selected features to df

### Setup Predictors and RFE
X = df_rfecv[df_features] # Save features columns as predictor data frame
Y = df_nev["outcome"] # Use outcome data frame 
RFE = LinearRegression() # Use regression coefficient as estimator
selector = RFECV(estimator = RFE, min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo

### Fit Recursive Feature Elimination
selected = selector.fit(X, Y) # This will take time

### Output RFE results
ar_rfe = selected.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, ar_rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
df_rfe = df_rfe.reset_index() # Reset Index
df_rfe = df_rfe.filter(["Features"]) # Keep only selected columns

### Verify
df_rfe.info() # Get class, memory, and column info: names, data types, obs.
df_rfe.head() # Print first 5 observations

### Write Summary to Text File
text_file = open("fp/ACS_FL_C19_ZCTA/ACS_FL_C19_ZCTA_README.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nRecursive Feature Elimination\n") # Line of text with space after
text_file.write("\nFeatures Selected from RF and PCA\n") # Line of text with space after
text_file.write(df_features.to_string()) # write string version of variable above
text_file.write("\nSelected Features by Cross-Validation\n") # Line of text with space after
text_file.write(df_rfe.to_string()) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

## Step 5: Regression Model

### Save selected Features to list
features = list(df_rfe["Features"]) # Save chosen featres as list
df_reg = df_long.filter(features) # Keep only selected columns from rfe
df_reg["outcome"] = df_long["outcome"] # Add outcome variable

### Linear regression: Multiple predictors
X = df_reg[features] # features as x
y = df_reg["outcome"] # Save outcome variable as y
mod = sm.OLS(y, X) # Describe linear model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

### Join with feature labels
df_stack = pd.concat([df_label, df_label2]) # Combine rows with same columns
df_vars = pd.merge(df_rfe, df_stack, on = "Features", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options
df_vars['Labels'] = df_vars['Labels'].str.replace("Percent Estimate!!","") # Strip all spaces from column in data frame
df_vars['Labels'] = df_vars['Labels'].str.replace("!!"," ") # Strip all spaces from column in data frame

### Verify
df_vars.info() # Get class, memory, and column info: names, data types, obs.
df_vars.head() # Print first 5 observations

### Write Summary to Text File
text_file = open("fp/ACS_FL_C19_ZCTA/ACS_FL_C19_ZCTA_README.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nRegression Model\n") # Line of text with space after
text_file.write("\nOLS Model on Original Data\n") # Line of text with space after
text_file.write(str(res.summary())) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

### Export feature labels
df_vars.to_csv(r"fp/ACS_FL_C19_ZCTA/ACS_FL_C19_ZCTA_results.csv") # Export df as csv

## statsmodels Regression Model

### Linear regression: Multiple predictors
features = ['K00_K99_R1000', 'DP05_0010PE', 'DP02_0114PE', 'Residual_R1000', 'N00_N99_R1000', 'V01_Y89_R1000', 'DP03_0004PE', 'DP03_0048PE', 'DP03_0024PE', 'DP03_0012PE', 'DP05_0027PE']
X = df_reg[features] # features as x
y = df_reg["outcome"] # Save outcome variable as y
mod = sm.OLS(y, X) # Describe linear model
res2 = mod.fit() # Fit model
print(res2.summary()) # Summarize model

### Write Summary to Text File
text_file = open("fp/ACS_FL_C19_ZCTA/ACS_FL_C19_ZCTA_README.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nModified Regression Model\n") # Line of text with space after
text_file.write("\nOLS Model on Original Data\n") # Line of text with space after
text_file.write(str(res2.summary())) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

########################

## Build Neural Network

### Save selected Features to list
features = list(df_rfe["Features"]) # Save chosen featres as list
df_reg = df_nev.filter(features) # Keep only selected columns from rfe
df_reg["outcome"] = df_nev["outcome"] # Add outcome variable

# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

### Train Test split
df_train, df_test = train_test_split(df_reg, random_state = 1) # Split data into test/train, default is 25% train

### Setup Neural Network
regr = MLPRegressor(random_state = 1, max_iter = 500).fit(df_train["outcome"], df_train[features]) # Predict outcomes with off the shelf NN

### Check Predict
predict = regr.predict(df_test[features]) # Predict outcomes using trained NN
score = regr.score(df_test[features], df_test["outcome"]) # Get prediction score from NN

### Write Summary to Text File
text_file = open("fp/ACS_FL_C19_ZCTA/ACS_FL_C19_ZCTA_README.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nNeural Network\n") # Line of text with space after
text_file.write("\nPrediction Score\n") # Line of text with space after
text_file.write(str(score)) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file