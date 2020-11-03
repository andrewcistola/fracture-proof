#### Healthy Neighborhoods Project: Using Ecological Data to Improve Community Health
### Neville Subproject: Using Random Forestes, Factor Analysis, and Logistic regression to Screen Variables for Imapcts on Public Health
## Florida Charts Diabetes Mortality by Census Tract: 2013-2017 Crude Diabetes Mortality Rates
# The Python Programming Langauge Script by DrewC!

#### Section A: Import Libraries, Import Dataset, Prepare for Classification

### Step 1: Import Libraries and Import Dataset

## Import Standard Libraries
import os # Inlcuded in every script DC!
import numpy as np # Inclduded in every code script DC!
import pandas as pd # Incldued in every code script for DC!
import scipy as sp # Incldued in every code script for DC!

## Import Statistics Packages
from sklearn.linear_model import LogisticRegression # Logisitc Regression in Sklean
from sklearn.linear_model import LinearRegression # Logisitc Regression in Sklean
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.preprocessing import StandardScaler # Scaling for Principal components analysis

## Import Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier as rfc # Random Forest classification component
from sklearn.feature_selection import RFECV as rfe # Recursive Feature selection component
from sklearn.svm import SVR as svr # Linear model for RFE

## Import Dataset
os.chdir("C:/Users/drewc/GitHub/Healthy_Neighborhoods") # Set wd to project repository
df_acs = pd.read_csv("_data/neville_dm2_acs.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository

## Tidy Data
df_acs = df_acs.apply(pd.to_numeric, errors = "coerce") # Convert all columns to numeric

## Verify
df_acs.info() # Get class, memory, and column info: names, data types, obs.
df_acs.head() # Print first 5 observations
df_acs.shape # Print dimensions of data frame

### Step 2: Prepare Data for Full Analysis

## Create outcome variable and remove variables used to calculate
df_acs["Diabetes"].describe() # Get summary statistics and recode outcome as top quartile
df_acs["outcome"] = np.where((df_acs["Diabetes"] >= 19.43), 2, 1) # Create New Column Based on Conditions with nested If then where DM = 2, PD = 1, Neither = 0
df_acs = df_acs.drop(columns = ["County", "Tract", "Diabetes"]) # Drop Unwanted Columns

## Rename Data
df_nev = df_acs # Save complete dataset as Neville (Reduction)

## Resolve missing values
df_nev = df_nev.dropna() # Drop col with less than threshold above

## Verify
df_nev.info() # Get class, memory, and column info: names, data types, obs.
df_nev.head() # Print first 5 observations
df_nev.shape # Print dimensions of data frame

# Create Results Text File
text_1 = str(df_acs.shape)
text_2 = str(df_nev.shape)
text_file = open("neville/neville_acs_flcharts_dm2_results.txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write("Healthy Neighborhoods Project: Using Ecological Data to Improve Community Health\n") # Line of text with space after
text_file.write("Neville Subproject: Using Random Forestes, Principal Component Analysis, and Logistic regression to Screen Variables for Imapcts on Public Health\n") # Line of text with space after
text_file.write("## Florida Charts Diabetes Mortality by Census Tract: 2013-2017 Crude Diabetes Mortality Rates\n") # Line of text with space after
text_file.write("The Python Programming Langauge Script by DrewC!\n") # Line of text with space after
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.write("Census Tracts in Florida with Mortality Data\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\nCohort subset for variable reduction\n") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.close() # Close file

#### Section B: Conduct Principal Component Analysis to Identify Latent Variables for Prediabetes SubCohort

### Step 4: Prepare for Principal Component Analysis

## Prepare Data for PCA
df_pca = df_nev.drop(columns = ["outcome"])  # Remove outcome variable
x = df_pca.values # Save feature values as x
x = StandardScaler().fit_transform(x) # While applying StandardScaler, each feature of your data should be normally distributed such that it will scale the distribution to a mean of zero and a standard deviation of one.

## Verify
x.shape # Verify that dimensions are same length
np.mean(x),np.std(x) # whether the normalized data has a mean of zero and a standard deviation of one.

## Convert to Data frame
df_pcn = pd.DataFrame(x, columns = df_pca.columns) # convert the normalized features into a tabular format with the help of DataFrame.

## Create PCA model to determine Components
pca = PCA(n_components = 100) # you will pass the number of components to make PCA model
pca.fit(df_pcn) # fit to data
df_pcs = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_pcs = df_pcs[(df_pcs[0] > 10)]
components = len(df_pcs.index) # Save count of values for Variabel reduction

### Step 5: Run PCA with principal component count

## PCA to reduce variables
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(df_pcn) # finally call fit_transform on the aggregate data to create PCA results object
df_pcs2 = pd.DataFrame(pca.components_, columns = df_pcn.columns) # Export eigenvectors to data frame

## Collect list important features
df_pcs3 = df_pcs2[(df_pcs2 > 0)] # Remove all values below or equal to 0
df_pc = pd.DataFrame(df_pcs3.max()) # select maximum value for each feature
df_pc = df_pc.reset_index() # Save index as first column named "index"
df_pc = df_pc.rename(columns = {"index": "Features", 0: "Eigenvectors"}) # Rename columns
df_pc = df_pc.sort_values(by = ["Eigenvectors"], ascending = False) # Sort Columns by Value
df_pc = df_pc[(df_pc["Eigenvectors"] > df_pc["Eigenvectors"].mean())] # Subset by Gini values higher than mean
df_pc = df_pc.dropna() # Drop all rows with NA values, 0 = rows, 1 = columns 

## Verify
df_pc.info() # Get class, memory, and column info: names, data types, obs.
df_pc.head() # Print first 5 observations

## Write Summary to Text File
text_1 = df_pc.head(10).to_string() # Save variable as string value for input below
text_file = open("neville/neville_acs_flcharts_dm2_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\n\nPrincipal Components Analysis\n\n") # Title of section with double space after
text_file.write("\nTop 10 Variables by Eigenvector\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

#### Section C: Create a Random Forest to Rank Variables by Importance for Prediabetes

### Step 6: Run Random Forest

## Prepare for Classification
df_rf = df_nev.fillna(0).astype(np.float64) # Remove NA and change to float64 zeros for feature selection and save as Neville
Y = df_rf["outcome"] # Isolate Outcome variable
features = df_rf.columns.drop(["outcome"]) # Drop outcome variable and Geo to isolate all predictor variable names as features
X = df_rf[features] # Save features columns as predictor data frame
forest = rfc(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 

## Run Forest 
forest.fit(X, Y) # This will take time

## Output importances
gini = forest.feature_importances_ # Output importances of features
l_gini = list(zip(X, gini)) # Create list of variables alongside importance scores 
df_gini = pd.DataFrame(l_gini, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_gini = df_gini.sort_values(by = ["Gini"], ascending = False) # Sort data frame by gini value in desceding order
df_gini = df_gini[(df_gini["Gini"] > df_gini["Gini"].mean())] # Subset by Gini values higher than mean

## Verify
df_gini.info() # Get class, memory, and column info: names, data types, obs.
df_gini.head() # Print first 5 observations

## Write Summary to Text File
text_1 = df_gini.head(10).to_string() # Save variable as string value for input below
text_file = open("neville/neville_acs_flcharts_dm2_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nRandom Forest\n") # Line of text with space after
text_file.write("\nTop 10 Variables by Gini Rankings\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

#### Section D: Run Recursive Feature Selection 

### Step 7: Run Recursive Feature Selection

## Join Forest and Fator Analysis
df_join = pd.merge(df_gini, df_pc, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_features = df_join["Features"] # Save features from data frame
features = df_features.tolist() # Convert to list

## Setup Predictors and RFE
df_rfe = df_nev[features] # Add selected features to df
df_rfe["outcome"] = df_nev["outcome"] # Add outcome to RFE df
df_rfe = df_rfe.dropna() # Drop all columns with NA
X = df_rfe[df_features] # Save features columns as predictor data frame
Y = df_rfe["outcome"] # Use outcome data frame 
Log_RFE = LogisticRegression(solver = "liblinear", max_iter = 4000) # Use regression coefficient as estimator
selector = rfe(estimator = Log_RFE, step = 1, min_features_to_select = 1) # define selection parameters, in this case all features are selected. See Readme for more ifo

## Run Recursive Feature Selection
selected = selector.fit(X, Y) # This will take time

## Output RFE results
ar_rfe = selected.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, ar_rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
df_rfe = df_rfe.drop(columns = ["RFE"]) # Drop Unwanted Columns

## Verify
df_rfe.info() # Get class, memory, and column info: names, data types, obs.
df_rfe.head() # Print first 5 observations

## Write Summary to Text File
text_1 = df_rfe.to_string() # Save variable as string value for input below
text_file = open("neville/neville_acs_flcharts_dm2_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nRecursive Feature Selection\n") # Line of text with space after
text_file.write("\nSelected Features by Cross-Validation\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

#### Section E: Build Regression Model to Evaluate Selected Features

### Step 8: Logistic Regression Model

## Save selected Features to list
features = list(df_rfe["Features"]) # Save chosen featres as list
df_log = df_nev.filter(features) # Keep only selected columns from rfe
df_log["outcome"] = df_nev["outcome"] # Add outcome variable

## Logisitc Regression in Scikit Learn
x = df_log[features] # features as x
y = df_log["outcome"] # Save outcome variable as y
Log = LogisticRegression(solver = "liblinear")
model_log = Log.fit(x, y) # Fit model
score_log = model_log.score(x, y) # rsq value
coef_log = model_log.coef_ # Coefficient models as scipy array
logfeatures = df_log[features].columns # Get columns from features df

## Output Coefficients
df_logfeatures = pd.DataFrame(logfeatures) # Convert list to data frame
df_logcoef = pd.DataFrame(coef_log) # Convert array to data frame
df_logcoef = df_logcoef.transpose() # Transpose Rows and Columns
df_logcoef = df_logcoef.reset_index() # Reset index and save as column
df_logfeatures = df_logfeatures.reset_index() # Reset index and save as column
df_logfeatures =  df_logfeatures.rename(columns = {0: "Question"}) # Rename column
df_logcoef =  df_logcoef.rename(columns = {0: "Coefficient"}) # Rename column
df_score = pd.merge(df_logfeatures, df_logcoef, on = "index", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_score["Score"] = round((df_score["Coefficient"] * 100), 1)
df_score = df_score.drop(columns = ["index", "Coefficient"])  # Remove outcome variable
df_score = df_score.sort_values(by = ["Score"], ascending = False) # Sort data frame by gini value in desceding order

## Verify
df_score.info() # Get class, memory, and column info: names, data types, obs.
df_score.head() # Print first 5 observations

## Write Summary to Text File
text_1 = str(df_score) # Save variable as string value for input below
text_2 = str(score_log) # Save variable as string value for input below
text_file = open("neville/neville_acs_flcharts_dm2_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nRegression Model\n") # Line of text with space after
text_file.write("\nCoefficient Scores\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n\nR sq = ") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.write("THE END")
text_file.close() # Close file

## Write to CSV
df_score.to_csv(r"neville/neville_acs_flcharts_dm2_results.csv") # Clean in excel and select variable

print("THE END")
#### End Script