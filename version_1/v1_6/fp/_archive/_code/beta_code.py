# FractureProof
# Most Recent Version

# Section A: Generate Hypothesis with Machine Learning Algorithms

## Step 1: Import Libraries and Import Dataset

### Import Python Libraries
import os # Operating system navigation

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes

### Import scikit-learn libraries: regression
from sklearn.linear_model import LogisticRegression # Used for machine learning with categorical outcome
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome

### Import scikit-learn: machine learning
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation

### Import scikit-learn libraries: data preparation 
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data

### Import statistics libraries
import statsmodels.api as sm # Statistics package best for regression models for statistical tests
import scipy.stats as st # Statistics package best for t-test, ANOVA, ChiSq
from patsy import dmatrices # Describes statistical models and builds design matrices using R-like formulas

### Import visualization libraries
import matplotlib.pyplot as plt

### Set working directory to subproject folder
os.chdir("C:/Users/drewc/GitHub/neville/dm2") # Set wd to project repository

### Import Dataset
df_nh = pd.read_csv("_data/nhanes_0506_noRX_mort_stage.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository

### Verify
df_nh.info() # Get class, memory, and column info: names, data types, obs.
df_nh.head() # Print first 5 observations

### Create Results Text File
text_1 = str(df_nh.shape)
text_file = open("_fig/_results.txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write("Project Neville\n") # Line of text with space after
text_file.write("NHANES 2005-2006: Undiagnosed Type 2 Diabetes\n") # Line of text with space after
text_file.write("\nTotal Cohort\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

## Step 2: Prepare Data for Analysis

### Create outcome value
df_nh["outcome"] = np.where(((df_nh["DIQ010"] == 2) & (df_nh["LBXGH"] >= 6.4)), 1, 0) # Create new column based on conditions

### Remove outcome, ID, sampling, and weight features
df_out = df_nh.drop(columns = ["LBXGH", "DIQ010"]) # Remove features used for outcome
df_out = df_out.drop(columns = ["SEQN", "SDMVPSU", "SDMVSTRA"]) # Remove Patient ID and Sampling Unit features
df_out = df_out.drop(columns = ["WTMEC2YR", "WTAL2YR", "WTINT2YR", "WTSA2YR", "WTSAF2YR", "WTSB2YR", "WTSC2YR", "WTSOG2YR", "WTSPC2YR", "WTSC2YRA", "WTDR2D"]) # Remove Sample weight features

### Get demographics of outcome group
df_desc = df_out[(df_out["outcome"] == 1)] # Subset data by outcome
df_desc = df_desc.filter(["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDHHINC"]) # Keep only selected demographic features
df_desc.describe() # Run descriptive statistics on all columns

### Drop features with less than 75% data
df_NA = df_sub # Rename data for missing values
df_NA = df_NA.dropna(axis = 1, thresh = 0.75*len(df_NA)) # Drop features less than 75% non-NA count for all columns
NAtot = df_NA.isnull().sum().sum() # Get count of all NA values
NAnot = df_NA.count().sum().sum() # Get count of all nonNA values
NAratio = NAtot / (NAtot + NAnot) # Percent of values with values
Nout = (df_NA["outcome"] == 1).sum() # Get cout of outcome variable
print(NAratio) # Print value
print(Nout) # Print value
df_NA.info() # Get info

### Impute missing values
imp = SimpleImputer(strategy = "median") # Build Imputer model. strategy = "mean" or " median" or "most_frequent" or "constant"
df_imp = pd.DataFrame(imp.fit_transform(df_NA)) # Impute missing data
df_imp.columns = df_NA.columns # Rename columns from new dataset

### Standard Scale Values
df_stsc = df_imp.drop(columns = ["outcome"])  # Remove outcome variable
x = df_stsc.values # Save feature values as x
x = StandardScaler().fit_transform(x) # While applying StandardScaler, each feature of your data should be normally distributed such that it will scale the distribution to a mean of zero and a standard deviation of one.
x.shape # Verify that dimensions are same length
np.mean(x),np.std(x) # whether the normalized data has a mean of zero and a standard deviation of one.
df_stsc2 = pd.DataFrame(x, columns = df_stsc.columns) # convert the normalized features into a tabular format with the help of DataFrame.
df_stsc2["outcome"] = df_imp["outcome"]

### Rename as Neville
df_nev = df_stsc2 # Rename Data

### Verify
df_nev.info() # Get class, memory, and column info: names, data types, obs.
df_nev.head() # Print first 5 observations
df_nev.shape # Print dimensions of data frame

### Write Summary to Text File
text_1 = str(df_nev.shape)
text_2 = str(df_desc.describe())
text_3 = str(NAratio)
text_4 = str(Nout)
text_file = open("_fig/_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nDescriptive Statistics\n") # Title of section with double space after
text_file.write("\nSubset Cohort\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\nDemographics\n") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\nMissing Values\n\n") # Title of section with double space after
text_file.write("\nNA Ratio\n") # Line of text with space after
text_file.write(text_3) # write string version of variable above
text_file.write("\nN outcome\n") # Line of text with space after
text_file.write(text_4) # write string version of variable above
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

## Step 3: Principal Component Analysis

### Isolate Data frame for Outcome
df_pca = df_nev[(df_nev["outcome"] == 1)] # Susbet for PD and DM
df_pca = df_pca.drop(columns = ["outcome"]) # Drop outcome variable

### Create PCA model to determine Components
pca = PCA(n_components = Nout) # you will pass the number of components to make PCA model based on Nout
pca.fit(df_pca) # fit to data
df_ev = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_ev = df_ev[(df_ev[0] > 10)]
components = len(df_ev.index) # Save count of values for Variabel reduction

### PCA to reduce variables
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(df_pca) # finally call fit_transform on the aggregate data to create PCA results object
df_pca2 = pd.DataFrame(pca.components_, columns = df_pca.columns) # Export eigenvectors to data frame

### Collect list important features
df_pca3 = df_pca2[(df_pca2 > 0)] # Remove all values below or equal to 0
df_pc = pd.DataFrame(df_pca3.max()) # select maximum value for each feature
df_pc = df_pc.reset_index() # Save index as first column named "index"
df_pc = df_pc.rename(columns = {"index": "Features", 0: "Eigenvectors"}) # Rename columns
df_pc = df_pc.sort_values(by = ["Eigenvectors"], ascending = False) # Sort Columns by Value
df_pc = df_pc[(df_pc["Eigenvectors"] > df_pc["Eigenvectors"].mean())] # Subset by Gini values higher than mean
df_pc = df_pc.dropna() # Drop all rows with NA values, 0 = rows, 1 = columns 

### Verify
df_pc.info() # Get class, memory, and column info: names, data types, obs.
df_pc.head() # Print first 5 observations

## Write Summary to Text File
text_1 = df_pc.head(10).to_string() # Save variable as string value for input below
text_file = open("_fig/_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nPrincipal Components Analysis\n") # Title of section with double space after
text_file.write("\nTop 10 Variables by Eigenvector\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

## Step 3: Random Forest Classifier

### Modify for RFC
Y = df_nev["outcome"] # Isolate Outcome variable
features = df_nev.columns.drop(["outcome"]) # Drop outcome variable and Geo to isolate all predictor variable names as features
X = df_nev[features] # Save features columns as predictor data frame
forest = RandomForestClassifier(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 

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
text_1 = df_gini.head(10).to_string() # Save variable as string value for input below
text_file = open("_fig/_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nRandom Forest\n") # Line of text with space after
text_file.write("\nTop 10 Variables by Gini Rankings\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
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
Log_RFE = LogisticRegression(max_iter = 100, solver = "liblinear") # Use regression coefficient as estimator
selector = RFECV(estimator = Log_RFE, min_features_to_select = 1) # define selection parameters, in this case all features are selected. See Readme for more ifo

### Fit Recursive Feature Elimination
selected = selector.fit(X, Y) # This will take time

### Output RFE results
ar_rfe = selected.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, ar_rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
df_rfe = df_rfe.drop(columns = ["RFE"]) # Drop Unwanted Columns

### Verify
df_rfe.info() # Get class, memory, and column info: names, data types, obs.
df_rfe.head() # Print first 5 observations

### Write Summary to Text File
text_1 = df_rfe.to_string() # Save variable as string value for input below
text_file = open("_fig/_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nRecursive Feature Elimination\n") # Line of text with space after
text_file.write("\nSelected Features by Cross-Validation\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

## Step 5: Logistic Regression Model

### Save selected Features to list
features = list(df_rfe["Features"]) # Save chosen featres as list
df_log = df_nev.filter(features) # Keep only selected columns from rfe
df_log["outcome"] = df_nev["outcome"] # Add outcome variable

### Logisitc Regression in Scikit Learn
x = df_log[features] # features as x
y = df_log["outcome"] # Save outcome variable as y
Log = LogisticRegression(solver = "liblinear")
model_log = Log.fit(x, y) # Fit model
score_log = model_log.score(x, y) # rsq value
coef_log = model_log.coef_ # Coefficient models as scipy array
logfeatures = df_log[features].columns # Get columns from features df

### Output Coefficients
df_logfeatures = pd.DataFrame(logfeatures) # Convert list to data frame
df_logcoef = pd.DataFrame(coef_log) # Convert array to data frame
df_logcoef = df_logcoef.transpose() # Transpose Rows and Columns
df_logcoef = df_logcoef.reset_index() # Reset index and save as column
df_logfeatures = df_logfeatures.reset_index() # Reset index and save as column
df_logfeatures =  df_logfeatures.rename(columns = {0: "Question"}) # Rename column
df_logcoef =  df_logcoef.rename(columns = {0: "Coefficient"}) # Rename column
df_score = pd.merge(df_logfeatures, df_logcoef, on = "index", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_score["AOR"] = round((2.71828 ** df_score["Coefficient"]), 2) # Take e to power of column and Round to 2 places
df_score["Ab"] = round((df_score["Coefficient"]), 2) # Round column to 2 places
df_score = df_score.drop(columns = ["index", "Coefficient"])  # Remove outcome variable
df_score = df_score.sort_values(by = ["AOR"], ascending = False) # Sort data frame by gini value in desceding order

### Write to CSV
df_score.to_csv(r"_fig/_model.csv") # Clean in excel and select variable

### Verify
df_score.info() # Get class, memory, and column info: names, data types, obs.
df_score.head() # Print first 5 observations

### Write Summary to Text File
text_1 = str(df_score) # Save variable as string value for input below
text_2 = str(score_log) # Save variable as string value for input below
text_file = open("_fig/_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nLogistic Regression Model\n") # Line of text with space after
text_file.write("\nAdjusted odd's ratios and beat coefficients\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n\nR sq = ") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

#################### Break ####################

# Section B: Test Hypothesis with Validated Risk Score

## Step 6: Calculate Risk Score

### Recreate Outcome from Original Data for Train Score
df_train = df_nh[(df_nh["LBXGH"] >= 6.4)] # Susbet for DM2
df_train["outcome"] = np.where((df_train["DIQ010"] == 2), 1, 0) # Create undiagnosed dm2 outcome

### Calculate Interview Score for Training Data
df_train["Interview1"] = np.where((df_train["DIQ190C"] == 2), 1, 0) # Not been told by health professional to reduce calories
df_train["Interview2"] = np.where((df_train["HUQ030"] == 2), 1, 0) # Has no routine place to go for healthcare
df_train["Diet1"] = np.where((df_train["DR2TSUGR"] > 72), 1, 0) # Consumes more than double recommeded sugar intake
df_train["Diet2"] = np.where((df_train["DR2TSODI"] < 2300), 1, 0) # Consumes more than double recommeded sugar intake
df_train["RiskScore"] = df_train["Interview1"] + df_train["Interview2"] + df_train["Diet1"] + df_train["Diet2"] # Add columns together into new column

### Logisitc Regression: Multiple and multilevel predictors 
y, X = dmatrices('outcome ~ Interview1 + Interview2 + Diet1 + Diet2 + RIDAGEYR + RIAGENDR + RIDRETH1 + INDHHINC', data = df_train, return_type = 'dataframe') # Use patsy to create dmatrices for easy multiple and multilevel modeling
mod = sm.Logit(y, X) # Describe logistic model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

### Logisitc regression
X = df_train["RiskScore"] # features as x
y = df_train["outcome"] # Save outcome variable as y
mod = sm.Logit(y, X) # Describe logistic model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

## Step 7: Validate Risk Score

### Import 15-16 NHANES Dataset to Test Score
df_1516 = pd.read_csv("_data/nhanes_1516_noRX_stage.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository

### Recreate Risk Score
df_test = df_1516[(df_1516["LBXGH"] >= 6.4)] # Susbet for DM2 and All Cause Mortality
df_test["outcome"] = np.where((df_test["DIQ010"] == 2), 1, 0) 

### Calculate Interview Score for Training Data
df_test["Interview1"] = np.where((df_test["DIQ190C"] == 2), 1, 0) # Not been told by health professional to reduce calories
df_test["Interview2"] = np.where((df_test["HUQ030"] == 2), 1, 0) # Has no routine place to go for healthcare
df_test["Diet1"] = np.where((df_test["DR2TSUGR"] > 72), 1, 0) # Consumes more than double recommeded sugar intake
df_test["Diet2"] = np.where((df_test["DR2TSODI"] < 2300), 1, 0) # Consumes more than double recommeded sugar intake
df_test["RiskScore"] = df_test["Interview1"] + df_train["Interview2"] + df_train["Diet1"] + df_train["Diet2"] # Add columns together into new column

### Logisitc Regression: Multiple and multilevel predictors 
y, X = dmatrices('outcome ~ Interview1 + Interview2 + Diet1 + Diet2 + RIDAGEYR + RIAGENDR + RIDRETH1 + INDHHINC', data = df_test, return_type = 'dataframe') # Use patsy to create dmatrices for easy multiple and multilevel modeling
mod = sm.Logit(y, X) # Describe logistic model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

### Logisitc regression
X = df_test["RiskScore"] # features as x
y = df_test["outcome"] # Save outcome variable as y
mod = sm.Logit(y, X) # Describe logistic model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

### Calculate Test AUC score
trainX = df_train["RiskScore"] 
trainY = df_train["outcome"] 
testX = df_test["RiskScore"] 
testY = df_test["outcome"] 
model = LogisticRegression()
model.fit(trainX, trainY)
probs = model.predict_proba(testX)

### Write Summary to Text File
text_1 = str(trainscore) # Save variable as string value for input below
text_2 = str(trainauc) # Save variable as string value for input below
text_3 = str(testscore) # Save variable as string value for input below
text_4 = str(testauc) # Save variable as string value for input below
text_file = open("_fig/_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("\nArea Under the Curve\n") # Line of text with space after
text_file.write("\n\nTrain Score Descriptive Statistics") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n\nTrain C-Statistic") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\n\nTest Score Descriptive Statistics") # Line of text with space after
text_file.write(text_3) # write string version of variable above
text_file.write("\n\nTest C-Statistic") # Line of text with space after
text_file.write(text_4) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

print("THE END")
#### End Script

