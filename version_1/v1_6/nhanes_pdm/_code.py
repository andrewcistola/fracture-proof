#### Healthy Neighborhoods Project: Using Ecological Data to Improve Community Health
### Neville Subproject: Using Random Forestes, Factor Analysis, and Logistic regression to Screen Variables for Imapcts on Public Health
## NHANES 2015-2016: Detecting different factors between individuals with a BMI over 30 and are between 40 and 70 that have pre-diabates and have diabetes
# The Python Programming Langauge Script by DrewC!

#### Section 1: Import Libraries, Import Dataset, Prepare for Classification

### Step 1: Import Libraries and Import Dataset

## Import Standard Libraries
import os # Inlcuded in every script DC!
import numpy as np # Inclduded in every code script DC!
import pandas as pd # Incldued in every code script for DC!
import scipy as sp # Incldued in every code script for DC!
import sklearn as sk # Incldued in every code script for DC!

## Import Statistics Packages
from sklearn.linear_model import LogisticRegression # Logisitc Regression in Sklean
from sklearn.metrics import roc_curve # ROC test
from sklearn.metrics import roc_auc_score # AUC score

## Import Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier as rfc # Random Forest classification component
from sklearn.feature_selection import RFE as rfe # Recursive Feature selection component

## Import Factor Analysis Packages
from factor_analyzer import FactorAnalyzer as fa
from factor_analyzer.factor_analyzer import calculate_kmo # Kaiser-Meyer-Olkin test package from Scikit-learn

## Import Dataset
os.chdir("C:/Users/drewc/GitHub/Healthy_Neighborhoods") # Set wd to project repository
df_nh = pd.read_csv("_data/nhanes_1516_noRX_stage.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository

## Tidy Data
df_nh = df_nh.drop(columns = ["AUATYMTL", "SLQ300", "SLQ310", "AUATYMTR"]) # Remove variables with factor value types

## Verify
df_nh.info() # Get class, memory, and column info: names, data types, obs.
df_nh.head() # Print first 5 observations

# Create Results Text File
text_file = open("Neville/neville_nhanes_pdm_results.txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write("Healthy Neighborhoods Project: Using Ecological Data to Improve Community Health\n") # Line of text with space after
text_file.write("Neville Subproject: Using Random Forestes, Factor Analysis, and Logistic regression to Screen Variables for Imapcts on Public Health\n") # Line of text with space after
text_file.write("NHANES 2015-2016: Detecting different factors between individuals with a BMI over 30 and are between 40 and 70 that have pre-diabates and have diabetes\n") # Line of text with space after
text_file.write("The Python Programming Langauge Script by DrewC!\n") # Line of text with space after
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

### Step 2: Prepare Data for Analysis

## Create new columns for outcomes
df_nh["outcome"] = np.where((df_nh["LBXGH"] >= 6.4), 2, np.where((df_nh["LBXGH"] < 6.4) & (df_nh["LBXGH"] >= 5.7), 1, 0)) # Create New Column Based on Conditions with nested If then where DM = 2, PD = 1, Neither = 0
df_nh = df_nh.drop(columns = ["LBXGH", "DIQ010", "DIQ160"]) # Drop HbA1c column and PD and DM dX

## Subset for screening group
df_nh = df_nh[(df_nh["RIDAGEYR"] >= 40) & (df_nh["RIDAGEYR"] <= 70) & (df_nh["BMXBMI"] >= 30)] # Subset data frame by USPSTF screening values
df_nev = df_nh.drop(columns = ["RIDAGEYR", "BMXBMI"]) # Drop columns for age and bmi

## Verify
df_nev.info() # Get class, memory, and column info: names, data types, obs.
df_nev.head() # Print first 5 observations

#### Section 2: Conduct Factor Analysis to Identify Latent Variables for Prediabetes SubCohort

### Step 3: Prepare for Factor Analysis

## Subset by group with outcome
df_fa = df_nev[df_nev["outcome"] == 1] # Subset for prediabetes
df_fa = df_fa.drop(columns = ["outcome", "SEQN"])  # Remove outcome variable
df_fa = df_fa.replace([np.inf, -np.inf], np.nan) # Convert all inf to NA
df_fa = df_fa[df_fa > 0] # Convert all zero to NA
df_fa.info() # Get class, memory, and column info: names, data types, obs.

## Remove NA values for correlation matrix
NAcol = df_fa.count(0).quantile(q = .95) # Get percentile for non-NA count for all columns
df_fa = df_fa.dropna(axis = 1, thresh = NAcol) # Drop col with less than threshold above
df_fa = df_fa.dropna() # Drop all NA values
df_fa = df_fa.drop(df_fa.std()[df_fa.std() == 0].index.values, axis=1) # Drop all columns that are constant
df_fa.info() # Get class, memory, and column info: names, data types, obs.

## Measure Suitability for Factor Analysis
kmo_all,kmo_model = calculate_kmo(df_fa) # Kaiser-Meyer-Olkin test for suitability for factor analysis 
kmo_model # Must be over than 0.6

### Step 4: Determine Number of Factors

## Get raw eigenvalues to determine factors
np_fa = df_fa.corr() # Create correlation matrix
fa = FactorAnalyzer(rotation = None) # create fa object with no rotation
fa.fit(np_fa) # Perform initial fa with enough factors to find point where they move below 0
ar_ev = fa.get_eigenvalues() # print eigenvalues
df_ev = pd.DataFrame(ar_ev) # Convert array to data frame
df_ev = df_ev[df_ev > 1] # Subset for all ev above 1

## Verify
df_ev.info() # Get class, memory, and column info: names, data types, obs.
df_ev.head() # Print first 5 observations

### Step 5: Identify Variable Loadings

## Identify number of factors
df_ev1 = df_ev.dropna(1) # Drop all rows with NA values, 0 = rows, 1 = columns
df_ev1 = df_ev1.count(1) # Get count of columns
factors = df_ev1.iloc[1] # Use iloc to save number of factors and pass into next step

## Get factor loadings and rotate
fa = FactorAnalyzer(rotation = "promax", n_factors = factors) # create fa object with no rotation
fa.fit(np_fa) # Use factors variable from above to perform fa with rotation
ar_ld = fa.loadings_ # Save factor feature loadings 
df_ld = pd.DataFrame(ar_ld) # Convert array to data frame
df_ld = df_ld[df_ld > 0.5] # Subset for all ev above 0.5

## Get Variable Names
df_ld["features"] = df_fa.columns
df_ld = df_ld.set_index("features")
df_ld = df_ld.transpose() # Transpose Rows and Columns
df_ld = df_ld.unstack().reset_index() # Convert from wide to long
df_ld = df_ld.rename(columns = {"features": "Features", "level_1": "Factors", 0 : "Loadings"}) # Rename columns
df_ld = df_ld[(df_ld["Loadings"] >= 0.5)]
df_ld = df_ld.sort_values(by = ["Factors"], ascending = False) # Sort Columns by Value
df_pd = df_ld

## Verify
df_ld.info() # Get class, memory, and column info: names, data types, obs.
df_ld.head() # Print first 5 observations

## Write Summary to Text File
text_1 = str(kmo_model) # Save variable as string value for input below
text_2 = df_ev.to_string() # Save variable as string value for input below
text_3 = df_pd.to_string() # Save variable as string value for input below
text_file = open("Neville/neville_nhanes_pdm_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("Factor Analysis Results: Prediabetes Subcohort\n\n") # Title of section with double space after
text_file.write("Kaiser-Meyer-Olkin Test\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\nRaw Eigenvalues\n") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\nFactor Loadings\n") # Line of text with space after
text_file.write(text_3) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

#### Section 2.5: Conduct Factor Analysis to Identify Latent Variables for Prediabetes SubCohort

### Step 3: Prepare for Factor Analysis

## Subset by group with outcome
df_fa = df_nev[df_nev["outcome"] == 2] # Subset for prediabetes
df_fa = df_fa.drop(columns = ["outcome", "SEQN"])  # Remove outcome variable
df_fa = df_fa.replace([np.inf, -np.inf], np.nan) # Convert all inf to NA
df_fa = df_fa[df_fa > 0] # Convert all zero to NA
df_fa.info() # Get class, memory, and column info: names, data types, obs.

## Remove NA values for correlation matrix
NAcol = df_fa.count(0).quantile(q = .99) # Get percentile for non-NA count for all columns
df_fa = df_fa.dropna(axis = 1, thresh = NAcol) # Drop col with less than threshold above
df_fa = df_fa.dropna() # Drop all NA values
df_fa = df_fa.drop(df_fa.std()[df_fa.std() == 0].index.values, axis=1) # Drop all columns that are constant
df_fa.info() # Get class, memory, and column info: names, data types, obs.

## Measure Suitability for Factor Analysis
kmo_all,kmo_model = calculate_kmo(df_fa) # Kaiser-Meyer-Olkin test for suitability for factor analysis 
kmo_model # Must be over than 0.6

### Step 4: Determine Number of Factors

## Get raw eigenvalues to determine factors
np_fa = df_fa.corr() # Create correlation matrix
fa = FactorAnalyzer(rotation = None) # create fa object with no rotation
fa.fit(np_fa) # Perform initial fa with enough factors to find point where they move below 0
ar_ev = fa.get_eigenvalues() # print eigenvalues
df_ev = pd.DataFrame(ar_ev) # Convert array to data frame
df_ev = df_ev[df_ev > 1] # Subset for all ev above 1

## Verify
df_ev.info() # Get class, memory, and column info: names, data types, obs.
df_ev.head() # Print first 5 observations

### Step 5: Identify Variable Loadings

## Identify number of factors
df_ev1 = df_ev.dropna(1) # Drop all rows with NA values, 0 = rows, 1 = columns
df_ev1 = df_ev1.count(1) # Get count of columns
factors = df_ev1.iloc[1] # Use iloc to save number of factors and pass into next step

## Get factor loadings and rotate
fa = FactorAnalyzer(rotation = "promax", n_factors = factors) # create fa object with no rotation
fa.fit(np_fa) # Use factors variable from above to perform fa with rotation
ar_ld = fa.loadings_ # Save factor feature loadings 
df_ld = pd.DataFrame(ar_ld) # Convert array to data frame
df_ld = df_ld2[df_ld > 0.5] # Subset for all ev above 0.5

## Get Variable Names
df_ld["features"] = df_fa.columns
df_ld = df_ld.set_index("features")
df_ld = df_ld.transpose() # Transpose Rows and Columns
df_ld = df_ld.unstack().reset_index() # Convert from wide to long
df_ld = df_ld.rename(columns = {"features": "Features", "level_1": "Factors", 0 : "Loadings"}) # Rename columns
df_ld = df_ld[(df_ld["Loadings"] >= 0.5)]
df_ld = df_ld.sort_values(by = ["Factors"], ascending = False) # Sort Columns by Value
df_dm = df_ld # Rename

## Verify
df_dm.info() # Get class, memory, and column info: names, data types, obs.
df_dm.head() # Print first 5 observations

## Write Summary to Text File
text_1 = str(kmo_model) # Save variable as string value for input below
text_2 = df_ev.to_string() # Save variable as string value for input below
text_3 = df_dm.to_string() # Save variable as string value for input below
text_file = open("Neville/neville_nhanes_pdm_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("Factor Analysis Results: Diabetes Subcohort\n\n") # Title of section with double space after
text_file.write("Kaiser-Meyer-Olkin Test\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\nRaw Eigenvalues\n") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\nFactor Loadings\n") # Line of text with space after
text_file.write(text_3) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

#### Section 3: Create a Random Forest to Rank Variables by Importance for Prediabetes Subcohort

### Step 6: Create Random Forest

## Prepare for Classification
df_rf = df_nev.fillna(0).astype(np.float64) # Remove NA and change to float64 zeros for feature selection and save as Neville
X = df_rf[features] # Save features columns as predictor data frame
Y = df_rf["outcome"] # Isolate Outcome variable
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
text_1 = str(forest) # Save variable as string value for input below
text_2 = df_gini.to_string() # Save variable as string value for input below
text_file = open("Neville/neville_nhanes_pdm_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("Random Forest Results: Prediabetes Subcohort\n\n") # Title of section with double space after
text_file.write("\nRandom Forest\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\nGini Rankings\n") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

### Step 7: Combine with Factor Analysis to Identify Cantidate Variables

## Join Forest and Fator Analysis
df_join = pd.merge(df_gini, df_ld1, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_join2 = pd.merge(df_join, df_ld2, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_features = df_join["Features"] # Save features from data frame
features = df_features.tolist() # Convert to list

## Write Summary to Text File
text_1 = str(forest) # Save variable as string value for input below
text_2 = df_gini.to_string() # Save variable as string value for input below
text_2 = df_join.to_string() # Save variable as string value for input below
text_file = open("Neville/neville_nhanes_pdm_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("Random Forest Results: Prediabetes Subcohort\n\n") # Title of section with double space after
text_file.write("\nRandom Forest\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\nGini Rankings\n") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\nFeatures Selected\n") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

#### Section 4 : Run Recursive Feature Selection to Develop Regression Model

### Step 7: Recursive Feature Selection

## Split into Test and Training Set
df_split = df_nev.reset_index()
df_test = df_split.sample(frac = 0.5) # 50-50 Split
df_test = df_test.reset_index() # Test set reset index
df_train = df_split.drop(df_test.index) # Train set drop index values from df_test
df_train = df_train.reset_index() # reset index
df_train.info()  # Get class, memory, and column info: names, data types, obs.
df_test.info()  # Get class, memory, and column info: names, data types, obs.

## Setup Predictors and RFE
df_rfe = df_test[df_features] # Add selected features to df
df_rfe["outcome"] = df_test["outcome"] # Add outcome to RFE df
df_rfe = df_rfe[(df_rfe["outcome"] != 0)] # Susbet for PD and DM
df_rfe = df_rfe.dropna() # Drop all columns with NA
df_rfe.info() # Get class, memory, and column info: names, data types, obs
X = df_rfe[df_features] # Save features columns as predictor data frame
Y = df_rfe["outcome"] # Use outcome data frame 
Log_RFE = LogisticRegression(solver = "liblinear") # Use regression coefficient as estimator
selector = rfe(Log_RFE, 5, step = 1) # define selection parameters, in this case all features are selected. See Readme for more ifo

## Run Recursive Feature Selection
selected = selector.fit(X, Y) # This will take time

## Output RFE results
ar_rfe = selected.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, ar_rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True

## Verify
df_rfe.info() # Get class, memory, and column info: names, data types, obs.
df_rfe.head() # Print first 5 observations

### Step 8: Logistic Regression Model

## Save selected Features to list
features = df_rfe["Features"].tolist() # Save 5 identified variables as list
df_log = df_train.filter(features) # Keep only selected columns from rfe
df_log["outcome"] = df_train["outcome"] # Add outcome variable
df_log.info() # Get class, memory, and column info: names, data types, obs.
df_log = df_log[(df_log["outcome"] != 0)] # Susbet for PD and DM

## Logisitc Regression in Scikit Learn
x = df_log[features] # features as x
y = df_log["outcome"] # Save outcome variable as y
Log_log = LogisticRegression(solver = "lbfgs")
model_log = Log_log.fit(x, y) # Fit model
score_log = model_log.score(x, y) # rsq value
coef_log = model_log.coef_ # Coefficient models as scipy array
logfeatures = df_log[features].columns # Get columns from features df

## Output Coefficients to Risk Score
df_logfeatures = pd.DataFrame(logfeatures) # Convert list to data frame
df_logcoef = pd.DataFrame(coef_log) # Convert array to data frame
df_logcoef = df_logcoef.transpose() # Transpose Rows and Columns
df_logcoef = df_logcoef.reset_index() # Reset index and save as column
df_logfeatures = df_logfeatures.reset_index() # Reset index and save as column
df_logfeatures =  df_logfeatures.rename(columns = {0: "Question"}) # Rename column
df_logcoef =  df_logcoef.rename(columns = {0: "Coefficient"}) # Rename column
df_score = pd.merge(df_logfeatures, df_logcoef, on = "index", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_score["Multipliers"] = round((df_score["Coefficient"] * 100), 1)
df_score = df_score.drop(columns = ["index", "Coefficient"])  # Remove outcome variable

## Verify
df_score.info() # Get class, memory, and column info: names, data types, obs.
df_score.head() # Print first 5 observations

## Write Summary to Text File
text_1 = str(features) # Save variable as string value for input below
text_2 = str(model.coef_) # Save variable as string value for input below
text_3 = str(score_log)
text_4 = str(df_score)
text_file = open("Neville/neville_nhanes_pdm_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("Regression Results\n\n") # Title of section with double space after
text_file.write("\nRecursive Feature Selection\n") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\nLogisitc Regression\n") # Line of text with space after
text_file.write(text_2) # write string version of variable above
text_file.write("\n R sq = ") # Line of text with space after
text_file.write(text_3) # write string version of variable above
text_file.write("\n\nRisk Score Multiplier\n") # Line of text with space after
text_file.write(text_4) # write string version of variable above
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.close() # Close file

#### Section 5: Validation

## Create Dichotomous Variables
df_test1 = df_test[logfeatures]
df_test1[str(df_score.iloc[0,0])] = np.where(df_test[str(df_score.iloc[0,0])] == 1, 1, 0) # Create New Column Based on Conditions
df_test1[str(df_score.iloc[1,0])] = np.where(df_test[str(df_score.iloc[1,0])] == 1, 1, 0) # Create New Column Based on Conditions
df_test1[str(df_score.iloc[2,0])] = np.where(df_test[str(df_score.iloc[2,0])] == 1, 1, 0) # Create New Column Based on Conditions
df_test1[str(df_score.iloc[3,0])] = np.where(df_test[str(df_score.iloc[3,0])] == 1, 1, 0) # Create New Column Based on Conditions
df_test1[str(df_score.iloc[4,0])] = np.where(df_test[str(df_score.iloc[4,0])] == 1, 1, 0) # Create New Column Based on Conditions

## Create Score Column
df_test1["score"] = 0
df_test1["score"] = (df_test1[str(
    df_score.iloc[0,0])] * float(df_score.iloc[0,1])) + (df_test1[str(
        df_score.iloc[1,0])] * float(df_score.iloc[1,1])) + (df_test1[str(
            df_score.iloc[2,0])] * float(df_score.iloc[2,1])) + (df_test1[str(
                df_score.iloc[3,0])] * float(df_score.iloc[3,1])) + (df_test1[str(
                    df_score.iloc[4,0])] * float(df_score.iloc[4,1]))
RiskScore_mean = df_test1["score"].mean()

## Logistic Regression in Scikit Learn
log = LogisticRegression() # import Log
x = np.array(df_test["score"]).reshape((-1, 1)) # Save score x
y = np.array(df_test["outcome"]).reshape((-1, 1)) # Save outcome variable as y
model = log.fit(x, y.ravel()) # Fit model
score = model.score(x, y) # rsq value

## AUC Score
yp = model.predict(x)
yp = np.array(yp).reshape((-1, 1)) # Save outcome variable as y
auc = roc_auc_score(y, yp)

## Write Summary to Text File
text_1 = str(RiskScore_mean) # Save variable as string value for input below
text_2 = str(auc) # Save variable as string value for input below
text_file = open("Neville/neville_nhanes_pdm_results.txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write("Validation\n\n") # Title of section with double space after
text_file.write("\nRisk Score Mean = ") # Line of text with space after
text_file.write(text_1) # write string version of variable above
text_file.write("\n\nAUC Score = ") # Line of text with space after
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.write("The END") # Add two lines of blank text at end of every section text
text_file.close() # Close file

print("THE END")
#### End Script