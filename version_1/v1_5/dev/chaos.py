File = "CMS_HRSA_ACS_FL_v1.3_predict"
path = "fp/VBHC/ADJ/delta/"
title = "FractureProof Final Payment Adjustments from CMS: CMS, ACS, HRSA, Aggregate Model"
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
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential # Sequential neural network modeling
from tensorflow.keras.models import Sequential

### Set Directory
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

### Import ACS Data and FIPS to join by ZCTA
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_doh = pd.read_csv("hnb/DOH/FL/113_5Y2018/FL_113_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_hrsa = pd.read_csv("hnb/HRSA/AHRF/AHRF_2018_2019_SAS/AHRF_full.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_hrsa = df_hrsa.set_index("FIPS") # Set column as index
df_hrsa = df_hrsa.loc[:, df_hrsa.columns.str.contains('2018')] # Select columns by string value
df_hrsa = df_hrsa.reset_index(level = ["FIPS"]) # Reset Index
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_cms = df_cms.rename(columns = {"2020 VBP Adjustment Factor": "quant"}) # Rename quantitative outcome
df_cms["train"] = np.where(df_cms["quant"] > 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms["test"] = np.where(df_cms["2019 VBP Adjustment Factor"] > 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms["test2"] = np.where(df_cms["2018 VBP Adjustment Factor"] > 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms = df_cms.drop(columns = ["Facility ID"]) # Drop ID variables
df_cms = df_cms.drop(columns = ["2018 VBP Adjustment Factor", "2019 VBP Adjustment Factor"]) # Drop proximity features: Adjustment factor scores
df_cms = df_cms.drop(columns = ["Total Performance Score", "Weighted Normalized Clinical Outcomes Domain Score", "Weighted Safety Domain Score", "Weighted Person and Community Engagement Domain Score", "Weighted Efficiency and Cost Reduction Domain Score"]) # Drop proximity features: Adjustment factor scores
df_fl = pd.merge(df_acs, df_doh, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_fl = pd.merge(df_fl, df_hrsa, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_fl = pd.merge(df_cms, df_fl, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_fl.info() # Get class, memory, and column info: names, data types, obs.

### Build Model of predictive features
features = ["Marketplace Enrollees, Active Enrl 2018", 
            "# Medicare Advantage Enrollees 2018",
            "Chiropractors w/NPI 2018",
            "Medicare Presc Drug Plan Enr 2018",
            "Skilled Nursing Facilities 2018",
            "DP02_0047PE_avg",
            "DP02_0124PE_gini",
            "DP03_0008PE_avg",
            "DP03_0034PE_avg",
            "DP04_0139PE_gini",
            "DP04_0086PE_avg",
            "DP05_0040PE_avg",
            "DP05_0077PE_avg",
            "Death rate for pneumonia patients",
            "Medicare hospital spending per patient (Medicare Spending per Beneficiary)",
            "Rate of readmission after discharge from hospital (hospital-wide)",
            "Hospital overall rating",
            "Hospital Ownership ForProfit",
            "quant",
            "train",
            "test",
            "test2"] # Hand select features from results table
df_final = df_fl.filter(features) # Subset by hand selected features for model
df_final = df_final.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_final.info() # Get class, memory, and column info: names, data types, obs.

### Build Regression Model
X = df_final.drop(columns = ["quant", "train", "test", "test2"])
Y = df_final.filter(["quant"])
mod = sm.OLS(Y, X) # Describe linear model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

### Build neural netowkr to predict outcomes
df_final = pd.DataFrame(StandardScaler().fit_transform(df_final.values), columns = df_final.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
input = df_final.shape[1] - 4 # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
classifier = Sequential() # Sequential model building in keras
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
X = df_final.drop(columns = ["quant", "train", "test", "test2"]) # Save features as X numpy data array
Y_train = df_final["train"] # Save test outcome as Y numpy data array
classifier.fit(X, Y_train, batch_size = 10, epochs = 500) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_train = (Y_train > 0)
fpr, tpr, threshold = roc_curve(Y_train, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_train = auc(fpr, tpr) # Plot ROC and get AUC score
Y_test = df_final["test"] # Save train outcome as Y numpy data array
classifier.fit(X, Y_test, batch_size = 10, epochs = 500) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_test = (Y_test > 0)
fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_test = auc(fpr, tpr) # Plot ROC and get AUC score
Y_test2 = df_final["test2"] # Save train outcome as Y numpy data array
classifier.fit(X, Y_test2, batch_size = 10, epochs = 500) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_test2 = (Y_test2 > 0)
fpr, tpr, threshold = roc_curve(Y_test2, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_test2 = auc(fpr, tpr) # Plot ROC and get AUC score

### Import ACS Data and FIPS to join by ZCTA
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_doh = pd.read_csv("hnb/DOH/FL/113_5Y2018/FL_113_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_hrsa = pd.read_csv("hnb/HRSA/AHRF/AHRF_2018_2019_SAS/AHRF_full.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_hrsa = df_hrsa.set_index("FIPS") # Set column as index
df_hrsa = df_hrsa.loc[:, df_hrsa.columns.str.contains('2018')] # Select columns by string value
df_hrsa = df_hrsa.reset_index(level = ["FIPS"]) # Reset Index
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_cms = df_cms.rename(columns = {"2020 VBP Adjustment Factor": "quant"}) # Rename quantitative outcome
df_cms["train"] = np.where(df_cms["quant"] > 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms["test"] = np.where(df_cms["2019 VBP Adjustment Factor"] > 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms["test2"] = np.where(df_cms["2018 VBP Adjustment Factor"] > 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms = df_cms.drop(columns = ["Facility ID"]) # Drop ID variables
df_cms = df_cms.drop(columns = ["2018 VBP Adjustment Factor", "2019 VBP Adjustment Factor"]) # Drop proximity features: Adjustment factor scores
df_cms = df_cms.drop(columns = ["Total Performance Score", "Weighted Normalized Clinical Outcomes Domain Score", "Weighted Safety Domain Score", "Weighted Person and Community Engagement Domain Score", "Weighted Efficiency and Cost Reduction Domain Score"]) # Drop proximity features: Adjustment factor scores
df_fl = pd.merge(df_acs, df_doh, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_fl = pd.merge(df_fl, df_hrsa, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_fl = pd.merge(df_cms, df_fl, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_fl.info() # Get class, memory, and column info: names, data types, obs.

### Build Model of predictive features
features = ["Marketplace Enrollees, Active Enrl 2018", 
            "# Medicare Advantage Enrollees 2018",
            "Chiropractors w/NPI 2018",
            "Medicare Presc Drug Plan Enr 2018",
            "Skilled Nursing Facilities 2018",
            "Adv Prct Regist Nurs,Male w/NPI 2018",
            "DP02_0037PE_avg",
            "DP02_0047PE_avg",
            "DP02_0124PE_gini",
            "DP02_0148PE_gini",
            "DP03_0008PE_avg",
            "DP03_0034PE_avg",
            "DP04_0139PE_gini",
            "DP04_0086PE_avg",
            "DP05_0040PE_avg",
            "DP05_0077PE_avg",
            "Death rate for pneumonia patients",
            "Medicare hospital spending per patient (Medicare Spending per Beneficiary)",
            "Rate of readmission after discharge from hospital (hospital-wide)",
            "Hospital overall rating",
            "TOTAL HAC SCORE",
            "Hospital Ownership ForProfit",
            "quant",
            "train",
            "test",
            "test2"] # Hand select features from results table
df_final = df_fl.filter(features) # Subset by hand selected features for model
df_final = df_final.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_final.info() # Get class, memory, and column info: names, data types, obs.

### Build neural netowkr to predict outcomes
df_final = pd.DataFrame(StandardScaler().fit_transform(df_final.values), columns = df_final.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
input = df_final.shape[1] - 4 # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
classifier = Sequential() # Sequential model building in keras
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
X = df_final.drop(columns = ["quant", "train", "test", "test2"]) # Save features as X numpy data array
Y_train = df_final["train"] # Save test outcome as Y numpy data array
classifier.fit(X, Y_train, batch_size = 10, epochs = 500) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_train = (Y_train > 0)
fpr, tpr, threshold = roc_curve(Y_train, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_trainb = auc(fpr, tpr) # Plot ROC and get AUC score
Y_test = df_final["test"] # Save train outcome as Y numpy data array
classifier.fit(X, Y_test, batch_size = 10, epochs = 500) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_test = (Y_test > 0)
fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_testb = auc(fpr, tpr) # Plot ROC and get AUC score
Y_test2 = df_final["test2"] # Save train outcome as Y numpy data array
classifier.fit(X, Y_test2, batch_size = 10, epochs = 500) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_test2 = (Y_test2 > 0)
fpr, tpr, threshold = roc_curve(Y_test2, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_test2b = auc(fpr, tpr) # Plot ROC and get AUC score

### Append to Text File
text_file = open(path + File + "_results.txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write(str(res.summary())) # Line of text with space after
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.write("Hospital Received Penalty FY 2018: C-Statistic = " + str(auc_test2) + "\n") # Line of text with space after
text_file.write("Hospital Received Penalty FY 2019: C-Statistic = " + str(auc_test) + "\n") # Line of text with space after
text_file.write("Hospital Received Penalty FY 2020: C-Statistic = " + str(auc_train) + "\n") # Line of text with space after
text_file.write("\n") # Add two lines of blank text at end of every section text
text_file.write("Hospital Received Bonus FY 2018: C-Statistic = " + str(auc_test2b) + "\n") # Line of text with space after
text_file.write("Hospital Received Bonus FY 2019: C-Statistic = " + str(auc_testb) + "\n") # Line of text with space after
text_file.write("Hospital Received Bonus FY 2020: C-Statistic = " + str(auc_trainb) + "\n") # Line of text with space after
text_file.close() # Close file