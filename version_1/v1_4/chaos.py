# FractureProof Chaos
label = "_chaos"
path = "fp/v1_4/"
version = "FractureProof v1.4"
title = "Chaos: Finding Significant Predictors in Multi-Level Geographic Data"
author = "DrewC!"
day = str(date.today())
stamp = str(dt.datetime.now())

## Setup Workspace

### Import python libraries
import os # Operating system navigation
from datetime import date # date stamping in Python
from datetime import time # time stamping in Python

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes
import statsmodels.api as sm # Statistics package best for regression models

### Import scikit-learn libraries
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

### Import keras libraries
from keras.models import Sequential # Uses a simple method for building layers in MLPs
from keras.models import Model # Uses a more complex method for building layers in deeper networks
from keras.layers import Dense # Used for creating dense fully connected layers
from keras.layers import Input # Used for designating input layers

### Set Directory
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

### Setup Script Results File
text_file = open(path + day + "_results" + label + ".txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write("####################" + "\n\n")
text_file.write(title + "\n") # Line of text with space after
text_file.write(version + "\n") # Line of text with space after
text_file.write(author + "\n") # Line of text with space after
text_file.write(stamp + "\n") # Line of text with space after
text_file.write("\n" + "####################" + "\n\n")
text_file.close() # Close file

# Step 1: Raw Data Processing
sub = "Step 1: Raw Data Processing and Feature Engineering"
y = "Final Payment Adjustments FY 2018-2020"
a = "CMS Hospital Compare 2018 release "
d = "CMS Hospital General Info 2018 release"
c = "US CENSUS American Community Survey 5 Year Average by Zip Code 2018 release"
g = "HRSA Area Health Resource File by County 2018 release"

## Process Raw Outcome Data

### Individual Features and Targets
df_raw = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_raw['Facility ID'] = df_raw['Facility ID'].astype("str") # Change data type of column in data frame
df_raw = df_raw.dropna(subset = ["2020 VBP Adjustment Factor"])
df_raw["train"] = np.where(df_raw["2020 VBP Adjustment Factor"] < 1, 1, 0) # Create categorical test target outcome based on conditions
df_raw["test"] = np.where(df_raw["2019 VBP Adjustment Factor"] < 1, 1, 0) # Create categorical test target outcome based on conditions
df_raw["test2"] = np.where(df_raw["2018 VBP Adjustment Factor"] < 1, 1, 0) # Create categorical test target outcome based on conditions
df_raw["quant"] = df_raw["2020 VBP Adjustment Factor"] # Rename multiple columns in place
df_raw = df_raw.drop(columns = ["2018 VBP Adjustment Factor", "2019 VBP Adjustment Factor", "2020 VBP Adjustment Factor"]) # Drop quantitative variables used to create target
df_raw.info() # Get class, memory, and column info: names, data types, obs.

### Export Targets
Y_raw = df_raw.filter(["FIPS", "train", "test", "test2", "quant", "Facility ID"])
Y_raw = Y_raw.set_index(["Facility ID", "FIPS"]) # Set column as index
Y_ss = pd.DataFrame(StandardScaler().fit_transform(Y_raw.values), columns = Y_raw.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
Y_raw = Y_raw.reset_index(level = ["Facility ID", "FIPS"]) # Reset Index
Y_ss["Facility ID"] = Y_raw["Facility ID"]
Y_ss["FIPS"] = Y_raw["FIPS"]
Y_ss = Y_ss.set_index(["Facility ID"]) # Set column as index
Y_quant = Y_ss["quant"]
Y_train = Y_ss["train"]
Y_test = Y_ss["test"]
Y_test2 = Y_ss["test2"]
Y_ss.info() # Get class, memory, and column info: names, data types, obs.

## Process Raw Predictor Data

### Individual Agent Predictors
X_a_raw = df_raw.drop(columns = ["Total Performance Score", 
                        "Weighted Normalized Clinical Outcomes Domain Score", 
                        "Weighted Safety Domain Score", 
                        "Weighted Person and Community Engagement Domain Score", 
                        "Weighted Efficiency and Cost Reduction Domain Score", 
                        "Medicare hospital spending per patient (Medicare Spending per Beneficiary)",
                        "Rate of readmission after discharge from hospital (hospital-wide)",
                        "Hospital overall rating",
                        "Hospital Ownership ForProfit",
                        "TOTAL HAC SCORE",
                        "FIPS",
                        "quant",
                        "train",
                        "test",
                        "test2"]) # Drop proximity features: Adjustment factor scores
X_a_raw = X_a_raw.set_index("Facility ID") # Set column as index
X_a_na = X_a_raw.dropna(axis = 1, thresh = 0.75*len(X_a_raw)) # Drop features less than 75% non-NA count for all columns
X_a_na = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(X_a_na), columns = X_a_na.columns) # Impute missing data
X_a = pd.DataFrame(StandardScaler().fit_transform(X_a_na.values), columns = X_a_na.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
X_a_raw = X_a_raw.reset_index(level = ["Facility ID"]) # Reset Index
X_a["Facility ID"] = X_a_raw["Facility ID"]
X_a = X_a.set_index(["Facility ID"]) # Set column as index
X_a.info() # Get class, memory, and column info: names, data types, obs.

### Individual Demographic Predictors
X_d_raw = df_raw.filter(["Medicare hospital spending per patient (Medicare Spending per Beneficiary)",
            "Rate of readmission after discharge from hospital (hospital-wide)",
            "Hospital overall rating",
            "Hospital Ownership ForProfit",
            "TOTAL HAC SCORE",
            "Facility ID"]) # Subset by hand selected features for model
X_d_raw = X_d_raw.set_index("Facility ID") # Set column as index
X_d_na = X_d_raw.dropna(axis = 1, thresh = 0.75*len(X_d_raw)) # Drop features less than 75% non-NA count for all columns
X_d_na = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(X_d_na), columns = X_d_na.columns) # Impute missing data
X_d = pd.DataFrame(StandardScaler().fit_transform(X_d_na.values), columns = X_d_na.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
X_d_raw = X_d_raw.reset_index(level = ["Facility ID"]) # Reset Index
X_d["Facility ID"] = X_d_raw["Facility ID"]
X_d = X_d.set_index(["Facility ID"]) # Set column as index
X_d.info() # Get class, memory, and column info: names, data types, obs.

### Ecological Contextual Predictors
X_c_raw = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_FIPS_gini.csv") # Import dataset saved as csv in _data folder
X_c_raw = pd.merge(Y_raw, X_c_raw, on = "FIPS", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options
X_c_raw = X_c_raw.set_index("Facility ID") # Set column as index
X_c_raw = X_c_raw.drop(columns = ["FIPS", "train", "test", "test2", "quant"]) # Drop quantitative variables used to create target
X_c_na = X_c_raw.dropna(axis = 1, thresh = 0.75*len(X_c_raw)) # Drop features less than 75% non-NA count for all columns
X_c_na = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(X_c_na), columns = X_c_na.columns) # Impute missing data
X_c = pd.DataFrame(StandardScaler().fit_transform(X_c_na.values), columns = X_c_na.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
X_c_raw = X_c_raw.reset_index(level = ["Facility ID"]) # Reset Index
X_c["Facility ID"] = X_c_raw["Facility ID"]
X_c = X_c.set_index(["Facility ID"]) # Set column as index
X_c.info() # Get class, memory, and column info: names, data types, obs.X_g.info() # Get class, memory, and column info: names, data types, obs.

### Ecological Global Predictors
X_g_raw = pd.read_csv("hnb/HRSA/AHRF/AHRF_2018_2019_SAS/AHRF_full.csv") # Import dataset saved as csv in _data folder
X_g_raw = X_g_raw.loc[:, X_g_raw.columns.str.contains('2018|2017|2016|2015|2014|FIPS')] # Select columns by string value
X_g_raw = pd.merge(Y_raw, X_g_raw, on = "FIPS", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options
X_g_raw = X_g_raw.set_index("Facility ID") # Set column as index
X_g_raw = X_g_raw.drop(columns = ["FIPS", "train", "test", "test2", "quant"]) # Drop quantitative variables used to create target
X_g_na = X_g_raw.dropna(axis = 1, thresh = 0.75*len(X_g_raw)) # Drop features less than 75% non-NA count for all columns
X_g_na = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(X_g_na), columns = X_g_na.columns) # Impute missing data
X_g = pd.DataFrame(StandardScaler().fit_transform(X_g_na.values), columns = X_g_na.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
X_g_raw = X_g_raw.reset_index(level = ["Facility ID"]) # Reset Index
X_g["Facility ID"] = X_g_raw["Facility ID"]
X_g = X_g.set_index(["Facility ID"]) # Set column as index
X_g.info() # Get class, memory, and column info: names, data types, obs.X_g.info() # Get class, memory, and column info: names, data types, obs.

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(sub + "\n\n") # Line of text with space after
text_file.write(y + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Quantitative = 2020 VBP Adjsutment Factor" + "\n")
text_file.write("   Binary = 0/1, No/Yes, Penalty Applied" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   quant, train, test, test2 = FY2020(Q), FY2020(B), FY2019(B), FY2018(B)" + "\n")
text_file.write(str(Y_raw.describe())  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(a + "\n") # Add two lines of blank text at end of every section text
text_file.write("   (Rows, Columns) = " + str(X_a.shape) + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Level = Hospital" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Year = 2018" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Feature Engineeering = 75% nonNA, Median Imputed NA, Standard Scaled" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(d + "\n") # Add two lines of blank text at end of every section text
text_file.write("   (Rows, Columns) = " + str(X_d.shape) + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Level = Hospital" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Year = 2018" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Feature Engineeering = 75% nonNA, Median Imputed NA, Standard Scaled" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(c + "\n") # Add two lines of blank text at end of every section text
text_file.write("   (Rows, Columns) = " + str(X_c.shape) + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Level = Hospital" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Year = 2018 relese, includes 5 year average for 2014-2018" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Feature Engineeering = Gini index and Average of Zip Codes by County, 75% nonNA, Median Imputed NA, Standard Scaled" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(g + "\n") # Add two lines of blank text at end of every section text
text_file.write("   (Rows, Columns) = " + str(X_g.shape) + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Level = County" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Year = 2018 release, kept features from 2015-2018" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Feature Engineeering = 75% nonNA, Median Imputed NA, Standard Scaled" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

# Step 2: Intital Prediction with Closed Box Models
sub2 = "Step 2: Initial Prediction with Closed Models"
m1 = "Multi-Layer Perceptron"
m2 = " with Autoencoder"

## Mutli-Layer Perceptron for Individual Agencies

### Build Network with keras Sequential API
# Prep Inputs
input = X_a.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
network = Sequential()
# Dense Layers
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
# Activation Layer
network.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
network.fit(X_a, Y_train, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
# Predict
Y_a = network.predict(X_a) # Predict values from testing model
# AUC Score
fpr, tpr, threshold = roc_curve((Y_train > 0), (Y_a > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Ia_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test > 0), (Y_a > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Ia_test = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test2 > 0), (Y_a > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Ia_test2 = auc(fpr, tpr) # Plot ROC and get AUC score

## Mutli-Layer Perceptron for Individual Demographics

### Build Network with keras Sequential API
# Prep Inputs
input = X_d.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
network = Sequential()
# Dense Layers
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
# Activation Layer
network.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
network.fit(X_d, Y_train, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
# Predict
Y_d = network.predict(X_d) # Predict values from testing model
# AUC Score
fpr, tpr, threshold = roc_curve((Y_train > 0), (Y_d > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Id_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test > 0), (Y_d > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Id_test = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test2 > 0), (Y_d > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Id_test2 = auc(fpr, tpr) # Plot ROC and get AUC score

## MLP with stacked autoencoder for ecological contexts

### Build Network with keras Functional API
# Prep Inputs
input = X_c.shape[1]# Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
# Input layer
i = Input(shape = (input,))
# Encoder
x = Dense(800, activation = 'relu')(i)
x = Dense(200, activation = 'relu')(x)
x = Dense(50, activation = 'relu')(x)
x = Dense(10, activation = 'relu')(x)
# Decoder
x = Dense(10, activation = 'relu')(x)
x = Dense(50, activation = 'relu')(x)
x = Dense(200, activation = 'relu')(x)
x = Dense(800, activation = 'relu')(x)
# Dense Layers
x = Dense(nodes, activation = 'relu')(x)
x = Dense(nodes, activation = 'relu')(x) # First Hidden Layer
# Output layer
x = Dense(1, activation = 'sigmoid')(x) # Output Layer
# Save network structure
network = Model(i, x)
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
network.fit(X_c, Y_train, batch_size = 10, epochs = 5) # Fitting the data to the train outcome
# Predict
Y_c = network.predict(X_c) # Predict values from testing model
# AUC Score
fpr, tpr, threshold = roc_curve((Y_train > 0), (Y_c > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Ic_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test > 0), (Y_c > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Ic_test = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test2 > 0), (Y_c > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Ic_test2 = auc(fpr, tpr) # Plot ROC and get AUC score

## MLP with stacked autoencoder for Ecological globals

### Build Network with keras Functional API
# Prep Inputs
input = X_g.shape[1]# Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
# Input layer
i = Input(shape = (input,))
# Encoder
x = Dense(1600, activation = 'relu')(i)
x = Dense(400, activation = 'relu')(x)
x = Dense(100, activation = 'relu')(x)
x = Dense(25, activation = 'relu')(x)
# Decoder
x = Dense(25, activation = 'relu')(x)
x = Dense(100, activation = 'relu')(x)
x = Dense(400, activation = 'relu')(x)
x = Dense(1600, activation = 'relu')(x)
# Dense Layers
x = Dense(nodes, activation = 'relu')(x)
x = Dense(nodes, activation = 'relu')(x) # First Hidden Layer
# Output layer
x = Dense(1, activation = 'sigmoid')(x) # Output Layer
# Save network structure
network = Model(i, x)
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
network.fit(X_g, Y_train, batch_size = 10, epochs = 10) # Fitting the data to the train outcome
# Predict
Y_g = network.predict(X_g) # Predict values from testing model
# AUC Score
fpr, tpr, threshold = roc_curve((Y_train > 0), (Y_g > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Ig_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test > 0), (Y_g > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Ig_test = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test2 > 0), (Y_g > 0.5)) # Create ROC outputs, true positive rate and false positive rate
Ig_test2 = auc(fpr, tpr) # Plot ROC and get AUC score

## Export Intitial Prediction Results

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(sub2 + "\n\n") # Line of text with space after
text_file.write(a + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Network = " + m1 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Layers = Dense, Dense, Activation" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Functions = ReLU, ReLU, Sigmoid" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Epochs = 100" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Targets = (train, test, test2), (FY2020, FY2019, FY2018)" + "\n")
text_file.write("   AUC Scores" + "\n") # Add two lines of blank text at end of every section text
text_file.write("       train = " + str(Ia_train) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(Ia_test) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(Ia_test2) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(d + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Network = " + m1 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Layers = Dense, Dense, Activation" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Functions = ReLU, ReLU, Sigmoid" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Targets = (train, test, test2), (FY2020, FY2019, FY2018)" + "\n")
text_file.write("   Epochs = 100" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   AUC Scores" + "\n") # Add two lines of blank text at end of every section text
text_file.write("       train = " + str(Id_train) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(Id_test) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(Id_test2) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(c + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Network = " + m1 + m2 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Layers = Input, Decoder, Encoder, Dense, Dense, Activation" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Functions = Input,ReLU, ReLU, ReLU, ReLU, Sigmoid" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Targets = (train, test, test2), (FY2020, FY2019, FY2018)" + "\n")
text_file.write("   Epochs = 10" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   AUC Scores" + "\n") # Add two lines of blank text at end of every section text
text_file.write("       train = " + str(Ic_train) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(Ic_test) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(Ic_test2) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(g + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Network = " + m1 + m2 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Layers = Input, Decoder, Encoder, Dense, Dense, Activation" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Functions = Input,ReLU, ReLU, ReLU, ReLU, Sigmoid" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Targets = (train, test, test2), (FY2020, FY2019, FY2018)" + "\n")
text_file.write("   Epochs = 10" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   AUC Scores" + "\n") # Add two lines of blank text at end of every section text
text_file.write("       train = " + str(Ig_train) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(Ig_test) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(Ig_test2) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

# Step 3: Identify Predictors with Open Box Models
sub3 = "Step 3: Identify Predictors with Open Models"
m3 = "Principal Component Analysis"
m4 = "Random Forests"
m5 = "Recursive feature Elimination"
m6 = "Multiple Regression"

## Identify Predictors for Individual Agents

### Principal Component Analysis
degree = len(X_a.columns) - 1  # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(X_a) # Fit initial PCA model
df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
components = len(df_comp.index) - 3 # Save count of components for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(X_a) # finally call fit_transform on the aggregate data to create PCA results object
df_pc = pd.DataFrame(pca.components_, columns = X_a.columns) # Export eigenvectors to data frame with column names from original data
df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pc = df_pc.abs() # Get absolute value of eigenvalues
df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
df_p = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_p = df_p[df_p.MaxEV > df_p.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_p = df_p.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
df_pca = df_p.rename(columns = {"index": "Features"}) # Rename former index as features
df_pca = df_pca.sort_values(by = ["MaxEV"], ascending = False) # Sort Columns by Value

### Random Forest Regressor
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X_a, Y_quant) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X_a, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean
df_rf = df_rf.sort_values(by = ["Gini"], ascending = False) # Sort Columns by Value

### Recursive Feature Elimination
df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X_a[pca_rf], Y_quant) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X_a[pca_rf], rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
df_rfe = pd.merge(df_rfe, df_pca_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options

### Multiple Regression
pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X_a[pca_rf_rfe], Y_quant) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X_a[pca_rf_rfe], coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names
df_reg = df_reg.sort_values(by = ["Coefficients"], ascending = False) # Sort Columns by Value

### Export feature attributes
fp_X_a = pd.merge(df_rfe, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options

## Identify Predictors for Individual Demographics

### Multiple Regression
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X_d, Y_quant) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X_d, coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names
df_reg = df_reg.sort_values(by = ["Coefficients"], ascending = False) # Sort Columns by Value

### Export feature attributes for Individual Demographics
fp_X_d = df_reg

## Identify Predictors for Ecological Contexts

### Principal Component Analysis
degree = len(X_c.columns) - 1  # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(X_c) # Fit initial PCA model
df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
components = len(df_comp.index) - 3 # Save count of components for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(X_c) # finally call fit_transform on the aggregate data to create PCA results object
df_pc = pd.DataFrame(pca.components_, columns = X_c.columns) # Export eigenvectors to data frame with column names from original data
df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pc = df_pc.abs() # Get absolute value of eigenvalues
df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
df_p = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_p = df_p[df_p.MaxEV > df_p.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_p = df_p.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
df_pca = df_p.rename(columns = {"index": "Features"}) # Rename former index as features
df_pca = df_pca.sort_values(by = ["MaxEV"], ascending = False) # Sort Columns by Value

### Random Forest Regresson
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X_c, Y_quant) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X_c, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean
df_rf = df_rf.sort_values(by = ["Gini"], ascending = False) # Sort Columns by Value

### Recursive Feature Elimination
df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X_c[pca_rf], Y_quant) # This will take time
rfe = recursive.ranking_ # Save Boolean values as numpy array
l_rfe = list(zip(X_c[pca_rf], rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe.sort_values(by = ["RFE"], ascending = True) # Sort Columns by Value
df_rfe = df_rfe[df_rfe["RFE"] <= 1] # Select Variables that were True
df_rfe = pd.merge(df_rfe, df_pca_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options

### Multiple Regression
pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X_c[pca_rf_rfe], Y_quant) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X_c[pca_rf_rfe], coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names
df_reg = df_reg.sort_values(by = ["Coefficients"], ascending = False) # Sort Columns by Value

### Export feature attributes
fp_X_c = pd.merge(df_rfe, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options

## Identify Predictors for Ecological Globals

### Principal Component Analysis
degree = len(X_g.index) - 1  # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(X_g) # Fit initial PCA model
df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
components = len(df_comp.index) - 3 # Save count of components for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(X_g) # finally call fit_transform on the aggregate data to create PCA results object
df_pc = pd.DataFrame(pca.components_, columns = X_g.columns) # Export eigenvectors to data frame with column names from original data
df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pc = df_pc.abs() # Get absolute value of eigenvalues
df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
df_p = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_p = df_p[df_p.MaxEV > df_p.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_p = df_p.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
df_pca = df_p.rename(columns = {"index": "Features"}) # Rename former index as features
df_pca = df_pca.sort_values(by = ["MaxEV"], ascending = False) # Sort Columns by Value

### Random Forest Regresson
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X_g, Y_quant) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X_g, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean
df_rf = df_rf.sort_values(by = ["Gini"], ascending = False) # Sort Columns by Value

### Recursive Feature Elimination
df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X_g[pca_rf], Y_quant) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X_g[pca_rf], rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
df_rfe = pd.merge(df_rfe, df_pca_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options

### Multiple Regression
pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X_g[pca_rf_rfe], Y_quant) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X_g[pca_rf_rfe], coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names
df_reg = df_reg.sort_values(by = ["Coefficients"], ascending = False) # Sort Columns by Value

### Export feature attributes
fp_X_g = pd.merge(df_rfe, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(sub3 + "\n\n") # Line of text with space after
text_file.write(a + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Models = " + m3 + m4 + m5 + m6 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Values = Eigenvectors, Gini Impurity, True, OLS" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Thresholds = Mean, Mean, Cross Validation, All" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Outcome = quant, 2020 VBP Adjustment Factor" + "\n")
text_file.write(str(fp_X_a)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(d + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Models = " m6 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Values = OLS" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Thresholds = All" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Outcome = quant, 2020 VBP Adjustment Factor" + "\n")
text_file.write(str(fp_X_d)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(c + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Models = " + m3 + m4 + m5 + m6 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Values = Eigenvectors, Gini Impurity, True, OLS" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Thresholds = Mean, Mean, Cross Validation, All" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Outcome = quant, 2020 VBP Adjustment Factor" + "\n")
text_file.write(str(fp_X_c)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(g + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Models = " + m3 + m4 + m5 + m6 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Values = Eigenvectors, Gini Impurity, Rank, OLS" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Thresholds = Mean, Mean, Cross Validation, All" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Outcome = quant, 2020 VBP Adjustment Factor" + "\n")
text_file.write(str(fp_X_g)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

# Step 2: Final Prediction with Open and Closed Box Models
sub4 = "Step 2: Final Prediction with Open and Closed Box Models"
f = "Final Model of Identified Predictors"

## Isolate Identified Predictors from Raw Data for Final Models

### Pull final feature list from raw data
f_a = fp_X_a["Features"].tolist()
f_d = fp_X_d["Features"].tolist()
f_c = fp_X_c["Features"].tolist()
f_g = fp_X_g["Features"].tolist()
X_f_a = X_a_raw[f_a]
X_f_d = X_d_raw[f_d]
X_f_c = X_c_raw[f_c]
X_f_g = X_g_raw[f_g]

### Join raw predictors and raw outcome data using Facility ID index
X_f_a = X_f_a.reset_index()
X_f_d = X_f_d.reset_index()
X_f_c = X_f_c.reset_index()
X_f_g = X_f_g.reset_index()
X_i = pd.merge(X_f_a, X_f_d, on = "Facility ID", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
X_e = pd.merge(X_f_g, X_f_c, on = "Facility ID", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
X_f = pd.merge(X_e, X_i, on = "Facility ID", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
Y_f = Y_raw.reset_index()
df_f = pd.merge(Y_f, X_f, on = "Facility ID", how = "outer") # Join by column while keeping only items that exist in both, select outer or left for other options
df_f = df_f.set_index("Facility ID") # Set column as index

### Tidy for Multiple Regression, Scale for Multi-layer Perceptron
df_f = df_f.drop(columns = ["FIPS"]) # Drop Unwanted Columns
df_f = df_f.dropna()
ss_f = pd.DataFrame(StandardScaler().fit_transform(df_f.values), columns = df_f.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.

## Compare Final Predictors with Regression Model 

### Build Regression Model
Y = df_f.filter(["quant"])
X = df_f.drop(columns = ["quant", "train", "test", "test2"])
mod = sm.OLS(Y, X) # Describe linear model
res_f = mod.fit() # Fit model

## Mutli-Layer Perceptron for Indetified Final Predictors

### Build Network with keras Sequential API
# Prep Inputs
Y_train = ss_f.filter(["train"])
Y_test = ss_f.filter(["test"])
Y_test2 = ss_f.filter(["test2"])
X = ss_f.drop(columns = ["quant", "train", "test", "test2"])
input = X.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
network = Sequential()
# Dense Layers
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
# Activation Layer
network.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
network.fit(X, Y_train, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
# Predict
Y_f = network.predict(X) # Predict values from testing model
# AUC Score
Y_pred = (Y_f > 0.5)
Y_train = (Y_train > 0)
fpr, tpr, threshold = roc_curve(Y_train, Y_pred) # Create ROC outputs, true positive rate and false positive rate
f_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
f_test = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve(Y_test2, Y_pred) # Create ROC outputs, true positive rate and false positive rate
f_test2 = auc(fpr, tpr) # Plot ROC and get AUC score

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(sub4 + "\n\n") # Line of text with space after
text_file.write(f + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Models = " m6 + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Outcome = quant, 2020 VBP Adjustment Factor" + "\n")
text_file.write(str(res_f.summary())  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(f + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Network = " + m1 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Layers = Dense, Dense, Activation" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Functions = ReLU, ReLU, Sigmoid" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Epochs = 100" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Targets = (train, test, test2), (FY2020, FY2019, FY2018)" + "\n")
text_file.write("   AUC Scores" + "\n") # Add two lines of blank text at end of every section text
text_file.write("       train = " + str(f_train) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(f_test) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(f_test2) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file