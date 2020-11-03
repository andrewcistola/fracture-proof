File = "neural.py"
path = "fp/v1_4/dev/"
title = "FractureProof Neural Netowrk Feature Identification"
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
from keras.models import Sequential # Uses a simple method for building layers in MLPs
from keras.models import Model # Uses a more complex method for building layers in deeper networks
from keras.layers import Dense # Used for creating dense fully connected layers
from keras.layers import Conv2D # Used for creaing convolutional layers
from keras.layers import Input # Used for designating input layers
from keras.layers import Dropout # Used for creating dropout layers
from keras.layers import concatenate # Used to combine inputs from multiple layers

### Set Directory
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

## Raw Data Processing

### Individual Features and Targets
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_cms['Facility ID'] = df_cms['Facility ID'].astype("str") # Change data type of column in data frame
df_cms["train"] = np.where(df_cms["2020 VBP Adjustment Factor"] > 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms["test"] = np.where(df_cms["2019 VBP Adjustment Factor"] > 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms["test2"] = np.where(df_cms["2018 VBP Adjustment Factor"] > 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms = df_cms.drop(columns = ["2018 VBP Adjustment Factor", "2019 VBP Adjustment Factor", "2020 VBP Adjustment Factor"]) # Drop quantitative variables used to create target
df_cms.info() # Get class, memory, and column info: names, data types, obs.

### Individual Agent Predictors
df_hc = df_cms.drop(columns = ["Total Performance Score", 
                        "Weighted Normalized Clinical Outcomes Domain Score", 
                        "Weighted Safety Domain Score", 
                        "Weighted Person and Community Engagement Domain Score", 
                        "Weighted Efficiency and Cost Reduction Domain Score", 
                        "Medicare hospital spending per patient (Medicare Spending per Beneficiary)",
                        "Rate of readmission after discharge from hospital (hospital-wide)",
                        "Hospital overall rating",
                        "Hospital Ownership ForProfit",
                        "TOTAL HAC SCORE",
                        "Facility ID",
                        "FIPS",
                        "train",
                        "test",
                        "test2"]) # Drop proximity features: Adjustment factor scores
df_hc = df_hc.dropna(axis = 1, thresh = 0.75*len(df_hc)) # Drop features less than 75% non-NA count for all columns
df_hc = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_hc), columns = df_hc.columns) # Impute missing data
df_hc = pd.DataFrame(StandardScaler().fit_transform(df_hc.values), columns = df_hc.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_hc.info() # Get class, memory, and column info: names, data types, obs.

### Individual Demographic Predictors
df_gen = df_cms.filter(["Medicare hospital spending per patient (Medicare Spending per Beneficiary)",
            "Rate of readmission after discharge from hospital (hospital-wide)",
            "Hospital overall rating",
            "Hospital Ownership ForProfit",
            "TOTAL HAC SCORE"]) # Subset by hand selected features for model
df_gen = df_gen.dropna(axis = 1, thresh = 0.75*len(df_gen)) # Drop features less than 75% non-NA count for all columns
df_gen = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_gen), columns = df_gen.columns) # Impute missing data
df_gen = pd.DataFrame(StandardScaler().fit_transform(df_gen.values), columns = df_gen.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_gen.info() # Get class, memory, and column info: names, data types, obs.

### Ecological Global Predictors
df_hrsa = pd.read_csv("hnb/HRSA/AHRF/AHRF_2018_2019_SAS/AHRF_full.csv") # Import dataset saved as csv in _data folder
df_hrsa = df_hrsa.set_index("FIPS") # Set column as index
df_hrsa = df_hrsa.loc[:, df_hrsa.columns.str.contains('2018|2017|2016|2015|2014')] # Select columns by string value
df_hrsa = df_hrsa.reset_index(level = ["FIPS"]) # Reset Index
df_Y = df_cms.filter(["FIPS", "train", "test", "test2"])
df_hrsa = pd.merge(df_Y, df_hrsa, on = "FIPS", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options
df_hrsa = df_hrsa.drop(columns = ["FIPS", "train", "test", "test2"]) # Drop quantitative variables used to create target
df_hrsa = df_hrsa.dropna(axis = 1, thresh = 0.75*len(df_hrsa)) # Drop features less than 75% non-NA count for all columns
df_hrsa = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_hrsa), columns = df_hrsa.columns) # Impute missing data
df_hrsa = pd.DataFrame(StandardScaler().fit_transform(df_hrsa.values), columns = df_hrsa.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_hrsa.info() # Get class, memory, and column info: names, data types, obs.

### Ecological Contextual Predictors
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_FIPS_gini.csv") # Import dataset saved as csv in _data folder
df_Y = df_cms.filter(["FIPS", "train", "test", "test2"])
df_acs = pd.merge(df_Y, df_acs, on = "FIPS", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options
df_acs = df_acs.drop(columns = ["FIPS", "train", "test", "test2"]) # Drop quantitative variables used to create target
df_acs = df_acs.dropna(axis = 1, thresh = 0.75*len(df_acs)) # Drop features less than 75% non-NA count for all columns
df_acs = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_acs), columns = df_acs.columns) # Impute missing data
df_acs = pd.DataFrame(StandardScaler().fit_transform(df_acs.values), columns = df_acs.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_acs.info() # Get class, memory, and column info: names, data types, obs.

### Export Targets
Y_train = df_Y["train"]
Y_test = df_Y["test"]
Y_test2 = df_Y["test2"]

## Mutli-Layer Perceptron

### Build Network with keras Sequential API
# Prep Inputs
X = df_acs # Save features as X numpy data array
input = df_acs.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
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
Y_pred = network.predict(X) # Predict values from testing model

## MLP with stacked autoencoder

### Build Network with keras Functional API
# Prep Inputs
X = df_ # Save features as X numpy data array
input = df_.shape[1]# Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
# Input layer
i = Input(shape = (input,))
# Encoder
x = Dense(400, activation = 'relu')(i)
x = Dense(200, activation = 'relu')(x)
x = Dense(100, activation = 'relu')(x)
x = Dense(50, activation = 'relu')(x)
x = Dense(20, activation = 'relu')(x)
#Decoder
x = Dense(20, activation = 'relu')(x)
x = Dense(50, activation = 'relu')(x)
x = Dense(100, activation = 'relu')(x)
x = Dense(200, activation = 'relu')(x)
x = Dense(400, activation = 'relu')(i)
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
network.fit(X, Y_train, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
# Predict
Y_pred = network.predict(X) # Predict values from testing model

##  Mutli-Input Convolutional Neural Network

### Build Network with keras Functional API
# Prep Input Tables
X_a = df_hc # Save features as X numpy data array
X_d = df_gen # Save features as X numpy data array
X_c = df_acs # Save features as X numpy data array
X_g = df_hrsa # Save features as X numpy data array
# Prep Input dimensions
agent = df_hc.shape[1] # Save number of columns as length minus quant, test, train
demo = df_demo.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
context = df_acs.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
globe = df_hrsa.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
# Individual Agent Layers
# Input layer
a = Input(shape = (agent),))
# Encoder
x = Dense(40, activation = 'relu')(a)
x = Dense(20, activation = 'relu')(x)
x = Dense(10, activation = 'relu')(x)
#Decoder
x = Dense(10, activation = 'relu')(x)
x = Dense(20, activation = 'relu')(x)
x = Dense(40, activation = 'relu')(x)
# Individual Demographic Layers
# Input layer
d = Input(shape = (demo,))
# Encoder
x = Dense(4, activation = 'relu')(d)
x = Dense(2, activation = 'relu')(x)
#Decoder
x = Dense(2, activation = 'relu')(x)
x = Dense(4, activation = 'relu')(x)
# Ecological Context Layers
# Input layer
c = Input(shape = (context,))
# Encoder
x = Dense(40, activation = 'relu')(c)
x = Dense(20, activation = 'relu')(x)
x = Dense(10, activation = 'relu')(x)
#Decoder
x = Dense(10, activation = 'relu')(x)
x = Dense(20, activation = 'relu')(x)
x = Dense(40, activation = 'relu')(x)
# Ecological Global Layers
# Input layer
g = Input(shape = (globe,))
# Encoder
x = Dense(40, activation = 'relu')(g)
x = Dense(20, activation = 'relu')(x)
x = Dense(10, activation = 'relu')(x)
# Decoder
x = Dense(10, activation = 'relu')(x)
x = Dense(20, activation = 'relu')(x)
x = Dense(40, activation = 'relu')(x)
# Create Models
mlp_a = models.create_mlp(agent, regress = False)
mlp_d = models.create_mlp(demo, regress = False)
mlp_c = models.create_mlp(context, regress = False)
mlp_g = models.create_mlp(globe, regress = False)
# Combine MLPs
w = concatenate([mlp_a.output, mlp_d.output, mlp_c.output, mlp_g.output])
# Combined Dense Layers
x = Dense(nodes, activation = 'relu')(w)
x = Dense(nodes, activation = 'relu')(x) # First Hidden Layer
# Output layer
x = Dense(1, activation = 'sigmoid')(x) # Output Layer
# Save network structure
network = Model(input = [mlp_a.input, mlp_d.input, mlp_c.input, mlp_g.input])
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
network.fit(x = [X_a, X_d, X_c, X_g], y = Y_train, batch_size = 10, epochs = 2)
# Predict
Y_pred = network.predict(x = [X_a, X_d, X_c, X_g]) # Predict values from testing model

## Export Results

### AUC Score
Y_pred = (Y_pred > 0.5)
Y_train = (Y_train > 0)
fpr, tpr, threshold = roc_curve(Y_train, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_test = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve(Y_test2, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_test2 = auc(fpr, tpr) # Plot ROC and get AUC score
print("FY 2020 = ", auc_train, " FY 2019 = ", auc_test, " FY 2018 = ", auc_test2)

### Extract Weights to Compare Feature
w1 = network.layers[0].get_weights()[0]
df_net = pd.DataFrame(w1) # Create data frame of importances with variables and gini column names
df_net = df_net.abs() # Get average of all columns by row
df_net["Weight"] = df_net.sum(axis = 1) # Get average of all columns by row
df_net["Features"] = X.columns
df_net = df_net.filter(["Weight", "Features"]) # Subset by hand selected features for model
df_net = df_net.sort_values(by = ["Weight"], ascending = False) # Sort Columns by Value
print(df_net)





























### Sequentyial Kera API

model = Sequential([Flatten(input_shape=(28, 28)), Dense(128, activation=’relu’), Dropout(0.2), Dense(10, activation=’softmax’)])
model.compile(optimizer=’adam’, loss=’sparse_categorical_crossentropy’, metrics=[‘accuracy’])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

### My Version
input = df_ann.shape[1] - 2 # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
classifier = Sequential() # Sequential model building in keras
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
classifier.fit(X, Y_train, batch_size = 10, epochs = 50) # Fitting the data to the train outcome


# Build the model using the functional API
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)
model = Model(i, x)

model.compile(optimizer=’adam’, loss=’sparse_categorical_crossentropy’, metrics=[‘accuracy’])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15





## Autoecnoders

### Encoder
encoder = Sequential()
encoder.add(Dense((nodes / 2), activation = "relu"))
encoder.add(Dense((nodes / 4), activation = "relu"))
encoder.add(Dense((nodes / 8), activation = "relu"))
encoder.add(Dense((nodes / 18), activation = "relu"))
encoder.add(Dense((nodes / 32), activation = "relu"))
 
### Decoder
decoder = Sequential()
decoder.add(Dense(50,input_shape = [2],activation='relu'))
decoder.add(Dense(100,activation='relu'))
decoder.add(Dense(200,activation='relu'))
decoder.add(Dense(400,activation='relu'))
decoder.add(Dense(28 * 28, activation="relu"))
decoder.add(Reshape([28, 28]))
 
### Autoencoder
autoencoder = Sequential([encoder, decoder])
autoencoder.compile(loss = "mse")
autoencoder.fit(X, X, epochs = 50)
encoded_2dim = encoder.predict(X_train)
AE = pd.DataFrame(encoded_2dim, columns = ['X1', 'X2']) 
AE['target'] = y_train


