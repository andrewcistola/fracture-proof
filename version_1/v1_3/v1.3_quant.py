# FractureProof
## Version 1.3

title = "fp_v1.3"
path = "fp/fp_v1.3/"

## Section A: Collect Possible Predictors from Public Access Data

### Import Python Libraries
import os # Operating system navigation
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes
import statsmodels.api as sm # Statistics package best for regression models for statistical tests
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
from sklearn.ensemble import RandomForestRegressor # Random Forest classification component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.linear_model import LogisticRegression # Used for machine learning with categorical outcome
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome
from sklearn.metrics import confusion_matrix # Ue for evaluating model prediction scores with true/false positives and negatives
from sklearn.metrics import roc_curve # Reciever operator curve
from sklearn.metrics import auc # Area under the curve 
from keras import Sequential # Sequential neural network modeling
from keras.layers import Dense # Used for creating layers within neural network

### Set working directory to local folder
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

### Data Processing
df_s1 = pd.read_csv("hnb/.csv", low_memory = 'false') # Import predictor and outcome data from csv
df_s1 = df_s1.drop(columns = ["(geographical key)"]) # Drop ID variables
df_s1 = df_s1.drop(columns = ["(proximity features)"]) # Drop proximity features
df_s1 = df_s1.rename(columns = {"(quantitative outcome)": "quant"}) # Rename multiple columns in place
df_s1 = df_s1.rename(columns = {"(training target)": "train"}) # Rename multiple columns in place
df_s1 = df_s1.rename(columns = {"(testing target)": "test"}) # Rename multiple columns in place
df_s1["train"] = np.where(df_s1["quant"] <= 1, 1, 0) # Create categorical test target outcome based on conditions
df_s1["test"] = np.where(df_s1["test"] <= 1, 1, 0) # Create categorical train target outcome based on conditions
quant = df_s1.pop("quant") # Remove quantitative outcome
train = df_s1.pop("train") # Remove quantitative outcome
test = df_s1.pop("test") # Remove quantitative outcome
df_s1 = df_s1.dropna(axis = 1, thresh = 0.75*len(df_s1)) # Drop features less than 75% non-NA count for all columns
df_s1 = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_s1), columns = df_s1.columns) # Impute missing data
df_s1 = pd.DataFrame(StandardScaler().fit_transform(df_s1.values), columns = df_s1.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_s1.insert(0, "quant", quant) # Reattach qunatitative outcome to front of data frame
df_s1.insert(0, "train", train) # Reattach outcome
df_s1.insert(0, "test", test) # Reattach outcome
df_s1 = df_s1.dropna() # Drop all rows with NA values (should be none, this is just to confirm)

### Principal Component Analysis
df_pca = df_s1.drop(columns = ["quant", "train", "test"]) # Drop outcomes and targets
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

### Random Forest Regressor
Y = df_s1["quant"] # Isolate Outcome variable
X = df_s1.drop(columns = ["quant", "train", "test"]) # Drop outcomes and targets
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X, Y) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean

### Recursive Feature Elimination
df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
X = df_s1[pca_rf] # Save features columns as predictor data frame
Y = df_s1["quant"] # Selected quantitative outcome from original data frame
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X, Y) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True

### Multiple Regression
pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
X = df_s1.filter(pca_rf_rfe) # Keep only selected columns from rfe
Y = df_s1["quant"] # Add outcome variable
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X, Y) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X, coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names

### Export Results
df_pc_rf_rfe_reg = pd.merge(df_pca_rf, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_pc_rf_reg.to_csv(r"fp/results.csv") # Export df as csv
print(df_pc_rf_reg)

### Build Model of predictive features
features = ["", 
            "", 
            "", 
            "",
            "", 
            "quant",
            "test",
            "test"] # Hand select features from results table
df_sub = df_s1.filter(features) # Subset by hand selected features for model
X = df_sub.drop(columns = ["quant", "train", "test"]) # features as x
Y = df_sub["quant"] # Save outcome variable as y
mod = sm.OLS(Y, X) # Describe linear model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

### Build Neural Network to Predict categorial outcome
input = df_sub.shape[1] - 2 # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
classifier = Sequential() # Sequential model building in keras
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
X = df_sub.drop(columns = ["train", "test"]) # Save features as X numpy data array
Y_test = df_sub["test"] # Save train outcome as Y numpy data array
Y_train = df_sub["test"] # Save test outcome as Y numpy data array
classifier.fit(X, Y_train, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
eval_train = classifier.evaluate(X, Y_train) # Evaluate loss and accuracy of training model
classifier.fit(X, Y_test, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
eval_test = classifier.evaluate(X, Y_train) # Evaluate loss and accuracy of training model
true_false = confusion_matrix(Y_test, Y_pred) # True Positive - False Positive / False Negative - True Negative
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5) # If the prediction is greater than 0.5 then the output is 1 else the output is 0
fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc = auc(fpr, tpr) # Plot ROC and get AUC score

### Export prediction scores
print(eval_train)
print(eval_test)
print(true_false)
print(auc)





