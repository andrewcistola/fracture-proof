# FractureProof
### Outcome 
#### CMS 2019 Medicare Spending per Beneficiary
### Predictors
#### ACS 2014-2018 Zip Code Percent Estimates for 50 States
#### <br/>BEA 2018 County Measures for 50 States

## Section A: Collect Possible Predictors from Public Access Data

### Import Python Libraries
import os # Operating system navigation
import sqlite3 # SQLite database manager

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes

### Import scikit-learn libraries: data preparation 
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data

### Step 1: Import and Join Data

### Set working directory to project folder
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

### Import ACS and BEA
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_bea = pd.read_csv("hnb/BEA/2018/BEA_5Y2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder

### Join ACS and BEA Data
df_join = pd.merge(df_acs, df_bea, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_acs = 0 # Clear variable
df_bea = 0 # Clear variable

### Join with CMS Data
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_join = pd.merge(df_cms, df_join, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_cms = 0 # Clear variable

### Rename and Verify
df_step1 = df_join
df_join = 0
df_step1.info() # Get class, memory, and column info: names, data types, obs.
df_step1.head() # Print first 5 observations

### Step 2: Data Manipulation

### Import Datasets

### Drop ID variables
df_man = df_step1.drop(columns = ["Facility ID", "FIPS"]) # Drop Unwanted Columns

### Rename outcome and test
df_man = df_man.rename(columns = {"2019 VBP Adjustment Factor": "outcome", "2020 VBP Adjustment Factor": "test"}) # Rename multiple columns in place

### Rename and Verify
df_step2 = df_man
df_man = 0
df_step2.info() # Get class, memory, and column info: names, data types, obs.
df_step2.head() # Print first 5 observations

## Step 3: Data Standardization

### Remove outcome and test
df_NA = df_step2
outcome = df_NA.pop("outcome") # 'pop' column from df
test = df_NA.pop("test") # 'pop' column from df

### Drop features with less than 75% data
df_NA = df_NA.dropna(axis = 1, thresh = 0.75*len(df_NA)) # Drop features less than 75% non-NA count for all columns

### Impute missing values
imp = SimpleImputer(strategy = "median") # Build Imputer model. strategy = "mean" or " median" or "most_frequent" or "constant"
df_imp = pd.DataFrame(imp.fit_transform(df_NA)) # Impute missing data
df_imp.columns = df_NA.columns # Rename columns from new dataset
df_NA = df_imp
df_NA = df_NA.dropna() # Drop all rows with NA values
df_imp = 0

### Reattach outcome
df_NA.insert(0, "outcome", outcome) # reinsert in index
df_NA.insert(1, "test", test) # reinsert in index
df_NA = df_NA.dropna() # Drop all rows with NA values

### Rename and Verify
df_step3 = df_NA
df_NA = 0
df_step3.info() # Get class, memory, and column info: names, data types, obs.
df_step3.head() # Print first 5 observations

## Section B: Identify Significant Predictors with Reduction Algorithms

### Import scikit-learn: machine learning
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
from sklearn.ensemble import RandomForestRegressor # Random Forest classification component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome

### Step 4: Principal Component Analysis

### Setup initial PCA model
df_pca = df_step3.drop(columns = ["outcome", "test"]) # Drop outcome variable
degree = len(df_step3.index) - 2 # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # you will pass the number of components to make PCA model based on degrees of freedom

### Fit initial PCA model
pca.fit(df_pca) # fit to data

### Setup final PCA model
df_ev = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_ev = df_ev[(df_ev[0] > 1)] # Save eigenvalues above 1
components = len(df_ev.index) # Save count of values for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model

### Fit final PCA model
pca.fit_transform(df_pca) # finally call fit_transform on the aggregate data to create PCA results object

### Collect feature list from PCA
df_pca2 = pd.DataFrame(pca.components_, columns = df_pca.columns) # Export eigenvectors to data frame
df_pca2["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pca2 = df_pca2[df_pca2.Variance > df_pca2.Variance.mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pca2 = df_pca2.abs() # get absolute value for column or data frame
df_pca3 = pd.DataFrame(df_pca2.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_pc = df_pca3[df_pca3.MaxEV > df_pca3.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_pc = df_pc.reset_index() # Add a new index of ascending values, existing index becomes column named "index"
df_pc =  df_pc.rename(columns = {"index": "Features"}) # Rename multiple columns in place

### Rename and Verify
df_step4 = df_pc
df_step4.info() # Get class, memory, and column info: names, data types, obs.
df_step4.head() # Print first 5 observations

### Step 5: Random Forest Regressor

### Setup RF model
Y = df_step3["outcome"] # Isolate Outcome variable
features = df_step3.columns.drop(["outcome", "test"]) # Drop outcome variable and Geo to isolate all predictor variable names as features
X = df_step3[features] # Save features columns as predictor data frame
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 

### Fit Forest model
forest.fit(X, Y) # This will take time

### Collect features from RF
gini = forest.feature_importances_ # Output importances of features
l_gini = list(zip(X, gini)) # Create list of variables alongside importance scores 
df_gini = pd.DataFrame(l_gini, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_gini = df_gini.sort_values(by = ["Gini"], ascending = False) # Sort data frame by gini value in desceding order
df_gini = df_gini[(df_gini["Gini"] > df_gini["Gini"].mean())] # Subset by Gini values higher than mean

### Rename and Verify
df_step5 = df_gini
df_step5.info() # Get class, memory, and column info: names, data types, obs.
df_step5.head() # Print first 5 observations

### Step 6: Recursive Feature Elimination

### Collect features from RF and PC
df_pc_gini = pd.merge(df_pc, df_gini, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_features = df_pc_gini["Features"] # Save features from data frame
pc_gini_features = df_features.tolist() # Convert to list
df_rfecv = df_step3[pc_gini_features] # Add selected features to df

### Setup RFE model
X = df_rfecv # Save features columns as predictor data frame
Y = df_step3["outcome"] # Use outcome data frame 
RFE = LinearRegression() # Use regression coefficient as estimator
selector = RFECV(estimator = RFE, min_features_to_select = 10) # define selection parameters, in this case all features are selected. See Readme for more ifo

### Fit RFE model
selected = selector.fit(X, Y) # This will take time

### Collect features from RFE model
ar_rfe = selected.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, ar_rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
df_rfe = df_rfe.reset_index() # Reset Index
df_rfe = df_rfe.filter(["Features"]) # Keep only selected columns

### Rename and Verify
df_step6 = df_rfe
df_step6.info() # Get class, memory, and column info: names, data types, obs.
df_step6.head() # Print first 5 observations

### Export to csv
df_pc_gini.to_csv(r"fp/VBHC/CMS_VBHC_BEA_ACS_pc_gini.csv") # Export df as csv
df_rfe.to_csv(r"fp/VBHC/CMS_VBHC_BEA_ACS_rfe.csv") # Export df as csv

## Section C: Evaluate Significant Features with Modeling and Prediction

### Import scikit-learn libraries: regression
from sklearn.linear_model import LogisticRegression # Used for machine learning with categorical outcome
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome

### Import scikit-learn: neural network
from sklearn.neural_network import MLPRegressor

### Step 7: Multiple Regression

### Setup MR Model
features = list(df_step6["Features"]) # Save chosen featres as list
df_mr = df_step3.filter(features) # Keep only selected columns from rfe
df_mr["outcome"] = df_step3["outcome"] # Add outcome variable
x = df_mr[features] # features as x
y = df_mr["outcome"] # Save outcome variable as y
LR = LinearRegression() # Linear Regression in scikit learn

### Fit MR model
regression = LR.fit(x, y) # Fit model

### Collect features from MR model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X, coef)) # Create list of variables alongside RFE value 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names

### Export feature attributes
df_pc_gini_reg = pd.merge(df_pc_gini, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_pc_gini_reg.to_csv(r"fp/VBHC/CMS_VBHC_BEA_ACS_results.csv") # Export df as csv
print(df_pc_gini_reg)

### Collect prediction results
determination = regression.score(x, y) # rsq value, ceofficient of determination
print(determination)

### Rename and Verify
df_step7 = df_pc_gini_reg
df_step7.info() # Get class, memory, and column info: names, data types, obs.
df_step7.head() # Print first 5 observations

### Step 8: Artificial Neural Network

### Setup ANN
rfe_features = list(df_step6["Features"]) # Save chosen featres as list
df_ANN = df_step3.filter(rfe_features) # Keep only selected columns from rfe
df_ANN["outcome"] = df_step3["outcome"] # Add outcome variable
df_ANN["test"] = df_step3["test"] # Add outcome variable
ANN = MLPRegressor(random_state = 1, max_iter = 10000)

### Fit ANN
ANN.fit(df_ANN, df_ANN["outcome"]) # Predict outcomes with off the shelf NN

### Collect ANN prediction results
predict = ANN.score(df_ANN, df_ANN["test"]) # Get prediction score from ANN
print(predict)

## Section D: Display Results with Geographic Visuals

### Import Mapping Libraries
import geopandas as gp # Simple mapping library for csv shape files with pandas like syntax for creating plots using matplotlib 
import matplotlib.pyplot as plt # Comprehensive graphing package in python
import folium # Mapping library with dynamic visuals
import json # library for importing and manipuation json files

### Import Shapefiles and Basemaps
gdf_shape = gp.read_file("maps/topic_geo_shape.shp") # Import shape files from folder with all other files downloaded
map_fl = folium.Map(location = [29.6516, -82.3248], tiles = 'OpenStreetMap', zoom_start = 11) # Florida Open Street map
map_json = json.load(open("crime/crime_neighborhoods.geojson")) # Save as object
folium.GeoJson(js.load(open("topic_shape.geojson")), name = "shape") # Name map in function

### Step 9: Chloropleth Mapping

### Setup CM
gdf_join = pd.merge(gdf_state, df_poly, on = "State", how = "inner") # Geojoins can use pandas merge as long as geo data is first passed in function
gdf_filter = gdf_join.filter(["State", "geometry"]) # Keep only selected columns
gdf_drop = gdf_filter[(gdf_filter.State != "AK") & (gdf_filter.State != "HI")]
map45m = map45.to_crs({'init': 'epsg:4326'}) ### Change projection

### Assemble CM
map = gdf_shape.plot(column = "ColA", cmap = "Blues", figsize = (16, 10), scheme = "equal_interval", k = 9, legend = True)
map.set_axis_off()
map.set_title("Map Title", fontdict = {'fontsize': 20}, loc = "left")
map.get_legend().set_bbox_to_anchor((.6, .4))
plt.savefig("maps/topic_map__fig.jpeg", dpi = 1000)
plt.show()

### Step 10: Interactive Mapping

### Setup IM
df_filter = df_join.filter(["State", "geometry"]) # Keep only selected columns

### Build Chorpleth
chor = choropleth(geo_data = js, data = df, columns = ["GeoID", "ColA"], threshold_scale = [100, 200], key_on = "feature.geoid", fill_color = "Blues", fill_opacity = 0.7, legend_name = "ColA Values").add_to(map) # Folium choropleth map

### Build Markers
for lat, lon, value in zip(df_["Lat"], df_["Lon"], df_["Value"]):
     fol.Marker(location = [lat, lon], popup = value, color = "blue").add_to(map) # For loop for creation of markers

### Export IM to HTML
map.save("_fig/crime_chi_map.html") # Save map as html file