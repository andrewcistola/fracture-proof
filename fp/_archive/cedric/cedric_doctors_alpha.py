

##########

## State names ACS
df_acs =  df_acs.rename(columns = {"State": "Lower"}) # Rename column
df_acs = pd.read_csv("_data/health_economic_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_acs.info() # Get class, memory, and column info: names, data types, obs.








## Reimport
df_acs = pd.read_csv("_data/health_economic_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_states = pd.read_csv("C:/Users/drewc/GitHub/Portfolio/Toolbox/_data/toolbox_states_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

## State names MSPB
df_mspb =  df_mspb.rename(columns = {"Score": "MSPB", "State": "Abbreviation"}) # Rename column
df_join = pd.merge(df_mspb, df_states, on = "Abbreviation", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_mspb = df_join
df_mspb.info()

## State names ACS
df_acs =  df_acs.rename(columns = {"State": "Lower"}) # Rename column
df_join = pd.merge(df_acs, df_states, on = "Lower", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_acs = df_join
df_acs.info()

## Join ACS and MSPB
df_join = pd.merge(df_acs, df_mspb, on = "Abbreviation", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_acs_mspb = df_rename
df_acs_mspb.info()

## Export to CSV
df_acs_mspb.to_csv(r"_data/health_acs_mspb_stage.csv") # Clean in excel and select variable

### Random Forest

## Random Forest package
from sklearn.ensemble import RandomForestClassifier as rfc # Random Forest classification component

## Rename
df_rf = pd.read_csv("_data/health_acs_mspb_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

## Setup Predictors and Forest
features = df_rf.columns.drop(["State", "MSPB"]) # Drop outcome variable and Geo to isolate all predictor variable names as features
X = df_rf[features] # Save features columns as predictor data frame
Y = pd.factorize(df_rf["MSPB"])[0] # Isolate Outcome variable and factorize as numbers
forest = rfc(n_estimators = 500, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 

## Run Forest 
forest.fit(X, Y) # This will take time

## Output importances
gini = forest.feature_importances_ # Output importances of features
l_gini = list(zip(X, gini)) # Create list of variables alongside importance scores 
df_gini = pd.DataFrame(l_gini, columns = ["Variables", "Gini"]) # Create data frame of importances with variables and gini column names
df_gini = df_gini.sort_values(by = ["Gini"], ascending = False) # Sort data frame by gini value in desceding order

## Drop
df_reg = df_acs_mspb.filter(["MSPB", "Estimate!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population!!No health insurance coverage"]) # Keep only selected columns
df_reg =  df_reg.rename(columns = {"Estimate!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population!!No health insurance coverage": "Independent"}) # Rename column
df_reg = df_reg.apply(pd.to_numeric, errors = "coerce") # Convert all columns to numeric
df_reg = df_reg.dropna() # Drop all rows with NA values, 0 = rows, 1 = columns
df_reg.info()

## Linear Regression
from scipy import stats as st
x = np.array(df_reg["Independent"]).reshape((-1, 1))
y = np.array(df_reg["MSPB"]).reshape((-1, 1))
spearman = st.spearmanr(x, y)
print("Rsq =", (spearman[0]), "P-Value =", (spearman[1])) # swap spearman for pearson

## Control for Case-Mix Index
df_mix = pd.read_csv("_data/health_casemix_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_mix = df_mix.filter(["NPI", "CaseMix"]) # Keep only selected columns

df_mspb1 = df_cms.filter(["State", "NPI"]) # Keep only selected columns
df_mspb1.info()

df_npi = pd.merge(df_mix, df_mspb1, on = "NPI", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_npi = df_npi.groupby(["State"], as_index = False).mean() # Group data By Columns and Sum
df_npi = df_npi.drop(columns = ["NPI"]) # Drop Unwanted Columns
df_npi.info()

## Export to CSV
df_npi.to_csv(r"_data/health_casemix_state_stage.csv") # Clean in excel and select variable



df_con = pd.merge(df_npi, df_rf, on = "State", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_con =  df_con.rename(columns = {"Estimate!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population!!Civilian noninstitutionalized population under 19 years!!No health insurance coverage": "Predictor"}) # Rename column
df_con = df_con.filter(["Case_Mix_Index", "MSPB", "State", "Predictor"]) # Keep only selected columns
df_con.info()
df_con.head()

## Linear Regression
import statsmodels.api as sm
x = df_con.filter(["Case_Mix_Index", "Predictor"]) # Save predictor variables as x
y = df_con.filter(["MSPB"]) # Save outcome variable as y
model = sm.OLS(y, x).fit() # This may but most likely wont take time
result = model.summary() # Create Summary of final model
print(result) # Print result to verify

## 2nd Try
df_con = pd.merge(df_npi, df_rf, on = "State", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_con =  df_con.rename(columns = {"Estimate!!HEALTH INSURANCE COVERAGE!!Civilian noninstitutionalized population!!Civilian noninstitutionalized population under 19 years!!No health insurance coverage": "Percent_Uninsured"}) # Rename column
df_con = df_con.filter(["Case_Mix_Index", "MSPB", "State", "Percent_Uninsured"]) # Keep only selected columns

df_kff = pd.read_csv("_data/health_kff_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_con2 = df_con = pd.merge(df_con, df_kff, on = "State", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_con2.info()

df_con2.to_csv(r"_data/health_check_stage.csv") # Clean in excel and select variable

## Linear Regression
import statsmodels.api as sm
x = df_con2.filter(["Mean_Physician_Income"]) # Save predictor variables as x
y = df_con2.filter(["MSPB"]) # Save outcome variable as y
model = sm.OLS(y, x).fit() # This may but most likely wont take time
result = model.summary() # Create Summary of final model
print(result) # Print result to verify

## Linear Regression
import statsmodels.api as sm
x = df_con2.filter(["Percent_For_Profit"]) # Save predictor variables as x
y = df_con2.filter(["MSPB"]) # Save outcome variable as y
model = sm.OLS(y, x).fit() # This may but most likely wont take time
result = model.summary() # Create Summary of final model
print(result) # Print result to verify

## Linear Regression
import statsmodels.api as sm
x = df_con2.filter(["Case_Mix_Index", "Percent_For_Profit"]) # Save predictor variables as x
y = df_con2.filter(["MSPB"]) # Save outcome variable as y
model = sm.OLS(y, x).fit() # This may but most likely wont take time
result = model.summary() # Create Summary of final model
print(result) # Print result to verify

## Linear Regression
import statsmodels.api as sm
x = df_con2.filter(["Percent_For_Profit", "Mean_Physician_Income"]) # Save predictor variables as x
y = df_con2.filter(["MSPB"]) # Save outcome variable as y
model = sm.OLS(y, x).fit() # This may but most likely wont take time
result = model.summary() # Create Summary of final model
print(result) # Print result to verify

## Linear Regression
import statsmodels.api as sm
x = df_con2.filter(["Case_Mix_Index", "Mean_Physician_Income", "Percent_For_Profit"]) # Save predictor variables as x
y = df_con2.filter(["MSPB"]) # Save outcome variable as y
model = sm.OLS(y, x).fit() # This may but most likely wont take time
result = model.summary() # Create Summary of final model
print(result) # Print result to verify

## Linear Regression
import statsmodels.api as sm
x = df_con2.filter(["Case_Mix_Index", "Mean_Physician_Income", "Percent_For_Profit", "Percent_Uninsured"]) # Save predictor variables as x
y = df_con2.filter(["MSPB"]) # Save outcome variable as y
model = sm.OLS(y, x).fit() # This may but most likely wont take time
result = model.summary() # Create Summary of final model
print(result) # Print result to verify

## Linear Regression
import statsmodels.api as sm
x = df_con2.filter(["Mean_Physician_Income", "Percent_For_Profit", "Percent_Uninsured"]) # Save predictor variables as x
y = df_con2.filter(["MSPB"]) # Save outcome variable as y
model = sm.OLS(y, x).fit() # This may but most likely wont take time
result = model.summary() # Create Summary of final model
print(result) # Print result to verify