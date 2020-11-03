#### Healthy Neighborhoods Project: Using Ecological Data to Improve Community Health
### Cedric subproject: Developing better ways to measure equity in health using the Gini coefficient
## Florida Charts Census Tract Mortality Data:  Pyhton Computing Language Code Script by DrewC!

### Step 1: Import Libraries and Import Dataset

## Import Standard Libraries
import os # Inlcuded in every script DC!
import pandas as pd # Incldued in every code script for DC!
import numpy as np # Inclduded in every code script DC!
import scipy as sp  # Incldued in every code script for DC!

## Import Datasets
os.chdir("C:/Users/drewc/GitHub/Healthy_Neighborhoods") # Set wd to project repository
df_gen = pd.read_csv("_data/cms_hosp_gen_stage.csv") # Import dataset from _data folder
df_mspb = pd.read_csv("_data/cms_hosp_mspb_stage.csv") # Import dataset from _data folder
df_err = pd.read_csv("_data/cms_hosp_readmit_stage.csv") # Import dataset from _data folder

## Tidy Data Types, Missing Values, and Variable Names
df_mspb["MSPB"] = df_mspb["MSPB"].apply(pd.to_numeric, errors = "coerce") # Convert all columns to numeric
df_err["ERR"] = df_err["ERR"].apply(pd.to_numeric, errors = "coerce") # Convert all columns to numeric

## Verify
df_gen.info() # Get class, memory, and column info: names, data types, obs.
df_mspb.info() # Get class, memory, and column info: names, data types, obs.
df_err.info() # Get class, memory, and column info: names, data types, obs.

### Step 2: Join Datasets

## Convert ERR from long to wide
df_err = df_err.pivot_table(index = "FacilityID", columns = "Measure", values = "ERR") # Pivot from Long to Wide Format

## Join by Column
df_merge1 = pd.merge(df_gen, df_mspb, on = "FacilityID", how = "inner")
df_hosp = pd.merge(df_merge1, df_err, on = "FacilityID", how = "inner")

## Export to CSV
df_hosp.to_csv(r"_data/cms_hosp_value_join.csv") # Clean in excel and select variable

## Reimport CSV
df_hosp = pd.read_csv("_data/cms_hosp_value_join.csv") # Import dataset from _data folder

## Verify
df_hosp.info() # Get class, memory, and column info: names, data types, obs.

#### Subset by MSPB
df_fl = df_hosp[(df_hosp["State"].str.contains("FL", na = False))]

## Subset Non-Profit or For-Profit
df_fl = df_fl[(df_fl["Ownership"].str.contains("Voluntary", na = False))]
df_fl = df_fl[(df_fl["Ownership"].str.contains("Proprietary", na = False))]

## Sort by Value
df_value = df_fl.sort_values(by = ["MSPB"], ascending = False) # Sort Columns by Value
df_value = df_value.filter(["MSPB", "FacilityID", "Name", "County"]) # Keep only selected columns

## Verify
df_value.info()
df_value.head()

## Export to CSV
df_value.to_csv(r"_data/cedric_cms_pr_value.csv") # Clean in excel and select variable



