# FractureProof
# DOH Florida COVID19 by County

## Step 1: Import Libraries and Import Dataset

### Import Python Libraries
import os # Operating system navigation

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes

### Set working directory to subproject folder
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

### Import Datasets
df_case = pd.read_csv("hnb/DOH/FL/COVID19_FIPS/21July20/Florida_COVID19_Case_Line_Data.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository
df_pop = pd.read_csv("hnb/DOH/FL/POP_FIPS_2019/POP_FIPS_2019_stage.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository
df_key = pd.read_csv("hnb/FIPS/FIPS_ZCTA_key.csv", encoding = "ISO-8859-1", low_memory= False) # Import dataset with outcome and ecological variable for each geographical id, all datasets in _data folder in repository

### Verify
df_case.info() # Get class, memory, and column info: names, data types, obs.
df_case.head() # Print first 5 observations

## Step 2: Prepare Case Line Data for Analysis

### Create Daily Count of New Cases
df_rename =  df_case.rename(columns = {"Case_": "Case"}) # Rename column
df_drop = df_rename.filter(["County", "Case", "EDvisit", "Hospitalized", "Died", "EventDate"]) # Keep only selected columns
df_drop['Case'] = np.where((df_drop["Case"] == "YES") | (df_drop["Case"] == "Yes"), 1, 0) # Create New Column Based on Conditions
df_drop['EDvisit'] = np.where((df_drop["EDvisit"] == "YES") | (df_drop["EDvisit"] == "Yes"), 1, 0) # Create New Column Based on Conditions
df_drop['Hospitalized'] = np.where((df_drop["Hospitalized"] == "YES") | (df_drop["Hospitalized"] == "Yes"), 1, 0) # Create New Column Based on Conditions
df_drop['Died'] = np.where((df_drop["Died"] == "YES") | (df_drop["Died"] == "Yes"), 1, 0) # Create New Column Based on Conditions
df_group = df_drop.groupby(["County"], as_index = False).sum() # Group data By Columns and Sum

### Create population adjusted rates
df_join = pd.merge(df_group, df_pop, on = "County", how = "inner") # Join by column and add counties without confirmed cases
df_join["Case_R1000"] = df_join["Case"] / df_join["Population"] * 1000 # Create new coluimn with math
df_join["EDvisit_R1000"] = df_join["EDvisit"] / df_join["Population"] * 1000 # Create new coluimn with math
df_join["Hospitalized_R1000"] = df_join["Hospitalized"] / df_join["Population"] * 1000 # Create new coluimn with math
df_join["Died_R1000"] = df_join["Died"] / df_join["Population"] * 1000 # Create new coluimn with math

### Join with ZCTA_FIPS key
df_rename =  df_join.rename(columns = {"County": "NAME"}) # Rename multiple columns in place
df_sub = df_key[df_key.ST == "FL"] # Susbet numeric column by condition
df_filter = df_sub.filter(["NAME", "FIPS", "ZCTA"]) # Keep only selected columns
df_join = pd.merge(df_sub, df_rename, on = "NAME", how = "left") # Join by column and add counties without confirmed cases

### Filter columns
df_c19_FIPS = df_join.filter(["FIPS", "ZCTA", "Case_R1000", "EDvisit_R1000", "Hospitalized_R1000", "Died_R1000"]) # Keep only selected columns

### Verify
df_c19_FIPS.info() # Get class, memory, and column info: names, data types, obs.
df_c19_FIPS.head() # Print first 5 observations



