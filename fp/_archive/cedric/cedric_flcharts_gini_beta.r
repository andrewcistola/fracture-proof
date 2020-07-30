#### Healthy Neighborhoods Project: Using Ecological Data to Improve Community Health
### Cedric subproject: Developing better ways to measure equity in health
## Florida Charts Census Tract Mortality Data: 2013-2017 Crude Mortality from Top 50 Causes
# The R project for Statistical Computing Script by DrewC!

#### Section A: Import Libraries and Datasets and Prepare Data

### Step 1: Import Libraries and Data

## Import Hadley Wickham Libraries
library(tidyverse) # All of the libraries above in one line of code
library(skimr) # Library used for easy summary of data

## Import Statistics Libraries
library(ineq) # Gini coefficient and Lorenz curve
library(pROC) # ROC tests with AUC output

## Import Data
setwd("C:/Users/drewc/GitHub/Healthy_Neighborhoods") # Set wd to project repository
df_fl = read.csv("_data/flcharts_50_stage.csv", fileEncoding = "UTF-8-BOM") # Import dataset from _data folder
df_causes = read.csv("_data/flcharts_aadr_causes_stage.csv", fileEncoding = "UTF-8-BOM") # Import dataset from _data folder
df_cms = read.csv("_data/cedric_cms_pr_value.csv", fileEncoding = "UTF-8-BOM") # Import dataset from _data folder

## Verify Data
head(df_fl) # Mini table with top 5 observations 
head(df_causes) # Mini table with top 5 observations 
head(df_cms) # Mini table with top 5 observations

### Step 2: Prepare Data for Analysis

## Tidy data types and objects
df_fl = df_fl %>% mutate_if(is.factor, as.character) # Change character to numeric values
df_causes = df_causes %>% mutate_if(is.factor, as.character) # Change character to numeric values
df_cms = df_cms %>% mutate_if(is.factor, as.character) # Change character to numeric values
df_fl = na.omit(df_fl) # Omit rows with NA from Data Frame

## Verify Data
glimpse(df_fl) # Rows, columns, variable types and 10 
glimpse(df_causes) # Rows, columns, variable types and 10 
glimpse(df_cms) # Rows, columns, variable types and 10 

## Write Output to File
text_1 = skim(df_fl)
file = file("cedric/cedric_flcharts_gini_results.txt") # Open result file in subproject repository
open(file, "w") # Open file and "a" to append
write("Cedric subproject: Developing better ways to measure equity in health",  file) # Insert title
write("Florida Charts Census Tract Mortality Data: 2013-2017 Crude Mortality from Top 50 Causes\n\n",  file) # Insert title
write("Data Summary\n",  file) # Insert title
capture.output(text_1, file = file, append = TRUE) # write summary to file
close(file) # Close file

### Step 3: Identify Top Mortality Measures by Inequality

## Create data frame of Gini by causes
df_features = subset(df_fl, select = -c(Tract, County)) # Remove Other Causes variable
df_gini = as.data.frame(apply(df_features, 2, ineq)) # Apply function to all comuns, 1 for rows and save to data frame
df_gini$rownames <- rownames(df_gini)
colnames(df_gini) <- c("Gini", "Causes") # Rename Columns to function and feature label
rownames(df_gini) <- c()

## Join with To 50 Causes by AADR for 2013-2017
df_join = inner_join(df_gini, df_causes, by = "Causes") ## Join By Columns

## Sort by top 10
df_rank1 = df_join[order(-df_join$AADR),] # Sort df by column
df_rank1 = head(df_rank1, 10)
df_rank2 = df_rank1[order(-df_rank1$Gini),] # Sort df by column

## Verify Data
head(df_rank1) # Mini table with top 5 observations 
head(df_rank2) # Mini table with top 5 observations 
head(df_join) # Mini table with top 5 observations 

## Write Output to File
text_1 = df_rank1
text_2 = df_rank2
file = file("cedric/cedric_flcharts_gini_results.txt") # Open result file in subproject repository
open(file, "a") # Open file and "a" to append
write("\n\nTop Causes of Mortality",  file) # Insert title
write("\nBy AADR",  file) # Insert title
capture.output(text_1, file = file, append = TRUE) # write summary to file
write("\nTop 10 Gini\n",  file) # Insert title
capture.output(text_2, file = file, append = TRUE) # write summary to file
close(file) # Close file

#### Section B: Get Outcome Specific Gini coefficient and MSPB at Hospitals by County

### Step 7: Compare Gini and For-Profit Hospitals

## Create health outcome variables
df_fl$Diabetes = df_fl$Diabetes.Mellitus + df_fl$Nephritis..Nephrotic.Syndrome..Nephrosis

## Group by County for internal Gini
tib_gini = df_fl %>% group_by(County) %>% summarise(ineq(Diabetes, type = "Gini")) # Group By Columns and Average
df_county = as.data.frame(tib_gini) # Convert tiblle to Data Frame
colnames(df_county) <- c("County", "Gini") # Change Column Names
df_county = df_county[order(-df_county$Gini),] # Sort df by column

## Join with County Value Measure 
df_join2 = inner_join(df_county, df_cms, by = "County", how = "left") ## Join By Columns
dim(df_join2)

## Spearman Rank
corr_diabetes = cor.test(x = df_join2$Gini, y = df_join2$MSPB, method = "spearman") # Pearson's Rank for Q->Q

## Write Output to File
text_1 = head(df_county, 10)
text_2 = corr_diabetes
file = file("cedric/cedric_flcharts_gini_results.txt") # Open result file in subproject repository
open(file, "a") # Open file and "a" to append
write("\n\nAssociation of Diabetes Inequity and MSPB",  file) # Insert title
write("\nTop 10 Counties by Inequality",  file) # Insert title
capture.output(text_1, file = file, append = TRUE) # write summary to file
write("\nFor Profit Hospitals",  file) # Insert title
capture.output(text_2, file = file, append = TRUE) # write summary to file
write("\n\n\nTHE END",  file) # Insert title
close(file) # Close file

print("THE END")
#### End Script