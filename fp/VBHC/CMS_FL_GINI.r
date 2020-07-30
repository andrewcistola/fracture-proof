

## Import Libraries and Data

## Open R Terminal
R # open R in VS Code (any terminal)

## Import Hadley Wickham Libraries
library(tidyverse) # All of the libraries above in one line of code

## Import Statistics Libraries
library(ineq) # Gini coefficient and Lorenz curve
library(pROC) # ROC tests with AUC output

## Import Data
setwd("C:/Users/drewc/GitHub/HNB") # Set wd to project repository
df_fl = read.csv("_data/flcharts_50_stage.csv") # Import dataset from _data folder

### Step 2: Prepare Data for Classificaiton

## Tidy data types and objects
df_fl = df_fl %>% mutate_if(is.factor, as.numeric) # Change character to numeric values
df_fl = na.omit(df_fl) # Omit rows with NA from Data Frame
df_fl = subset(df_fl, select = -c(X.2, X.1, X)) # Remove variables with high missing values

## Verify Data
glimpse(df_fl) # Rows, columns, variable types and 10 

### Step 3: Develop Inequality Measures for Data

## Sort by mortality total
df_causes = subset(df_fl, select = -c(Tract, Total.Population, Pop.per.10000, Other.Non.rankable.Cause.of.Death)) # Remove variables with high missing values
df_sum = colSums(df_causes)
df_sum = as.data.frame(df_sum)
df_sum = rownames_to_column(df_sum) ## Invert Rows and Columns
colnames(df_sum) <- c("Cause", "Total") ## Change Column Names
df_sum = arrange(df_sum, desc(Total)) # Descend by variable in data frame
head(df_sum, 10)

## Create new column as sum of others
df_fl$Vascular = df_fl$Heart.Diseases + df_fl$Cerebrovascular.Diseases +  df_fl$Essen.Hypertension...Hypertensive.Renal.Dis + df_fl$ Aortic.Aneurysm...Dissection + df_fl$Atherosclerosis  # Create new column based on conditions
df_fl$Metabolic = df_fl$Diabetes.Mellitus + df_fl$Nephritis..Nephrotic.Syndrome..Nephrosis # Create new column based on conditions
df_fl$Cancer = df_fl$Malignant.Neoplasm.or.Cancer

## Gini coefficient
gini_Metabolic = ineq(df_fl$Metabolic, type = "Gini")
gini_Vascular = ineq(df_fl$Vascular, type = "Gini")
gini_Cancer = ineq(df_fl$Cancer, type = "Gini")

## Sum
sum_Metabolic = sum(df_fl$Metabolic)
sum_Vascular = sum(df_fl$Vascular)
sum_Cancer = sum(df_fl$Cancer)

## Combine into Data Frame
df_topcause = data.frame("Cause" = c("Metabolic", "Vascular", "Cancer"), "Sum" = c(sum_Metabolic, sum_Vascular, sum_Cancer), "Gini" = c(gini_Metabolic, gini_Vascular, gini_Cancer))
print(df_topcause)

## Write Output to File
result = df_topcause # Save result df to variable
file = file("cedric/cedric_flcharts_gini_results.txt") # Open result file in subproject repository
open(file, "w") # Open file and "a" to append
write("Top Causes of Mortality",  file) # Insert title
write(" ", file) # Insert space below title
capture.output(result, file = file, append = TRUE) # write summary to file
write(" ", file) # Insert space below result
close(file) # Close file

### 

## Outcome
df_fl$Outcome = df_fl$Vascular

## Group by County for internal Gini
tib_gini = df_fl %>% group_by(County) %>% summarise(ineq(Outcome, type = "Gini")) # Group By Columns and Average
df_gini = as.data.frame(tib_gini) # Convert tiblle to Data Frame
colnames(df_gini) <- c("County", "Gini") # Change Column Names

tib_sum = df_fl %>% group_by(County) %>% summarise(sum(Outcome)) # Group By Columns and Average
df_sum = as.data.frame(tib_sum) # Convert tiblle to Data Frame
colnames(df_sum) <- c("County", "Sum") # Change Column Names

tib_pop = df_fl %>% group_by(County) %>% summarise(sum(Total.Population)) # Group By Columns and Average
df_pop = as.data.frame(tib_pop) # Convert tiblle to Data Frame
colnames(df_pop) <- c("County", "Population") # Change Column Names

df_join = inner_join(df_gini, df_pop, by = "County") ## Join By Columns
df_join = inner_join(df_join, df_sum, by = "County") ## Join By Columns
df_join$Rate = df_join$Sum/df_join$Population
df_top = arrange(df_join, desc(Gini)) # Descend by variable in data frame
head(df_top, 10) 
  
## Spearman Rank
corr_gini = cor.test(x = df_join$Gini, y = df_join$Rate, method = "pearson") # Pearson's Rank for Q->Q
corr_gini

## Write Output to File
result2 = head(df_join, 10)
result3 = corr_gini
file = file("cedric/cedric_flcharts_gini_results.txt") # Open result file in subproject repository
open(file, "a") # Open file and "a" to append
write("Vascular",  file) # Insert title
write(" ", file) # Insert space below title
capture.output(result2, file = file, append = TRUE) # write summary to file
write(" ", file) # Insert space below result
write("Pearson Rank",  file) # Insert title
write(" ", file) # Insert space below title
capture.output(result3, file = file, append = TRUE) # write summary to file
write(" ", file) # Insert space below result
close(file) # Close file

###

## Outcome
df_fl$Outcome = df_fl$Cancer

## Group by County for internal Gini
tib_gini = df_fl %>% group_by(County) %>% summarise(ineq(Outcome, type = "Gini")) # Group By Columns and Average
df_gini = as.data.frame(tib_gini) # Convert tiblle to Data Frame
colnames(df_gini) <- c("County", "Gini") # Change Column Names

tib_sum = df_fl %>% group_by(County) %>% summarise(sum(Outcome)) # Group By Columns and Average
df_sum = as.data.frame(tib_sum) # Convert tiblle to Data Frame
colnames(df_sum) <- c("County", "Sum") # Change Column Names

tib_pop = df_fl %>% group_by(County) %>% summarise(sum(Total.Population)) # Group By Columns and Average
df_pop = as.data.frame(tib_pop) # Convert tiblle to Data Frame
colnames(df_pop) <- c("County", "Population") # Change Column Names

df_join = inner_join(df_gini, df_pop, by = "County") ## Join By Columns
df_join = inner_join(df_join, df_sum, by = "County") ## Join By Columns
df_join$Rate = df_join$Sum/df_join$Population
df_top = arrange(df_join, desc(Gini)) # Descend by variable in data frame
head(df_top, 10) 
  
## Spearman Rank
corr_gini = cor.test(x = df_join$Gini, y = df_join$Rate, method = "pearson") # Pearson's Rank for Q->Q
corr_gini

## Write Output to File
result2 = head(df_join, 10)
result3 = corr_gini
file = file("cedric/cedric_flcharts_gini_results.txt") # Open result file in subproject repository
open(file, "a") # Open file and "a" to append
write("Cancer",  file) # Insert title
write(" ", file) # Insert space below title
capture.output(result2, file = file, append = TRUE) # write summary to file
write(" ", file) # Insert space below result
write("Pearson Rank",  file) # Insert title
write(" ", file) # Insert space below title
capture.output(result3, file = file, append = TRUE) # write summary to file
write(" ", file) # Insert space below result
close(file) # Close file

###

## Outcome
df_fl$Outcome = df_fl$Nephritis..Nephrotic.Syndrome..Nephrosis

## Group by County for internal Gini
tib_gini = df_fl %>% group_by(County) %>% summarise(ineq(Outcome, type = "Gini")) # Group By Columns and Average
df_gini = as.data.frame(tib_gini) # Convert tiblle to Data Frame
colnames(df_gini) <- c("County", "Gini") # Change Column Names

tib_sum = df_fl %>% group_by(County) %>% summarise(sum(Outcome)) # Group By Columns and Average
df_sum = as.data.frame(tib_sum) # Convert tiblle to Data Frame
colnames(df_sum) <- c("County", "Sum") # Change Column Names

tib_pop = df_fl %>% group_by(County) %>% summarise(sum(Total.Population)) # Group By Columns and Average
df_pop = as.data.frame(tib_pop) # Convert tiblle to Data Frame
colnames(df_pop) <- c("County", "Population") # Change Column Names

df_join = inner_join(df_gini, df_pop, by = "County") ## Join By Columns
df_join = inner_join(df_join, df_sum, by = "County") ## Join By Columns
df_join$Rate = df_join$Sum/df_join$Population
df_top = arrange(df_join, desc(Gini)) # Descend by variable in data frame
head(df_top, 10) 
  
## Spearman Rank
corr_gini = cor.test(x = df_join$Gini, y = df_join$Rate, method = "pearson") # Pearson's Rank for Q->Q
corr_gini

## Write Output to File
result2 = head(df_join, 10)
result3 = corr_gini
file = file("cedric/cedric_flcharts_gini_results.txt") # Open result file in subproject repository
open(file, "a") # Open file and "a" to append
write("Nephritis..Nephrotic.Syndrome..Nephrosis",  file) # Insert title
write(" ", file) # Insert space below title
capture.output(result2, file = file, append = TRUE) # write summary to file
write(" ", file) # Insert space below result
write("Pearson Rank",  file) # Insert title
write(" ", file) # Insert space below title
capture.output(result3, file = file, append = TRUE) # write summary to file
write(" ", file) # Insert space below result
close(file) # Close file