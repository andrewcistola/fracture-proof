# FractureProof
A three step process for feature selection using principal component analysis, random forests, and recursive feature selection for public health informatics. 

## Real World Feature Selection
For many health outcomes, there are many possible causes with large quantites of observed variables. In order to find these possible but unknown causes, an analysis of structured datasets containling large numbers of variables (N > 1000), multiple dimensions, different variable types, and chunks of missing data. Also, this analysis is only beneficial if it can be easily translated into translatable practice. 

The FracureProof process uses a three step emthodology to select important features (variables/columns) with these existing constraints. The result is a list of features that can be input into a validation method like risk prediction or regression. Think of this process as an automated method for selecting variables to input into a stepwise regression model. 

## Three Open Source Steps
FractureProof uses three different open source processes to reduce features for different purposes. Principal component analysis, random forests, and recursive feature elimination. Each of these processes are available from widely utilized open source Python module Scikit Learn. Please consider citing, donating, or contirbuting to the project. https://github.com/scikit-learn/scikit-learn

### Principal Component Analysis (PCA)
Using linear transformations on a normalized covariace matrix, PCA creates combinations of features in a regression model into principal components that explain a proportion of the variance observed in the dataset. The coefficients of the principal component models (eigenvectors) that explain a significant amount of variance can be compared and variables that do not explain significant variance can be dropped.  

### Random Forests (RF)
By aggregating decision trees from a bootstrapped sample of features, random forests can measure the importance of a given feature in predicting the outcome of interest. By calculating the change in prediction capability when the feature is removed, the importance value (Gini Impurity) of a feature can be compared to others. Features that are not important compared to the others can be dropped. 

### Recursive Feature Elimination
Using regression methods, RFE creates a predictive model and removes weakest features by comparing prediction scores. RFE can cross-validate to remove cosnistently weaker features until a ideal feature set is defined. A minimum or predefined set of features can be used. 

## The FractureProof (FP) Process
The FP process uses these methods above in the following sequence:

1. The raw data set is cleaned of observation labels, the outcome is defined, and a missing value methodology is applied to create a cohort for feature selection.  
2. The absolue value of the eigenevctors for each feature in the cohort from components that explain significant variance in a PCA model  is collected. 
3. Gini impurity measures for each feature in the cohort from a RF classification model are collected. 
4. The features with below average eigenvectors and gini impurity are dropped from the cohort. 
5. RFE with cross-validation is used to identify the final list of features in the cohort. 

Once the final set of features are collected, a validation method can be used to evaluate the ability for these variables to predict the outcome of interest. The example in this repository uses logistic regression to make a predictive model with the raw data. 

## Improving Public Health
FractureProof is a process that uses open source tools and publically available data so any entity focused on the improvement of the public health can utilize the process. This project fits inside two different doamins within public health research: Precision Public Health and Public Health Informatics.   

### Precision Public Health
"M Khoury, the Director of Office of Public Health Genomics at the Centers for Diseases Control and Prevention (CDC), defined ‘precision’ in the context of public health as “improving the ability to prevent disease, promote health, and reduce health disparities in populations by: 
1) applying emerging methods and technologies for measuring disease, pathogens, exposures, behaviors, and susceptibility in populations; 2) developing policies and targeted implementation programs to improve health” (Prosperi M, 2019).

This process is designed to implement a methodlogy for easy analysis of public data to determine important elements associated with a health outcome. These outputs can tailored be provided to practicioners and researchers informative elements for the improvement of program or policy. 

### Public Health Informatics
"Public health informatics (PHI) has been described as the field that optimizes the use of information to improve individual health, health care, public health practice, biomedical and health services research, and health policy" (Edmunds M, 2014).
