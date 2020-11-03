# FractureProof
A three step process for feature selection using principal component analysis, random forests, and recursive feature selection for public health informatics. 

## Version 2.1
Updates from v1.6 include:
Standardized scripts
'Mr. Fracture K Proof' option for prediction
'Mr. Fracture K Proof's Wager' option for geographically weighted regression

## FractureProof
Scrips labeled `fp_` use three different open source processes to reduce features for different purposes. Each of these processes are available from widely utilized open source Python module Scikit Learn. Please consider citing, donating, or contirbuting to the project. https://github.com/scikit-learn/scikit-learn

### Principal Component Analysis (PCA)
Using linear transformations on a normalized covariace matrix, PCA creates combinations of features in a regression model into principal components that explain a proportion of the variance observed in the dataset. The coefficients of the principal component models (eigenvectors) that explain a significant amount of variance can be compared and variables that do not explain significant variance can be dropped.  

### Random Forests (RF)
By aggregating decision trees from a bootstrapped sample of features, random forests can measure the importance of a given feature in predicting the outcome of interest. By calculating the change in prediction capability when the feature is removed, the importance value (Gini Impurity) of a feature can be compared to others. Features that are not important compared to the others can be dropped. 

### Recursive Feature Elimination
Using regression methods, RFE creates a predictive model and removes weakest features by comparing prediction scores. RFE can cross-validate to remove cosnistently weaker features until a ideal feature set is defined. A minimum or predefined set of features can be used. 

### The 'FractureProof' Process
The FP process uses these methods above in the following sequence:

1. The raw data set is cleaned of observation labels, the outcome is defined, and a missing value methodology is applied to create a cohort for feature selection.  
2. The absolue value of the eigenevctors for each feature in the cohort from components that explain significant variance in a PCA model  is collected. 
3. Gini impurity measures for each feature in the cohort from a RF classification model are collected. 
4. The features with below average eigenvectors and gini impurity are dropped from the cohort. 
5. RFE with cross-validation is used to identify the final list of features in the cohort. 

## Mr. Fracture K Proof
Scrips labeled `mrfkp_` add 3 extra steps to the `fp_` scrips for the purpose of creating an informative regression model. 

### k-means clustering

### Multiple regression modeling

### The 'Mr. Fracture K Proof' Process
The FP process uses these methods above in the following sequence:

1. The Fracture proof steps are completed as described above
2. PCA is used to identify components of subset data
3. Components are input into k-means clustering to label variables intop groups
4. Random forest is used to identify variables of highest importance from each group
5. Variables are placed into multiple regression model with selected confounding variables

