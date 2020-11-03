# FractureProof
FractureProof is consolidated multi-step process for high-dimensional dataset reduction, feature selection, and outcome prediction. The process utilizes using principal component analysis, random forests, recursive feature selection and regression modeling to identify signficiant relationships within large population based datasets. 

### Real World Feature Selection
For many outcomes that important for population focused research, there are many possible causes manifesting in large quantites of observed predictors. In order to find these possible but unknown causes, FractureProof provides a stable process for many real world issues in data collection (large numbers of variables (N > 1000), multiple dimensions, different variable types, and chunks of missing data). Also, this analysis is only beneficial if it can be easily translated into translatable practice. 

### Four open source algorithms from scikit-learn
FractureProof uses three different open source processes to reduce features for different purposes. Principal component analysis, random forests, recursive feature elimination, and everyday regression modeling. Each of these processes are available from the widely utilized open source Python module Scikit Learn. Please consider citing, donating, or contirbuting to their project. https://github.com/scikit-learn/scikit-learn

### Integration with neural networks with keras
FractureProof is deisgned to integrate with aritfical neural networks for supervised learning in keras. The feature reduction method in FractureProof identifies a small subset of 5-20 variables with high predictive value that can easily be passed into a 3D neural network for training and tesing on target outcomes.

## The FractureProof Process
FractureProof uses the following algorithms:

### Principal Component Analysis (PCA)
Using linear transformations on a normalized covariace matrix, PCA creates combinations of features in a regression model into principal components that explain a proportion of the variance observed in the dataset. The coefficients of the principal component models (eigenvectors) that explain a significant amount of variance can be compared and variables that do not explain significant variance can be dropped.  

### Random Forests (RF)
By aggregating decision trees from a bootstrapped sample of features, random forests can measure the importance of a given feature in predicting the outcome of interest. By calculating the change in prediction capability when the feature is removed, the importance value (Gini Impurity) of a feature can be compared to others. Features that are not important compared to the others can be dropped. 

### Recursive Feature Elimination (RFE)
Using regression methods, RFE creates a predictive model and removes weakest features by comparing prediction scores. RFE can cross-validate to remove cosnistently weaker features until a ideal feature set is defined. A minimum or predefined set of features can be used. 

### Regression Modeling (R)
For those that have taken a statistics course, this shoud be familiar. Features are used to create an equation with negative or positive coefficients of various values corresponding to their value in the full model. Adjusted R squared can be used to determine the validity of the final featuires and significannce testiing can be conducted based on p-values and confidence intervals. The final product of the process is understandable by old school and contemporary data scientists. 

The FP process uses these methods above in the following sequence:

1. The raw data set is cleaned of observation labels, the outcome is defined, and a missing value methodology is applied to create a cohort for feature selection.  
2. The absolue value of the eigenevctors for each feature in the cohort from components that explain significant variance in a PCA model  is collected.
3. Gini impurity measures for each feature in the cohort from a RF classification model are collected.
4. The features with below average eigenvectors and gini impurity are dropped from the cohort.
5. RFE with cross-validation is used to identify the final list of features in the cohort.
6. The final list of variables is used to create a regresion model so coefficients can be compared for magnitue and direction.
7. Once the final set of features are collected, a validation method can be used to evaluate the ability for these variables to predict the outcome of interest. Artifical neural networks work great for this.

This code is open source, use it to do research and improve your data analysis. Please cite and mention FractureProof whenever you use this software. For specifics, refer to the open source license associated with this repository.