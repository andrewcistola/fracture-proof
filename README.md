# FractureProof
An Artificial Intelligence process for feature selection for multiple geographic layers wiothout the need of human selection for the purpose of informative policy analysis.

## Feature Selection
For many health outcomes, there are many possible causes with large quantites of observed variables. In order to find these possible but unknown causes, an analysis of structured datasets containling large numbers of variables (N > 1000), multiple dimensions, different variable types, and chunks of missing data. This process is designed to process, identify, and evaluate imprtant features in a small actionable list informative to policy analysis and decision making. These features can also be processed from multiple geographic layers for informative GIS analyses. 

## Open Source Models
The FracureProof process uses open source machine learning algorithms to select important features with these existing constraints. Modeling tehcniques are utilized from common libraries such as scikit-learn, keras, statsmodels, and PySal. Data processing steps use familair libraires including numpy, scipy, pandas, and geopandas. 

## Familair Statististical Tests
While many of the FractureProof modeling techniques involve machine learning algorithms or artificial neural networks, the final result of selected features are evaluated using multiple regression modeling and reciever operator curves. This allows for researchers not familair with these tools to comfortably evalaute trhe signficance of the final selection with fmaialir metrics.

## Options in Version 2.1
The FractureProof repository contains four different products that build on each other or can be used independently;
1. FractureProof - Feature processing and reduction for a single geographic layer. Result: a set of 5-25 features selected from possible catidates. 
2. Mr Fracture Proof - Selected features are placed alongside confounders for an inforamtive regression model. Result: R-squared, F-statistic, and t-statistics for a single layer.
3. Mr Fracture Proofs Woodcarvings - Features from a 2nd geographic layer are identified and placed into an inforamtive regression model. Result: R-squared, F-statistic, and t-statistics for both layers. 
4. Mr Fracture Proofs Contemplative Woodcarvings - Artifical neural networks are used to evaluate predictive ability of selected features comapred to all possible catidates. Result: C-statistics for feature groups. 

### FractureProof
Scrips labeled `fp_` use different open source modeling tehcniquies to reduce possible features significant for a given quantitative target. Each of these processes are available from widely utilized open source Python module Scikit Learn. Please consider citing, donating, or contirbuting to the project. https://github.com/scikit-learn/scikit-learn

#### Principal Component Analysis (PCA)
Using linear transformations on a normalized covariace matrix, PCA creates combinations of features in a regression model into principal components that explain a proportion of the variance observed in the dataset. The coefficients of the principal component models (eigenvectors) that explain a significant amount of variance can be compared and variables that do not explain significant variance can be dropped.  

#### Random Forests (RF)
By aggregating decision trees from a bootstrapped sample of features, random forests can measure the importance of a given feature in predicting the outcome of interest. By calculating the change in prediction capability when the feature is removed, the importance value (Gini Impurity) of a feature can be compared to others. Features that are not important compared to the others can be dropped. 

#### Recursive Feature Elimination
Using regression methods, RFE creates a predictive model and removes weakest features by comparing prediction scores. RFE can cross-validate to remove cosnistently weaker features until a ideal feature set is defined. A minimum or predefined set of features can be used. 

#### Feature Processing
Before features are input into the above models, features are selected if they have over 75% non-NA values. The remaining NA values are imputed with the median values for each feature. Users can also eliminate observations that do not fit a certain critera (ex. population under 100). The resulting values are then standard scaled. All observations missing a target value are dropped and descriptive statistics are calculated.

#### Results Gathering
After each of the steps in the above processes, results are automatically exported to a corresponding textfile with a datestamp for easy reference. 

#### The 'FractureProof' Process
The FP process uses these methods above in the following sequence:

1. The raw data set is cleaned of observation labels, the outcome is defined, and a missing value methodology is applied to create a cohort for feature selection.  
2. The absolue value of the eigenevctors for each feature in the cohort from components that explain significant variance in a PCA model  is collected. 
3. Gini impurity measures for each feature in the cohort from a RF classification model are collected. 
4. The features with below average eigenvectors and gini impurity are dropped from the cohort. 
5. RFE with cross-validation is used to identify the final list of features in the cohort. 
6. Results are automatically exported to the corresponding textfile for easy reference. 

### Mr. Fracture Proof
Scrips labeled `mrfp_` add 2 extra steps to the `fp_` scrips for the purpose of creating an informative regression model. 

#### Multiple regression modeling (MR)
Using a multiple linear regression model, confounders are added and the selected features can be evaluated using R-squared, F-statistic and significance values. Features can also be compared for magnitude and direction. This model is created using the raw data before imputation and standard scaling. All observations with missing data are dropped. 

#### The 'Mr. Fracture Proof' Process
The FP process uses these methods above in the following sequence:

1. The Fracture proof steps are completed as described above
2. Variables are placed into multiple regression model with selected confounding variables
3. Results are automatically exported to the corresponding textfile for easy reference. 

### Mr. Fracture Proof's Woodcarvings
Scrips labeled `mrfspwc_` add 2 extra steps to the `mrfp_` scrips for the purpose of conducting geographic weighted regression and identifying features froma  second geographic layer. 

#### Geographic Weighted Regression (GWR)
Using GIS data for for observations from the first layer, regression models are caluclated from existing features with coefficients weighted based on location. Locations where the given features are highest are labeled for the purpose of identifying 2nd layer locations where sleected features have higher predictive weight. This process utilizes the PySal library (https://pysal.org)

#### Support Vector Machines
Using the labeles creted by the GWR, support vector mahcines are used to identify 2nd layer features with the highest weights for the given location. Since the 2nd layer will often have signficiantly fewer obserrvations and predicts a multi-level categorical target, SVMs were slected for their ability to handle these constraints better than other available models. The result is a set of 2nd layer features that independently contribute or interact with the 1st layer features. 

#### The 'Mr. Fracture Proofs Woodcarvings' Process
The FP process uses these methods above in the following sequence:

1. The Fracture proof steps are completed as described above
2. GWR identifies weighted coefficients for the selected features for each 1st layer observation
3. 1st layer observations are averaged by 2nd layer location boundaires. 
4. Each 2nd layer location is labeled with a categorical target based on the 1st layer feature with the highest coefficient.
5. SVM are used to identify the 2nd layer feature with the highest average cofficients for each category.
6. 2nd layer features are selected and placed into a second multiple regression model along side 1st layer features. 
7. Results are automatically exported to the corresponding textfile for easy reference. 

### Mr. Fracture Proof's Contemplative Woodcarvings
Scrips labeled `mrfsctpwc_` add 2 extra steps to the `mrfpswc_` scrips for the purpose of evaluating the predictive ability of the slected features using an artificial neural network. 

#### Multi-layered Perceptrons (MLP)
An artificical neural network consisting of 2 dense layers, and a binary activatrion layer is used to predict a binary outcome calculated based on the original quantitative target. Predictions are made with all possible features, 1st layer features only, 2nd layer features only, and the final list of 1st and 2nd layer features. 

#### Receiver Operator Curve (ROC)
In order to indetify whether the selected vfeatures provide real world practicality in improving prediction, ROCs are created and C-statistics are calcuklated to determine the amount opf true positives to false positives. This allows for easy comparison of whether the selected features are relevant to decision making. 

#### Feature Processing
The raw data for the 1st and 2nd layers are processed using the steps in FractureProcess. The table is then randomly split 50-50 into test and train tables for evalatuing target prediction from the MLPs. 

#### The 'Mr. Fracture Proofs Contemplative Woodcarvings' Process
The FP process uses these methods above in the following sequence:

1. The Fracture proof steps are completed as described above
2. Raw 1st and 2nd layer data is joined, processed, and test-train 50-50 split
3. MLPs are run with 50-500 epochs based on loss and accuracy measures during training for each of the feature sets. 
4. C-statistics are calculated from the ROCs for comparison of accuracy in identifying true positives.
5. Results are automatically exported to the corresponding textfile for easy reference. 

## Repository Structure
The repository uses the following file organization and naming convenstions.

### File Organization
`_code`: code files related to the project
<br>`_data`: staged data files related to the project
<br>`_fig`: graphs, images, and maps related to the project
<br>`_pubs`: presentations and manuscrips related to the project
<br>`_raw`: raw data downloads related to the project
<br>`_refs`: selected literature related to the project

### File Naming Structure:
`subrepo_suffix_version.ext`

#### Suffixes:
`dev_`: Development script for working in an IDE
<br>`book_`: Jupyter notebook 
<br>`stage_`: Data files that have been modified from raw source
<br>`2020-01-01`: Text scripts displaying results output from a script are marked with date stamp they were created
<br>`map_`: 2D geographic display
<br>`graph_`: 2D chart or graph representing numeric data

#### Versions:
`alpha_`: first version of the file
<br>`beta_` : second version of the file
<br>`gamma_` : third version of the file
<br>`delta_` : fourth version of the file
<br>etc...

### Disclaimer
While the author (Andrew Cistola) is a Florida DOH employee and a University of Florida PhD student, these are NOT official publications by the Florida DOH, the University of Florida, or any other agency. 
No information is included in this repository that is not available to any member of the public. 
All information in this repository is available for public review and dissemination but is not to be used for making medical decisions. 
All code and data inside this repository is available for open source use per the terms of the included license. 

### allocativ
This repository is part of the larger allocativ project dedicated to prodiving analytical tools that are 'open source for public health.' Learn more at https://allocativ.com. 