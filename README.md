# FractureProof
An Artificial Intelligence process for identifying significant predictors of public health outcomes among among multiple geographic layers in large scale population datasets.

## About this Repository
For many health outcomes, there are many possible causes with large quantites of observed variables. In order to find these possible but unknown causes, an analysis of structured datasets containling large numbers of variables (N > 1000), multiple dimensions, different variable types, and chunks of missing data. FractureProof uses different open source processes to reduce features to help identify possible causes and opportunities for interventions. This repository contains code files and case studies for anybody to use in their analyses. 

### Multi-Dimensional Feature Selection
This process is designed to process, identify, and evaluate imprtant features in a small actionable list informative to policy analysis and decision making. These features can also be processed from multiple geographic layers for informative GIS analyses. 

### Open Source Models
The FracureProof process uses open source machine learning algorithms to select important features with these existing constraints. Modeling tehcniques are utilized from common libraries such as scikit-learn, keras, statsmodels, and PySal. Data processing steps use familair libraires including numpy, scipy, pandas, and geopandas. 

### Familair Statististical Tests
While many of the FractureProof modeling techniques involve machine learning algorithms or artificial neural networks, the final result of selected features are evaluated using multiple regression modeling and reciever operator curves. This allows for researchers not familair with these tools to comfortably evalaute trhe signficance of the final selection with fmaialir metrics.

## Options in Version 2.1
The FractureProof repository contains four different products that build on each other or can be used independently;<br>
<br>
**FractureProof** - Feature processing and reduction for a single geographic layer. Result: a set of 5-25 features selected from possible catidates.<br>
<br>
**Mr Fracture Proof** - Selected features are placed alongside confounders for an inforamtive regression model. Result: R-squared, F-statistic, and t-statistics for a single layer.<br>
<br>
**Mr Fracture Proofs Woodcarvings** - Features from a 2nd geographic layer are identified and placed into an inforamtive regression model. Result: R-squared, F-statistic, and t-statistics for both layers.<br> 
<br>
**Mr Fracture Proofs Contemplative Woodcarvings** - Artifical neural networks are used to evaluate predictive ability of selected features comapred to all possible catidates. Result: C-statistics for feature groups. 

## Repository Structure
The repository uses the following file organization and naming convenstions. This applies to all files in v2.1 and forward.

### File Naming Structure:
`version/prefix_suffix.ext`

### Subdirectories
`v#.#` All code files deployed for that specific release
<br>`_data` staged data files related to the project
<br>`_fig` graphs, images, and maps related to the project

### Prefixes:
`fp_` FractureProof
<br>`mrfp_` Mr Fracture Proof
<br>`mrfpswc_` Mr Fracture Proofs Woodcarvings
<br>`mrfpsctwc_` Mr Fracture Proofs Contemplative Woodcarvings<br>
`fldm2` Diabetes in Florida case study files

### Suffixes:
`_code` Development code script for working in an IDE
<br>`_book` Jupyter notebook 
<br>`_stage` Data files that have been modified from raw source
<br>`_2020-01-01` Text scripts displaying results output from a script are marked with date stamp they were created
<br>`_map` 2D geographic display
<br>`_graph` 2D chart or graph representing numeric data

### PEP-8
Whenever possible code scripts follow PEP-8 standards. 

## Disclaimer
While the author (Andrew Cistola) is a Florida DOH employee and a University of Florida PhD student, these are NOT official publications by the Florida DOH, the University of Florida, or any other agency. 
No information is included in this repository that is not available to any member of the public. 
All information in this repository is available for public review and dissemination but is not to be used for making medical decisions. 
All code and data inside this repository is available for open source use per the terms of the included license. 

### allocativ
This repository is part of the larger allocativ project dedicated to prodiving analytical tools that are 'open source for public health.' Learn more at https://allocativ.com. 
