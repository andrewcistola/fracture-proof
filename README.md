# FractureProof
An open source artificial intelligence process for precision public health research.

## About this Repository
FractureProof uses a variety of machine learning algorithms to identify an informative subset social determinants of health from multi-layered high dimensional public use datasets. FractureProof is built with commonly used Python Libraries included in the Anaconda Distribution and deployed as open-source package under the MIT license on GitHub. 
 
### Precion Public Health Research
Improving public health outcomes will require comprehensive population health management (PHM) that considers marked disparities among minority groups, seeks to address upstream factors related to socio-economic status, and includes updates to institutional policies, particularly at the state level. PHM efforts will need to be driven by location-based analyses that incorporate social stratification and community infrastructure relevant to public health. While studies have identified that neighborhood level disadvantage is a significant predictor of clinical outcomes the full extent of ecological health impacts needs further investigation. Interoperability issues related to health care information technology present significant hurdles for researching the effect of social determinants on public health outcomes.<br>
<br>
Precision Public Health involves the application of geographic data science techniques for small-area analysis to create targeted public health interventions. As a new area of health research, PPH strives to incorporate innovative tools in machine learning, artificial intelligence, and geographic information systems in a manner analogous to analyses of genomic data in Precision Medicine. Recent studies using some of these methods with public data have produced innovative analyses on the drivers of health outcomes and begun to assist in population health management efforts, but more research on how to utilize these methods are needed. 

### Multi-layered Data with High Dimensionality
When ecological data is collected, methods for identifying an actionable set (5-15) of possible factors have not been well established. Many common methods using probability based statistics do not have the ability to handle data from multiple geographies (Zip Code and County) with high dimensionality (2000+ variables) without significant limitations. Many new approaches in Artificial Intelligence that utilize “black-box” algorithms can predict outcomes very effectively, but do not provide interpretable results for understanding possible causal pathways. Similarly, many advanced algorithms suffer from “overfitting” or “noise” and may not provide informative results when deployed without proper processing. 
<br>
FractureProof uses various algorithms for specific purposes in identifying important predictors that account for high variation while accounting for geography and eliminating redundant measures. A final list of selected features is evaluated using multiple regression modeling and receiver operator curves. This allows for researchers to search large datasets for possible causes, avoid type 1 errors familiar, and comfortably evaluate the significance of the results. FractureProof automates this process so that exploratory analyses can be conducted without prior hypotheses or feature engineering. 

### Reproducibility and Validation
PPH studies will need to be validated by showing that public health efforts can be improved by results. This will require designing studies that can be replicated in various contexts and contain information relevant to practitioners. Using Artificial Intelligence in PPH research can be difficult for many entities that lack the computational resources and technical expertise in order to translate information effectively.<br>
<br>
The FracureProof process uses open source algorithms that have been widely deployed in many fields and have significant presence in scientific literature. Algortihms are taken from commonly used Python libraries including: scikit-learn, keras, statsmodels, PySal, numpy, scipy, pandas, and geopandas. Each of these libraries are available through the Anaconda Distribution, which provides users with the Python programming language, graphical user interface, easy to use software, and development environments without cost. FractureProof is deployed through GitHub under the MIT license and available as a software package for download to local devices. FractureProof does not require advanced computational resources and can be used on desktop devices common in enterprise settings. 

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
