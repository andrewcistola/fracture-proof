# FractureProof
An Artificial Intelligence process for identifying significant predictors of public health outcomes among among multiple geographic layers in large scale population datasets.

## Case Study: Diabetes Mortality in Florida
Version 2.1 includes a case stuidy using the Mr Fracture proof's Contemplative Woodcarvings option to conduct research on social and infastructural factors related to Diabetes mortality in Florida.

### Finding Equity: Utilizing Artificial Intelligence to Identify Social and Infrastructural Predictors of Diabetes Mortality in Florida
Andrew S. Cistola, MPH<br>
Department of Health Services Research, Management, and Policy<br>
University of Florida
<br><br>
**Introduction**: DM outcomes represent one of the largest avoidable cost burdens with opportunity for improvement in the U.S. health care system. Improving health equity in the context of DM will require targeted community improvements, infrastructure investments, and policy interventions that are designed to maximize the impact of resource allocation through the use of available data and computational resources.
<br><br>
**Methods**: By using an Artificial Intelligence approach to evaluate over 2000 socio-economic and infrastructural predictors of DM mortality, this study used a specific series of modeling techniques to identify significant predictors without human selection and compare their predictive ability with all possible factors when passed through artificial neural networks.
<br><br>
**Results**: The final regression model using zip code and county level predictors had an R2 of 0.863. Significant predictors included: Population % White, Population % Householders, Population % Spanish spoken at home, Population % Divorced males, Population % With public health insurance coverage, Population % Employed with private health insurance coverage, Manufacturing-Dependent Designation, Low Education Designation, Population % Medicare Part A & B Female Beneficiaries, Number of Short Term General Hospitals with 50-99 Beds. Using a multi-layered perceptron to predict zip codes at risk the C-statistic for all 2000+ predictors was 0.7938 while the 13 selected predictors was 0.8232.
<br><br>
**Discussion**: This indicates that these factors are highly relevant for DM mortality in Florida. This process was completed without the need of human variable selection and indicates how AI can be used for informative precision public health analyses for targeted population health management efforts.<br>
<br>
![](_fig/fldm2_map.png)<br>

### Repository contents:
`fldm2_manuscript.doc` Manuscript for Diabates Mortality in Florida case study<br>
`fldm2_2020-11-06.txt` Results file from Diabates Mortality in Florida case study<br>
`fldm2_book.ipynb` Jupyter notebook for Diabetes Mortality in Florida case study<br>
`fldm2_code.py` Development script for Diabates Mortality in Florida case study<br>
`_data` Subrepository for data files
`_fig` Subrepository for images

### Disclaimer
While the author (Andrew Cistola) is a Florida DOH employee and a University of Florida PhD student, these are NOT official publications by the Florida DOH, the University of Florida, or any other agency. 
No information is included in this repository that is not available to any member of the public. 
All information in this repository is available for public review and dissemination but is not to be used for making medical decisions. 
All code and data inside this repository is available for open source use per the terms of the included license. 

### allocativ
This repository is part of the larger allocativ project dedicated to prodiving analytical tools that are 'open source for public health.' Learn more at https://allocativ.com. 
