allocativ
FractureProof
ACS 2014-2018 Zip Code Percent Estimates for 50 States
Florida DOH 2014-2018 Zip Code 113 Causes of Death
COVID 19 Cases by Zip Code 20 July 2020

Full Dataset
(Zip Codes, Features)
(939, 492)
Dataset with features <25% NA
(939, 492)
NA ratio (median values imputed)
0.005028269132531581
Demographics
count     939.000000
mean       25.953590
std       262.778757
min         0.000000
25%         8.609123
50%        12.622176
75%        19.213310
max      8000.000000
Name: outcome, dtype: float64

Principal Components Analysis

Features with above average absolute value of Eigenvectors on components with above average Explained Variance Ratios 
(91, 1)
Top 10 Features by Eigenvector
Features         MaxEV
A00_B99_R1000   0.214759
Benign_R1000    0.230886
E00_E99_R1000   0.203345
G00_G99_R1000   0.137862
J00_J99_R1000   0.154437
K00_K99_R1000   0.138921
N00_N99_R1000   0.197805
Residual_R1000  0.149281
V01_Y89_R1000   0.174444
DP05_0034PE     0.125727

Random Forest

Features with above average Gini Impurity Values
(83, 2)
Top 10 Variables by Gini Impurity
          Features      Gini
425    DP03_0056PE  0.060142
330    DP02_0112PE  0.047756
119    DP04_0022PE  0.040895
56     DP05_0073PE  0.040613
0    A00_B99_R1000  0.040060
6    K00_K99_R1000  0.038087
284    DP02_0064PE  0.035451
80     DP05_0010PE  0.032869
137    DP04_0041PE  0.031688
128    DP04_0031PE  0.027711

Recursive Feature Elimination

Features Selected from RF and PCA
0      A00_B99_R1000
1      K00_K99_R1000
2        DP05_0010PE
3        DP04_0041PE
4        DP03_0042PE
5        DP02_0114PE
6     Residual_R1000
7      N00_N99_R1000
8      V01_Y89_R1000
9        DP05_0002PE
10       DP03_0004PE
11       DP03_0048PE
12       DP03_0003PE
13     E00_E99_R1000
14       DP02_0129PE
15       DP03_0047PE
16       DP03_0013PE
17       DP03_0024PE
18       DP04_0040PE
19       DP03_0012PE
20       DP02_0115PE
21       DP05_0027PE

Selected Features by Cross-Validation
    index        Features
0       1   K00_K99_R1000
1       2     DP05_0010PE
2       4     DP03_0042PE
3       5     DP02_0114PE
4       6  Residual_R1000
5       7   N00_N99_R1000
6       8   V01_Y89_R1000
7       9     DP05_0002PE
8      10     DP03_0004PE
9      11     DP03_0048PE
10     12     DP03_0003PE
11     14     DP02_0129PE
12     15     DP03_0047PE
13     17     DP03_0024PE
14     18     DP04_0040PE
15     19     DP03_0012PE
16     20     DP02_0115PE
17     21     DP05_0027PE


Regression Model

OLS Model on Original Data
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:                outcome   R-squared (uncentered):                   0.985
Model:                            OLS   Adj. R-squared (uncentered):              0.984
Method:                 Least Squares   F-statistic:                              3299.
Date:                Tue, 21 Jul 2020   Prob (F-statistic):                        0.00
Time:                        13:21:10   Log-Likelihood:                         -4604.5
No. Observations:                 939   AIC:                                      9245.
Df Residuals:                     921   BIC:                                      9332.
Df Model:                          18                                                  
Covariance Type:            nonrobust                                                  
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
K00_K99_R1000     33.2001      0.564     58.895      0.000      32.094      34.306
DP05_0010PE        1.9132      0.334      5.735      0.000       1.258       2.568
DP03_0042PE        0.7256      0.208      3.483      0.001       0.317       1.134
DP02_0114PE        0.2099      0.198      1.060      0.290      -0.179       0.599
Residual_R1000     4.4848      0.485      9.239      0.000       3.532       5.438
N00_N99_R1000     11.4898      1.970      5.831      0.000       7.623      15.357
V01_Y89_R1000     -3.2159      0.217    -14.810      0.000      -3.642      -2.790
DP05_0002PE       -1.9853      0.352     -5.645      0.000      -2.676      -1.295
DP03_0004PE        0.6200      0.555      1.117      0.264      -0.469       1.709
DP03_0048PE        1.8964      0.356      5.324      0.000       1.197       2.595
DP03_0003PE        0.8657      0.650      1.332      0.183      -0.410       2.142
DP02_0129PE       -2.1868      0.841     -2.601      0.009      -3.837      -0.537
DP03_0047PE        1.8207      0.321      5.675      0.000       1.191       2.450
DP03_0024PE        1.6166      0.218      7.428      0.000       1.190       2.044
DP04_0040PE       -0.5875      0.141     -4.176      0.000      -0.864      -0.311
DP03_0012PE       -0.5545      0.300     -1.851      0.064      -1.142       0.033
DP02_0115PE        1.2345      0.393      3.141      0.002       0.463       2.006
DP05_0027PE       -4.1916      0.324    -12.936      0.000      -4.828      -3.556
==============================================================================
Omnibus:                      829.831   Durbin-Watson:                   1.863
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           106870.447
Skew:                           3.417   Prob(JB):                         0.00
Kurtosis:                      54.815   Cond. No.                         278.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Regression Model

OLS Model on Original Data
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:                outcome   R-squared (uncentered):                   0.985
Model:                            OLS   Adj. R-squared (uncentered):              0.984
Method:                 Least Squares   F-statistic:                              3299.
Date:                Tue, 21 Jul 2020   Prob (F-statistic):                        0.00
Time:                        14:10:11   Log-Likelihood:                         -4604.5
No. Observations:                 939   AIC:                                      9245.
Df Residuals:                     921   BIC:                                      9332.
Df Model:                          18                                                  
Covariance Type:            nonrobust                                                  
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
K00_K99_R1000     33.2001      0.564     58.895      0.000      32.094      34.306
DP05_0010PE        1.9132      0.334      5.735      0.000       1.258       2.568
DP03_0042PE        0.7256      0.208      3.483      0.001       0.317       1.134
DP02_0114PE        0.2099      0.198      1.060      0.290      -0.179       0.599
Residual_R1000     4.4848      0.485      9.239      0.000       3.532       5.438
N00_N99_R1000     11.4898      1.970      5.831      0.000       7.623      15.357
V01_Y89_R1000     -3.2159      0.217    -14.810      0.000      -3.642      -2.790
DP05_0002PE       -1.9853      0.352     -5.645      0.000      -2.676      -1.295
DP03_0004PE        0.6200      0.555      1.117      0.264      -0.469       1.709
DP03_0048PE        1.8964      0.356      5.324      0.000       1.197       2.595
DP03_0003PE        0.8657      0.650      1.332      0.183      -0.410       2.142
DP02_0129PE       -2.1868      0.841     -2.601      0.009      -3.837      -0.537
DP03_0047PE        1.8207      0.321      5.675      0.000       1.191       2.450
DP03_0024PE        1.6166      0.218      7.428      0.000       1.190       2.044
DP04_0040PE       -0.5875      0.141     -4.176      0.000      -0.864      -0.311
DP03_0012PE       -0.5545      0.300     -1.851      0.064      -1.142       0.033
DP02_0115PE        1.2345      0.393      3.141      0.002       0.463       2.006
DP05_0027PE       -4.1916      0.324    -12.936      0.000      -4.828      -3.556
==============================================================================
Omnibus:                      829.831   Durbin-Watson:                   1.863
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           106870.447
Skew:                           3.417   Prob(JB):                         0.00
Kurtosis:                      54.815   Cond. No.                         278.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


Regression Model

OLS Model on Original Data
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:                outcome   R-squared (uncentered):                   0.985
Model:                            OLS   Adj. R-squared (uncentered):              0.984
Method:                 Least Squares   F-statistic:                              3299.
Date:                Tue, 21 Jul 2020   Prob (F-statistic):                        0.00
Time:                        14:13:00   Log-Likelihood:                         -4604.5
No. Observations:                 939   AIC:                                      9245.
Df Residuals:                     921   BIC:                                      9332.
Df Model:                          18                                                  
Covariance Type:            nonrobust                                                  
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
K00_K99_R1000     33.2001      0.564     58.895      0.000      32.094      34.306
DP05_0010PE        1.9132      0.334      5.735      0.000       1.258       2.568
DP03_0042PE        0.7256      0.208      3.483      0.001       0.317       1.134
DP02_0114PE        0.2099      0.198      1.060      0.290      -0.179       0.599
Residual_R1000     4.4848      0.485      9.239      0.000       3.532       5.438
N00_N99_R1000     11.4898      1.970      5.831      0.000       7.623      15.357
V01_Y89_R1000     -3.2159      0.217    -14.810      0.000      -3.642      -2.790
DP05_0002PE       -1.9853      0.352     -5.645      0.000      -2.676      -1.295
DP03_0004PE        0.6200      0.555      1.117      0.264      -0.469       1.709
DP03_0048PE        1.8964      0.356      5.324      0.000       1.197       2.595
DP03_0003PE        0.8657      0.650      1.332      0.183      -0.410       2.142
DP02_0129PE       -2.1868      0.841     -2.601      0.009      -3.837      -0.537
DP03_0047PE        1.8207      0.321      5.675      0.000       1.191       2.450
DP03_0024PE        1.6166      0.218      7.428      0.000       1.190       2.044
DP04_0040PE       -0.5875      0.141     -4.176      0.000      -0.864      -0.311
DP03_0012PE       -0.5545      0.300     -1.851      0.064      -1.142       0.033
DP02_0115PE        1.2345      0.393      3.141      0.002       0.463       2.006
DP05_0027PE       -4.1916      0.324    -12.936      0.000      -4.828      -3.556
==============================================================================
Omnibus:                      829.831   Durbin-Watson:                   1.863
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           106870.447
Skew:                           3.417   Prob(JB):                         0.00
Kurtosis:                      54.815   Cond. No.                         278.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

