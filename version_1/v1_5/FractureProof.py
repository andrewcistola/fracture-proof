# FractureProof
## Version 1.4

class FractureProof:
    # A three step process for feature selection using principal component analysis, random forests, and recursive feature selection for public health informatics. 
    def Find(df, quant, path = "", title = ""):
        import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
        import numpy as np # Widely used matrix library for numerical processes
        from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
        from sklearn.ensemble import RandomForestRegressor # Random Forest classification component
        from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
        from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome
        pop = df.pop(quant) # Remove quantitative outcome
        df = df.dropna(axis = 1, thresh = 0.75*len(df)) # Drop features less than 75% non-NA count for all columns
        df = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df), columns = df.columns) # Impute missing data
        df = pd.DataFrame(StandardScaler().fit_transform(df.values), columns = df.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
        df = df.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
        degree = len(df.columns) - 1 # Save number of features -1 to get degrees of freedom
        pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
        pca.fit(df) # Fit initial PCA model
        df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
        df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
        components = len(df_comp.index) - 1 # Save count of components for Variable reduction
        pca = PCA(n_components = components) # you will pass the number of components to make PCA model
        pca.fit_transform(df) # finally call fit_transform on the aggregate data to create PCA results object
        df_pc = pd.DataFrame(pca.components_, columns = df.columns) # Export eigenvectors to data frame with column names from original data
        df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
        df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
        df_pc = df_pc.abs() # Get absolute value of eigenvalues
        df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
        df_pca = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
        df_pca = df_pca[df_pca.MaxEV > df_pca.MaxEV.mean()] # Susbet by above average max eigenvalues 
        df_pca = df_pca.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
        df_pca = df_pca.rename(columns = {"index": "Features"}) # Rename former index as features
        print(df_pca)
        df.insert(0, "quant", pop) # Reattach qunatitative outcome to front of data frame
        X = df.drop(columns = ["quant"]) # Drop outcomes and targets
        Y = df["quant"] # Isolate Outcome variable
        forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
        forest.fit(X, Y) # Fit Forest model, This will take time
        rf = forest.feature_importances_ # Output importances of features
        l_rf = list(zip(X, rf)) # Create list of variables alongside importance scores 
        df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
        df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean
        print(df_rf)
        df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
        pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
        X = df[pca_rf] # Save features columns as predictor data frame
        Y = df["quant"] # Selected quantitative outcome from original data frame
        recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
        recursive.fit(X, Y) # This will take time
        rfe = recursive.support_ # Save Boolean values as numpy array
        l_rfe = list(zip(X, rfe)) # Create list of variables alongside RFE value 
        df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
        df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
        print(df_rfe)
        pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
        X = df.filter(pca_rf_rfe) # Keep only selected columns from rfe
        Y = df["quant"] # Add outcome variable
        regression = LinearRegression() # Linear Regression in scikit learn
        regression.fit(X, Y) # Fit model
        coef = regression.coef_ # Coefficient models as scipy array
        l_reg = list(zip(X, coef)) # Create list of variables alongside coefficient 
        df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names
        print(df_reg)
        df_final = pd.merge(df_pca_rf, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
        final = df_final["Features"].tolist() # Save chosen featres as list
        print(df_final) # Show in terminal
        df_final.to_csv(path + title + "_fp_v1.4_quant.csv") # Export df as csv

    def FindCat(self, df, cat, path = "", title = ""):
        import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
        import numpy as np # Widely used matrix library for numerical processes
        from sklearn.impute import SimpleImputer # Univariate imputation for missing data
        from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
        from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
        from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
        from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
        from sklearn.linear_model import LogisticRegression # Used for machine learning with categorical outcome
        pop = df.pop(cat) # Remove quantitative outcome
        df = df.dropna(axis = 1, thresh = 0.75*len(df)) # Drop features less than 75% non-NA count for all columns
        df = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df), columns = df.columns) # Impute missing data
        df = pd.DataFrame(StandardScaler().fit_transform(df.values), columns = df.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
        df = df.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
        degree = len(df.columns) - 1 # Save number of features -1 to get degrees of freedom
        pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
        pca.fit(df) # Fit initial PCA model
        df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
        df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
        components = len(df_comp.index) - 1 # Save count of components for Variable reduction
        pca = PCA(n_components = components) # you will pass the number of components to make PCA model
        pca.fit_transform(df) # finally call fit_transform on the aggregate data to create PCA results object
        df_pc = pd.DataFrame(pca.components_, columns = df.columns) # Export eigenvectors to data frame with column names from original data
        df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
        df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
        df_pc = df_pc.abs() # Get absolute value of eigenvalues
        df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
        df_pca = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
        df_pca = df_pca[df_pca.MaxEV > df_pca.MaxEV.mean()] # Susbet by above average max eigenvalues 
        df_pca = df_pca.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
        df_pca = df_pca.rename(columns = {"index": "Features"}) # Rename former index as features
        print(df_pca)
        df.insert(0, "cat", pop) # Reattach qunatitative outcome to front of data frame
        X = df.drop(columns = ["cat"]) # Drop outcomes and targets
        Y = df["cat"] # Isolate Outcome variable
        forest = RandomForestClassifier(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
        forest.fit(X, Y) # Fit Forest model, This will take time
        rf = forest.feature_importances_ # Output importances of features
        l_rf = list(zip(X, rf)) # Create list of variables alongside importance scores 
        df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
        df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean
        print(df_rf)
        df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
        pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
        X = df[pca_rf] # Save features columns as predictor data frame
        Y = df["cat"] # Selected quantitative outcome from original data frame
        recursive = RFECV(estimator = LogisticRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
        recursive.fit(X, Y) # This will take time
        rfe = recursive.support_ # Save Boolean values as numpy array
        l_rfe = list(zip(X, rfe)) # Create list of variables alongside RFE value 
        df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
        df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
        print(df_rfe)
        pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
        X = df[pca_rf_rfe] # Keep only selected columns from rfe
        Y = df["cat"] # Add outcome variable
        regression = LogisticRegression() # Linear Regression in scikit learn
        regression.fit(X, Y) # Fit model
        coef = regression.coef_ # Coefficient models as scipy array
        l_reg = list(zip(X, coef.T)) # Create list of variables alongside coefficient 
        df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names
        print(df_reg)
        df_final = pd.merge(df_pca_rf, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
        final = df_final["Features"].tolist() # Save chosen featres as list
        print(df_final) # Show in terminal
        df_final.to_csv(path + title + "_fp_v1.4_cat.csv") # Export df as csv

    def Predict(self, df, quant, train, test, features, path = "", title = ""):
        import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
        import numpy as np # Widely used matrix library for numerical processes
        import statsmodels.api as sm # Statistics package best for regression models for statistical tests
        from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
        FractureProofQuant(df = df, quant = quant, path = path, title = title)
        features = df_final["Features"]
        df = df.filter(features) # Subset by hand selected features for model
        df = df.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
        X = df.drop(columns = [quant, train, test]) # features as x
        Y = df[quant] # Save outcome variable as y
        mod = sm.OLS(Y, X) # Describe linear model
        res = mod.fit() # Fit model
        print(res.summary()) # Summarize model
        df = pd.DataFrame(StandardScaler().fit_transform(df.values), columns = df.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
        input = df.shape[1] - 4 # Save number of columns as length minus quant, test, train and round to nearest integer
        nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
        classifier = Sequential() # Sequential model building in keras
        classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
        classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
        classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
        X = df.drop(columns = [quant, train, test]) # Save features as X numpy data array
        Y_train = df[train] # Save test outcome as Y numpy data array
        classifier.fit(X, Y_train, batch_size = 10, epochs = 50) # Fitting the data to the train outcome
        Y_pred = classifier.predict(X) # Predict values from testing model
        Y_pred = (Y_pred > 0.5)
        Y_train = (Y_train > 0)
        fpr, tpr, threshold = roc_curve(Y_train, Y_pred) # Create ROC outputs, true positive rate and false positive rate
        auc_train = auc(fpr, tpr) # Plot ROC and get AUC score
        print(auc_train)
        Y_test = df[test] # Save train outcome as Y numpy data array
        classifier.fit(X, Y_test, batch_size = 10, epochs = 50) # Fitting the data to the train outcome
        Y_pred = classifier.predict(X) # Predict values from testing model
        Y_pred = (Y_pred > 0.5)
        Y_test = (Y_test > 0)
        fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
        auc_test = auc(fpr, tpr) # Plot ROC and get AUC score
        print(auc_test)
        text_file = open(path + title + "_fp_v1.4_sup.txt", "w") # Open text file and name with subproject, content, and result suffix
        text_file.write(str(res.summary())) # Line of text with space after
        text_file.write("\n\n") # Add two lines of blank text at end of every section text
        text_file.write("C-Statistic Train = " + str(auc_train) + "\n") # Line of text with space after
        text_file.write("C-Statistic Test = " + str(auc_test) + "\n") # Line of text with space after
        text_file.close() # Close file

    def Chaos(self, df, id, path = "", title = ""):
        import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
        import numpy as np # Widely used matrix library for numerical processes
        from sklearn.impute import SimpleImputer # Univariate imputation for missing data
        from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
        ID = df.pop(id) # Remove quantitative outcome
        df = df.dropna(axis = 1, thresh = 0.75*len(df)) # Drop features less than 75% non-NA count for all columns
        df = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df), columns = df.columns) # Impute missing data
        df = pd.DataFrame(StandardScaler().fit_transform(df.values), columns = df.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
        df = df.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
        degree = len(df.index) - 1 # Save number of features -1 to get degrees of freedom
        pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
        pca.fit(df) # Fit initial PCA model
        df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
        df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
        clusters = len(df_comp.index) # Save count of components for Variable reduction
        kmeans = KMeans(n_clusters = clusters, random_state = 0) # Setup Kmeans model, pre-select number of clusters
        kmeans.fit(df) # Fit Kmeans
        km = kmeans.labels_ # Output importances of features
        l_km = list(zip(ID, km)) # Create list of variables alongside importance scores 
        df_cl = pd.DataFrame(l_km, columns = ["ID", "Cluster"]) # Create data frame of importances with variables and gini column names
        df.to_csv(path + title + "_fp_v1.4_un.csv") # Export df as csv
        print(df["ID"].unique())
        for x in df["ID"].unique():
            df[x] = np.where(df_["Cluster"] == 1, 1, 0) # Create New Column Based on Conditions
            df_x = df # Drop all rows with NA values (should be none, this is just to confirm)
            print(df_x)
            FractureProofCat(df = df_x, cat = x, path = path, title = x)







