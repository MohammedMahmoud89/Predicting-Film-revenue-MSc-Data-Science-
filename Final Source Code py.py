#!/usr/bin/env python
# coding: utf-8

# # Data Science master degree project's source code
# 
# ## Topic: Predicting films revenue using internal film industry-based features combined with external world’s Social and Economic indicators
# 
# 
# ### Contents:
# <ul>
# <li><a href="#Importing the required libraries">Importing the required libraries</a></li>
# <li><a href="#Creating functions">Creating functions</a></li>    
# <li><a href="#Data Wrangling">Data Wrangling</a></li>
# <li><a href="#Exploratory analysis">Exploratory analysis</a></li>
# <li><a href="#Preparing and applying the machine learning models">Preparing and applying the machine learning models</a></li>
# <li><a href="#Regression 'value estimation'">Regression 'value estimation'</a></li>
# <li><a href="#Binary Classification">Binary Classification</a></li>
# <li><a href="#Multi Classification">Multi Classification</a></li>
# <li><a href="#References">References</a></li>    
# </ul>

# In[1]:


#test


# <a id='Importing the required libraries'></a>
# # Importing the required libraries:

# In[2]:


import pandas as pd
print("Pandas version:", pd.__version__)
import numpy as np
print("NumPy version:", np.__version__)
import matplotlib
print("Matplotlib version:", matplotlib.__version__)
import matplotlib.pyplot as plt
import seaborn as sns
print("seaborn version:", sns.__version__)
from matplotlib.offsetbox import AnchoredText
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import export_graphviz
from subprocess import check_call
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import chi2
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import f_regression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import r2_score
import catboost
print("CatBoost version:", catboost.__version__)
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import xgboost
print("xgboost version:", xgboost.__version__)
from lightgbm import LGBMRegressor
import lightgbm 
print("lightgbm version:", lightgbm.__version__)
from sklearn.metrics import explained_variance_score, median_absolute_error
from sklearn.preprocessing import Binarizer
from sklearn.metrics import make_scorer, r2_score
print("scikit-learn version:", sklearn.__version__)
import time


# <a id='Creating functions'></a>
# # Creating Functions:

# In[3]:


#Function to read the datasets
def read_dataset(dataset):
    #references: [25]
    df=pd.read_csv(dataset) 
    return df


# In[4]:


#Function to change the data type of selected column in the dataframe
def change_type_to(df,col,to):
    #references: [26]
    df[col]=df[col].astype(to) 
    return df[col]


# In[5]:


#Ref [1]
#function to apply the label encodeing for the selected categorical columns
def labelencode (df,columns):
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df


# In[6]:


# reference: [2] 
# function to standerdise the distribution of the training and testing fearures
def scale (x_train,x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return (x_train,x_test)


# In[7]:


# reference: [2]
#Function to perform the decision tree model 
def DT (x_train, y_train,x_test, y_test):
    decision_tree = tree.DecisionTreeClassifier()
    fitted_tree = decision_tree.fit(x_train, y_train)
    tree_train_acc = fitted_tree.score(x_train, y_train)
    tree_test_acc = fitted_tree.score(x_test, y_test)
    print(f"Decision Tree training data accuracy: {round(tree_train_acc, 3)}")
    print(f"Decision Tree test data accuracy: {round(tree_test_acc, 3)}")


# In[8]:


#reference: [3],[17],[22],[23],[24]
#Function to perform linear regression model:
def linreg(xtrain, xtest, ytrain, ytest,X):
    lr_model = LinearRegression()
    lr_model.fit(xtrain, ytrain)
    
    
    #Training:
    y_pred_training = lr_model.predict(xtrain)
    mse_training=metrics.mean_squared_error(ytrain, y_pred_training)
    print('Mean squared error (MSE) for training set : ', mse_training)
    r2_training = r2_score(ytrain, y_pred_training)
    print("R squared for training set is:", r2_training)
    mae_training = mean_absolute_error(ytrain, y_pred_training)
    print('Mean absolute error (MAE) for training set is  : ', mae_training)
    rmse_training = np.sqrt(mse_training)
    print('Root mean squared error (RMSE) for the training set  : ', rmse_training)

    
    #Testing:
    y_pred = lr_model.predict(xtest)
    # Mean Square Error calculation: 
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean squared error (MSE) for testing set : ', mse)
    r2 = r2_score(ytest, y_pred)
    print("R squared for testing set is:", r2)
    mae = mean_absolute_error(ytest, y_pred)
    print('Mean absolute error (MAE) for testing set is  : ', mae)
    rmse = np.sqrt(mse)
    print('Root mean squared error (RMSE) for the testing set  : ', rmse_training)
    
    
   
    print("Coefficients are : ",lr_model.coef_)
    print("Intercept is :" ,lr_model.intercept_)
    
    #https://www.statology.org/sklearn-regression-coefficients/


# In[9]:


#reference: [4],[14],[15],[17],[22],[23],[24]
#function to perform lightgbm model:
def light (xtrain,xtest,ytrain,ytest,X,Y):
    lgb_train_data = lgb.Dataset(xtrain, label=ytrain)
    # hyper parameters of the model
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42
    }
    
    lgb_model = lgb.train(params, lgb_train_data, num_boost_round=100)

        
     #Training:
    y_pred_training = lgb_model.predict(xtrain)
    mse_training=metrics.mean_squared_error(ytrain, y_pred_training)
    print('Mean squared error (MSE) for training set : ', mse_training)
    r2_training = r2_score(ytrain, y_pred_training)
    print("R squared for training set is:", r2_training)
    mae_training = mean_absolute_error(ytrain, y_pred_training)
    print('Mean absolute error (MAE) for training set is  : ', mae_training)
    rmse_training = np.sqrt(mse_training)
    print('Root mean squared error (RMSE) for the training set  : ', rmse_training)

    
    #Testing:
    y_pred = lgb_model.predict(xtest)
    # Mean Square Error calculation: 
    mse=metrics.mean_squared_error(ytest, y_pred)
    print('Mean squared error (MSE) for testing set : ', mse)
    r2 = r2_score(ytest, y_pred)
    print("R squared for testing set is:", r2)
    mae = mean_absolute_error(ytest, y_pred)
    print('Mean absolute error (MAE) for testing set is  : ', mae)
    rmse = np.sqrt(mse)
    print('Root mean squared error (RMSE) for the testing set  : ', rmse_training)   
        

    

    
    # K fold :
    # Define scoring function to be R squared
    scoring = make_scorer(r2_score)
    # Initialization:
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    
    # Perform k-fold cross-validation and calculate R-squared for each fold
    results = cross_val_score(LGBMRegressor(**params), X, Y, cv=kfold, scoring=scoring)
    
    # Calculate mean and standard deviation of R-squared
    mean_r_squared = results.mean()
    std_r_squared = results.std()
    # Print the results
    print("Kfold results are : ")
    print(f"Mean R squared: {mean_r_squared}")
    print(f"Results are : {results}")
    print(f"Standard Deviation of R squared: {std_r_squared}")


# In[10]:


# install:
#pip install xgboost
#!pip install catboost


# In[11]:


#ref :[4],[17],[19],[20],[22],[23],[24]
# function of the extreme gradient boosting model:
def xgboostmodel (x_enc_s_train,y_enc_s_train, x_enc_s_test,y_enc_s_test,X,Y):
    xgb_model = xgb.XGBRegressor(random_state=42)
    # Training:
    xgb_model.fit(x_enc_s_train,y_enc_s_train)

    #Training:
    y_pred_training = xgb_model.predict(x_enc_s_train)
    mse_training=metrics.mean_squared_error(y_enc_s_train, y_pred_training)
    print('Mean squared error (MSE) for training set : ', mse_training)
    r2_training = r2_score(y_enc_s_train, y_pred_training)
    print("R squared for training set is:", r2_training)
    mae_training = mean_absolute_error(y_enc_s_train, y_pred_training)
    print('Mean absolute error (MAE) for training set is  : ', mae_training)
    rmse_training = np.sqrt(mse_training)
    print('Root mean squared error (RMSE) for the training set  : ', rmse_training)

    
    #Testing:
    y_pred = xgb_model.predict(x_enc_s_test)
    # Mean Square Error calculation: 
    mse=metrics.mean_squared_error(y_enc_s_test, y_pred)
    print('Mean squared error (MSE) for testing set : ', mse)
    r2 = r2_score(y_enc_s_test, y_pred)
    print("R squared for testing set is:", r2)
    mae = mean_absolute_error(y_enc_s_test, y_pred)
    print('Mean absolute error (MAE) for testing set is  : ', mae)
    rmse = np.sqrt(mse)
    print('Root mean squared error (RMSE) for the testing set  : ', rmse_training)  
    


    # Assigning R squared to be the scoring indicator:
    scoring = make_scorer(r2_score)
    # initilization
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    # Perform k-fold cross-validation and calculate R-squared for each fold
    results = cross_val_score(xgb_model, X, Y, cv=kfold, scoring=scoring)
    # mean and standard deviation of R squared calculation:
    mean_r_squared = results.mean()
    std_r_squared = results.std()
    print("Kfold results are : ")
    print(f"Mean R squared: {mean_r_squared}")
    print(f"Results are : {results}")
    print(f"Standard deviation of R squared: {std_r_squared}")


# In[12]:


#References: [4],[17],[21],[22],[24]
#function of Catboost model:

def catboostmodel (x_enc_s_train,y_enc_s_train, x_enc_s_test,y_enc_s_test,X,Y):
        
    model = CatBoostRegressor(iterations=1000,  # Number of training iterations a
                          learning_rate=0.1,  # It is according to the new tress preictions as in the formula on [21]
                          depth=6,  # depth of each tree [21]
                          loss_function='RMSE',  # optimized metric [21]
                          random_seed=42  # The extent of how much results are repeated [21] 
                          )
    model.fit(x_enc_s_train, y_enc_s_train, verbose=100)  #verbose is to see the numebr of times that the output is printed[21]
    
    #Training:
    y_pred_training = model.predict(x_enc_s_train)
    mse_training=metrics.mean_squared_error(y_enc_s_train, y_pred_training)
    print('Mean squared error (MSE) for training set : ', mse_training)
    r2_training = r2_score(y_enc_s_train, y_pred_training)
    print("R squared for training set is:", r2_training)
    mae_training = mean_absolute_error(y_enc_s_train, y_pred_training)
    print('Mean absolute error (MAE) for training set is  : ', mae_training)
    rmse_training = np.sqrt(mse_training)
    print('Root mean squared error (RMSE) for the training set  : ', rmse_training)

    
    #Testing:
    y_pred = model.predict(x_enc_s_test)
    # Mean Square Error calculation: 
    mse=metrics.mean_squared_error(y_enc_s_test, y_pred)
    print('Mean squared error (MSE) for testing set : ', mse)
    r2 = r2_score(y_enc_s_test, y_pred)
    print("R squared for testing set is:", r2)
    mae = mean_absolute_error(y_enc_s_test, y_pred)
    print('Mean absolute error (MAE) for testing set is  : ', mae)
    rmse = np.sqrt(mse)
    print('Root mean squared error (RMSE) for the testing set  : ', rmse_training)
    
    
    
    # K fold :
    
    # Assigning R squared to be the scoring indicator:
    scoring = make_scorer(r2_score)
    # Initialization
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    # Perform k-fold cross-validation and calculate R-squared for each fold
    results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    # mean and standard deviation of R squared calculation:
    mean_r_squared = results.mean()
    std_r_squared = results.std()
    # Print the results
    print("Kfold results are : ")
    print(f"Mean R squared: {mean_r_squared}")
    print(f"Results are : {results}")
    print(f"Standard deviation of R squared: {std_r_squared}")


# In[13]:


#function to encode all the categorical features into numerical:(here it is not done after splitting to training and testing sets) )
def labelencoding (x):
    #References : [1],[52],[26]
    #confirming the datatype of the categorical columns by converting them to strings :
    cat_columns = x.select_dtypes(include='object').columns
    x[cat_columns] = x[cat_columns].astype(str)

    # Copy the original dataFrame to avoid modifying it
    encoded_df_x = x.copy()
    # Create a LabelEncoder instance
    label_encoder = LabelEncoder()
    # Iterate over each column in the dataFrame to make sure whether it is categorical or not
    for col in encoded_df_x.columns:
        # Check if the column is of object datatype (categorical)
        if encoded_df_x[col].dtype == 'object':
            # Fit and transform the LabelEncoder on the current column
            encoded_df_x[col] = label_encoder.fit_transform(encoded_df_x[col])
            
    return encoded_df_x


# In[14]:


def select_top22_regression_features (x,encoded_df_x,y):
    
    #select top 22 features using f_regression:
    #reference:[54]
    test = SelectKBest(score_func=f_regression, k=22)
    fit = test.fit(encoded_df_x,y)

    # summarize scores
    set_printoptions(precision=3)
    # print(fit.scores_)
    features = fit.transform(encoded_df_x)

    column_names=x.columns
    #sorting and writing tuples of the fetature name and corresponding score and printing them:
    top_features = sorted(zip(column_names, fit.scores_), key=lambda x: x[1], reverse=True)
    print(top_features[:22])
    
    #references :[69],[70],[71]
    # retrieving coulmn names and scores separately and putting them into lists
    column_names = [feature[0] for feature in top_features] #list of column names
    scores = [feature[1] for feature in top_features] #list of corresponding scores
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(column_names[0:22], scores[0:22], color='blue')
    plt.xlabel('Column names')
    plt.ylabel('Scores')
    plt.title('Top features')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# In[15]:


#references : [54]
def select_top21_binary_class_features (x,encoded_df_x,y_bin):
    #select top 21 features using chi2:
    test = SelectKBest(score_func=chi2, k=21)
    fit = test.fit(encoded_df_x,y_bin)

    # summarize scores
    set_printoptions(precision=3)
    # print(fit.scores_)
    features = fit.transform(encoded_df_x)

    column_names=x.columns

    top_features = sorted(zip(column_names, fit.scores_), key=lambda x: x[1], reverse=True)
    print(top_features[:21])
    
    column_names = [feature[0] for feature in top_features]
    scores = [feature[1] for feature in top_features]
    
    
    
    # Plotting the bar chart
    #references :[69],[70],[71],[72]
    plt.figure(figsize=(10, 6))
    plt.bar(column_names[0:21], scores[0:21], color='blue')
    plt.xlabel('Column names')
    plt.ylabel('Scores (log scale)')  
    plt.title('Top features')
    plt.xticks(rotation=90)
    # Set log scale on the y axis
    plt.yscale('log')  
    plt.tight_layout()
    plt.show()


# In[16]:


#references [4],[56],[58],[59],[60],[61]
# Function to perform logistic regression classification model:
def LogR (X_train, y_train,X_test,y_test,rescaledX,y_bin): 
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    
    #training
    logreg_pred_tr = logreg.predict(X_train)
    logreg_accuracy_tr = accuracy_score(y_train, logreg_pred_tr)
    print("Logistic Regression Accuracy:", logreg_accuracy_tr)
    auc_roc_tr = roc_auc_score(y_train,logreg_pred_tr)
    print("Training AUC-ROC:", auc_roc_tr)
    y_pred_precision_tr = np.round(logreg_pred_tr)  
    precision_tr = precision_score(y_train, y_pred_precision_tr)
    print("Training Precision:", precision_tr)
    
    
    #testing
    logreg_pred = logreg.predict(X_test)
    logreg_accuracy = accuracy_score(y_test, logreg_pred)
    print("Logistic Regression Accuracy:", logreg_accuracy)
    auc_roc = roc_auc_score(y_test,logreg_pred)
    print("AUC-ROC:", auc_roc)
    y_pred_precision = np.round(logreg_pred)  
    precision = precision_score(y_test, y_pred_precision)
    print("Precision:", precision)

    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed ,shuffle=True)
    results = cross_val_score(logreg, rescaledX,y_bin, cv=kfold)
    print ("K fold results are: ")
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[17]:


#references [4],[57],[58],[59],[60],[61]
# Function to perform decision tree  classification model:
def DT_bin (X_train, y_train,X_test,y_test,rescaledX,y_bin):
    d_tree_clf = DecisionTreeClassifier()
    d_tree_clf.fit(X_train, y_train)
    
    
    #Training
    dt_pred_tr = d_tree_clf.predict(X_train)
    dt_accuracy_tr = accuracy_score(y_train, dt_pred_tr)
    print("Training Decision tree accuracy:", dt_accuracy_tr)
    auc_roc_tr = roc_auc_score(y_train,dt_pred_tr)
    print("Training AUC-ROC:", auc_roc_tr)
    y_pred_precision_tr = np.round(dt_pred_tr)  
    precision_tr = precision_score(y_train, y_pred_precision_tr)
    print("Precision:", precision_tr)
    

    #Testing
    dt_pred = d_tree_clf.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    print("Decision tree testing accuracy:", dt_accuracy)
    auc_roc = roc_auc_score(y_test,dt_pred)
    print("Testing AUC-ROC:", auc_roc)
    y_pred_precision = np.round(dt_pred)  
    precision = precision_score(y_test, y_pred_precision)
    print("Testing Precision:", precision)

    #reference [73]
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed ,shuffle=True)
    results = cross_val_score(d_tree_clf, rescaledX,y_bin, cv=kfold)
    print ("K fold results are: ")
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[18]:


#references [4],[58],[59],[60],[61],[62]
#Function to perform random forest classification model
def RF_bin (X_train, y_train,X_test,y_test,rescaledX,y_bin):
    random_f_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    random_f_clf.fit(X_train, y_train)
    
    #Training
    rf_pred_tr = random_f_clf.predict(X_train)
    rf_accuracy_tr = accuracy_score(y_train, rf_pred_tr)
    print("Training Random forest accuracy:", rf_accuracy_tr)
    auc_roc_tr = roc_auc_score(y_train,rf_pred_tr)
    print("Training AUC-ROC:", auc_roc_tr)
    y_pred_precision_tr = np.round(rf_pred_tr) 
    precision_tr = precision_score(y_train, rf_pred_tr)
    print("Training Precision:", precision_tr)
    
    
    #Testing
    rf_pred = random_f_clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print("Random forest testing accuracy:", rf_accuracy)
    auc_roc = roc_auc_score(y_test,rf_pred)
    print("Testing AUC-ROC:", auc_roc)
    y_pred_precision = np.round(rf_pred) 
    precision = precision_score(y_test, rf_pred)
    print("Testing Precision:", precision)

    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed ,shuffle=True)
    results = cross_val_score(random_f_clf, rescaledX,y_bin, cv=kfold)
    print ("K fold results are: ")
    print(f"results are : {results}")
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[19]:


#references [4],[58],[59],[60],[61],[63]
# Function for extreme gradient boosting machine classification model
def xgb_bin (X_train, y_train,X_test,y_test,rescaledX,y_bin):
    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_clf.fit(X_train, y_train)
    
    
    #Training
    xgb_pred_tr = xgb_clf.predict(X_train)
    xgb_accuracy_tr = accuracy_score(y_train, xgb_pred_tr)
    print("XGB Training accuracy:", xgb_accuracy_tr)
    auc_roc_tr = roc_auc_score(y_train,xgb_pred_tr)
    print("Training AUC-ROC:", auc_roc_tr)
    #rounding the predctions results
    y_pred_precision_tr = np.round(xgb_pred_tr)  
    precision_tr = precision_score(y_train, xgb_pred_tr)
    print("Training Precision:", precision_tr)
    
    #Testing
    xgb_pred = xgb_clf.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print("XGB testing accuracy:", xgb_accuracy)
    auc_roc = roc_auc_score(y_test,xgb_pred)
    print("Testing AUC-ROC:", auc_roc)
    #rounding the predctions results
    y_pred_precision = np.round(xgb_pred)  
    precision = precision_score(y_test, xgb_pred)
    print("Testing Precision:", precision)
    
    #reference [73]
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed ,shuffle=True)
    results = cross_val_score(xgb_clf, rescaledX,y_bin, cv=kfold)
    print ("K fold results are: ")
    print(f"results are : {results}")
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[20]:


#references [4],[56],[58],[59],[60],[61],[64],[65],[66]
# Function to perform logistic regression multi classification model
def LogR_multi (X_train, y_train,X_test,y_test,rescaledX,y_6classes):
    logreg_clf = LogisticRegression()
    logreg_clf.fit(X_train, y_train)
    
    
    #Training
    logreg_pred_tr = logreg_clf.predict(X_train)
    logreg_pred_prob_tr = logreg_clf.predict_proba(X_train)
    logreg_accuracy_tr = accuracy_score(y_train, logreg_pred_tr)
    print("Training Logistic Regression Accuracy:", logreg_accuracy_tr)
    # multi_class='ovr' as per [65] is to use binary classification algorithms to multi classification problems
    auc_roc_tr = roc_auc_score(y_train,logreg_pred_prob_tr,average='macro',multi_class='ovr')
    print("Training AUC-ROC:", auc_roc_tr)
    y_pred_precision_tr = np.round(logreg_pred_tr) 
    precision_tr = precision_score(y_train, y_pred_precision_tr,average='macro')
    print("Training Precision:", precision_tr)
    
    # from the formula in [66],first the exact and 1 class away classified cases are counted then divided by the sample quantity 
    one_a_corr_pred_tr=0
    for true_label_tr, pred_label_tr in zip(y_train, logreg_pred_tr):
        if abs(true_label_tr - pred_label_tr) <= 1:
            one_a_corr_pred_tr += 1
    
    one_a_acc_tr = one_a_corr_pred_tr / len(y_train)
    print("Training One away accuracy:", one_a_acc_tr)
    
    
    
    
    #Testing
    logreg_pred = logreg_clf.predict(X_test)
    logreg_pred_prob = logreg_clf.predict_proba(X_test)
    logreg_accuracy = accuracy_score(y_test, logreg_pred)
    print("Logistic regression testing accuracy:", logreg_accuracy)
    # multi_class='ovr' as per [65] is to use binary classification algorithms to multi classification problems
    auc_roc = roc_auc_score(y_test,logreg_pred_prob,average='macro',multi_class='ovr')
    print("Testing AUC-ROC:", auc_roc)
    y_pred_precision = np.round(logreg_pred) 
    precision = precision_score(y_test, y_pred_precision,average='macro')
    print("Testing Precision:", precision)
    
    # from the formula in [66],first the exact and 1 class away classified cases are counted then divided by the sample quantity 
    one_a_corr_pred=0
    for true_label, pred_label in zip(y_test, logreg_pred):
        if abs(true_label - pred_label) <= 1:
            one_a_corr_pred += 1
    
    one_a_acc = one_a_corr_pred / len(y_test)
    print("Testing One away accuracy:", one_a_acc)

    #reference [73]
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed ,shuffle=True)
    results = cross_val_score(logreg_clf, rescaledX,y_6classes, cv=kfold)
    print ("K fold results are: ")
    print (f"results are : {results}")
    print("Avg Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[21]:


#references [4],[57],[58],[59],[60],[61],[65],[66]
# Function for decision tree multi classification model 
def DT_multi (X_train, y_train,X_test,y_test,rescaledX,y_6classes):
    d_tree_clf = DecisionTreeClassifier()
    d_tree_clf.fit(X_train, y_train)
    
    #training
    dt_pred_tr = d_tree_clf.predict(X_train)
    dt_pred_prob_tr = d_tree_clf.predict_proba(X_train)
    dt_accuracy_tr = accuracy_score(y_train, dt_pred_tr)
    print("Training Decision Tree Accuracy:", dt_accuracy_tr)
    # multi_class='ovr' as per [65] is to use binary classification algorithms to multi classification problems
    auc_roc_tr = roc_auc_score(y_train,dt_pred_prob_tr,average='macro',multi_class='ovr')
    print("Training AUC-ROC:", auc_roc_tr)
    y_pred_precision_tr = np.round(dt_pred_tr)  
    precision_tr = precision_score(y_train, y_pred_precision_tr,average='macro')
    print("Training Precision:", precision_tr)
    
    # from the formula in [66],first the exact and 1 class away classified cases are counted then divided by the sample quantity
    one_a_corr_pred_tr=0
    for true_label_tr, pred_label_tr in zip(y_train, dt_pred_tr):
        if abs(true_label_tr - pred_label_tr) <= 1:
            one_a_corr_pred_tr += 1

    one_a_acc_tr = one_a_corr_pred_tr / len(y_train)
    print("Training One away accuracy:", one_a_acc_tr)
    
    
    
    
    #testing 
    dt_pred = d_tree_clf.predict(X_test)
    dt_pred_prob = d_tree_clf.predict_proba(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    print("Decision tree testing accuracy:", dt_accuracy)
    # multi_class='ovr' as per [65] is to use binary classification algorithms to multi classification problems
    auc_roc = roc_auc_score(y_test,dt_pred_prob,average='macro',multi_class='ovr')
    print("Testing AUC-ROC:", auc_roc)
    y_pred_precision = np.round(dt_pred)  # Convert probabilities to binary predictions (0 or 1)
    precision = precision_score(y_test, y_pred_precision,average='macro')
    print("Testing Precision:", precision)
    
    # from the formula in [66],first the exact and 1 class away classified cases are counted then divided by the sample quantity
    one_a_corr_pred=0
    for true_label, pred_label in zip(y_test, dt_pred):
        if abs(true_label - pred_label) <= 1:
            one_a_corr_pred += 1

    one_a_acc = one_a_corr_pred / len(y_test)
    print("Testing One away accuracy:", one_a_acc)
    
    #reference [73]
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed ,shuffle=True)
    results = cross_val_score(d_tree_clf, rescaledX,y_6classes, cv=kfold)
    print ("K fold results are: ")
    print (f"results are : {results}")
    print("Avg Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))    


# In[22]:


#references [4],[58],[59],[60],[61],[62],[65],[66]
# Function to perform random forest multi classification
def RF_multi (X_train, y_train,X_test,y_test,rescaledX,y_6classes):
    random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_clf.fit(X_train, y_train)
    
    
    #training
    rf_pred_tr = random_forest_clf.predict(X_train)
    rf_pred_prob_tr = random_forest_clf.predict_proba(X_train)
    rf_accuracy_tr = accuracy_score(y_train, rf_pred_tr)
    print("Training Random Forest Accuracy:", rf_accuracy_tr)
    auc_roc_tr = roc_auc_score(y_train,rf_pred_prob_tr,average='macro',multi_class='ovr')
    print("Training AUC-ROC:", auc_roc_tr)
    y_pred_precision_tr = np.round(rf_pred_tr)
    precision_tr = precision_score(y_train, rf_pred_tr,average='macro')
    print("Training Precision:", precision_tr)
    # from the formula in [66],first the exact and 1 class away classified cases are counted then divided by the sample quantity
    one_a_corr_pred_tr=0
    for true_label_tr, pred_label_tr in zip(y_train, rf_pred_tr):
        if abs(true_label_tr - pred_label_tr) <= 1:
            one_a_corr_pred_tr += 1

    one_a_acc_tr = one_a_corr_pred_tr / len(y_train)
    print("Training One away accuracy:", one_a_acc_tr)

    #testing
    rf_pred = random_forest_clf.predict(X_test)
    rf_pred_prob = random_forest_clf.predict_proba(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print("Random forest testing accuracy:", rf_accuracy)
    auc_roc = roc_auc_score(y_test,rf_pred_prob,average='macro',multi_class='ovr')
    print("Testing AUC-ROC:", auc_roc)
    y_pred_precision = np.round(rf_pred)
    precision = precision_score(y_test, rf_pred,average='macro')
    print("Precision:", precision)

    one_a_corr_pred=0
    for true_label, pred_label in zip(y_test, rf_pred):
        if abs(true_label - pred_label) <= 1:
            one_a_corr_pred += 1

    one_a_acc = one_a_corr_pred / len(y_test)
    print("Testing One away accuracy:", one_a_acc)


    #reference [73]
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed ,shuffle=True)
    results = cross_val_score(random_forest_clf, rescaledX,y_6classes, cv=kfold)
    print ("K fold results are: ")
    print (f"results are : {results}")
    print("Avg Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[23]:


#references [4],[58],[59],[60],[61],[65],[66],[63]
# function for extreme gradient boosting model for multi classification (XGBoost)
def xgb_multi (X_train, y_train,X_test,y_test,rescaledX,y_6classes):
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    
    #training 
    xgb_pred_tr = xgb_model.predict(X_train)
    xgb_pred_prob_tr = xgb_model.predict_proba(X_train)
    xgb_accuracy_tr = accuracy_score(y_train, xgb_pred_tr)
    print("Training XGBoost accuracy:", xgb_accuracy_tr)
    auc_roc_tr = roc_auc_score(y_train,xgb_pred_prob_tr,average='macro',multi_class='ovr')
    print("Training AUC-ROC:", auc_roc_tr)
    y_pred_precision_tr = np.round(xgb_pred_tr)
    precision_tr = precision_score(y_train, xgb_pred_tr,average='macro')
    print("Training Precision:", precision_tr)
    # from the formula in [66],first the exact and 1 class away classified cases are counted then divided by the sample quantity
    one_a_corr_pred_tr=0
    for true_label_tr, pred_label_tr in zip(y_train, xgb_pred_tr):
        if abs(true_label_tr - pred_label_tr) <= 1:
            one_a_corr_pred_tr += 1

    one_a_acc_tr = one_a_corr_pred_tr / len(y_train)
    print("Training One away accuracy:", one_a_acc_tr)
    
    
    #testing 
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_prob = xgb_model.predict_proba(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print("XGBoost testing accuracy:", xgb_accuracy)
    auc_roc = roc_auc_score(y_test,xgb_pred_prob,average='macro',multi_class='ovr')
    print("Testing AUC-ROC:", auc_roc)
    y_pred_precision = np.round(xgb_pred)
    precision = precision_score(y_test, xgb_pred,average='macro')
    print("Precision:", precision)

    
    one_a_corr_pred = 0

    for true_label, pred_label in zip(y_test, xgb_pred):
        if abs(true_label - pred_label) <= 1:
            one_a_corr_pred += 1

    one_a_acc = one_a_corr_pred / len(y_test)
    print("One away accuracy:", one_a_acc)


    #reference [73]
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed ,shuffle=True)
    results = cross_val_score(xgb_model, rescaledX,y_6classes, cv=kfold)
    print ("K fold results are: ")
    print (f"results are : {results}")
    print("Avg Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# <a id='Data Wrangling'></a>
# # Data Wrangling: 

# ## Data gathering (Exploring (Discovery)):

# ### Read all datasets and cleaning and merging them: 

# In[24]:


#Read all datasets:
df_movies_meta=read_dataset('movies_metadata.csv')
df_credits=read_dataset('credits.csv')
df_filmtv=read_dataset('filmtv_movies - ENG.csv')
big_datasets=read_dataset('big_datasets.csv')
big_df=read_dataset('big.csv')
GDP_ALL=read_dataset('GDP ALL.csv')
INFLATION_ALL=read_dataset('INFLATION ALL.csv')
POPULATION_ALL=read_dataset('POPULATION ALL.csv')
INTERNET=read_dataset('internet_use2.csv')
UNEMPLOYMENT=read_dataset('unemp2.csv')
OSCAR=read_dataset('the_oscar_award 1927 -2023.csv')


# In[25]:


df_movies_meta.columns


# In[26]:


df_movies_meta.head(1)


# In[27]:


df_movies_meta.shape


# In[28]:


df_movies_meta.info()


# In[29]:


df_movies_meta.describe()


# In[30]:


df_filmtv.columns


# In[31]:


df_filmtv.shape


# In[32]:


df_filmtv.head(1)


# ## Structuring:

# ### Merging the main datasets and adjusting the collective coulmns and data

# In[33]:


#references:[27]
# Coverting the column 'release_date' into three seprate columns ,year,month,and day.
df_movies_meta[["year","month","day" ]] = df_movies_meta["release_date"].str.split("-", expand = True)


# In[34]:


#investigating the coulmns' datatypes
df_movies_meta.dtypes


# In[35]:


#change month and day columns types from objects to numerical (float) :
df_movies_meta['month']=change_type_to(df_movies_meta,'month',float)
df_movies_meta['day']=change_type_to(df_movies_meta,'day',float)
df_movies_meta['year']=change_type_to(df_movies_meta,'year',float)


# In[36]:


# discovering the descriptive statistics of the dataset features
df_movies_meta.describe()


# In[37]:


# Change datatype of 'id' to prepare it for merging :
df_credits['id']=change_type_to(df_credits,'id',str)


# In[38]:


# Reference : [28] 
# Merging datasets (movies dataset and credits dataset): 
df_merge=df_movies_meta.merge(df_credits,how='left', left_on='id', right_on='id')


# In[39]:


df_merge.info()


# In[40]:


# Reference : [28]
# Merging the previous merged collective dataset with FilmTV dataset 
df_merge_r=df_merge.merge(df_filmtv,how='right', left_on='original_title', right_on='title')


# In[41]:


df_merge_r.columns


# In[42]:


df_merge_r.head(1)


# In[43]:


#References: [35]
# Showing all columns of the dataset
pd.set_option('display.max_columns', None)


# In[44]:


df_merge_r.info()


# In[45]:


df_merge_r.describe()


# In[46]:


# Saving the merged dataset into csv file
# df_merge_r.to_csv('big_datasets.csv')


# In[47]:


big_datasets.dtypes


# In[48]:


big_datasets.isnull().sum()


# In[49]:


big_datasets=big_datasets.drop(['GDP PER CAPITA Year','GDP Per Capita','homepage','tagline','belongs_to_collection',],axis=1)


# In[50]:


big_datasets=big_datasets.drop(['Unnamed: 0'],axis=1)


# ## Cleaning: 

# ### Get rid of the missing values ,data incosistencey ,and wrong or uneeded data 

# In[51]:


big_datasets.shape


# In[52]:


big_datasets.isnull().sum()


# In[53]:


# Deleting the uneeded columns or that include the highest number of missing values (NaN)
big_datasets=big_datasets.drop(['erotism','tension','effort','rhythm','humor','notes','filmtv_id','id','imdb_id','overview','original_title','poster_path','release_date','runtime','vote_average','vote_count','year_x'],axis=1)


# In[54]:


big_datasets=big_datasets.drop(['status','title_x','video'],axis=1)


# In[55]:


big_datasets=big_datasets.drop(['genres'],axis=1)


# In[56]:


# References: [36]
# Discovering the correlations between the dataset's variables
big_datasets.corr()


# In[57]:


big_datasets.isnull().sum()


# In[58]:


# Deleting (Nan) values
big_datasets.dropna(inplace=True)


# In[59]:


big_datasets.isnull().sum()


# In[60]:


big_datasets.shape


# In[61]:


big_datasets['year_y'].value_counts().sort_index()


# In[62]:


big_datasets_b=big_datasets.copy()


# In[63]:


# Saving the resulted cleaned dataset into csv file called ('big')
#big_datasets.to_csv('big.csv')


# ## Enriching and Validating:

# ### Adding extra datasets that enriches the previous collective merged and cleaned dataset

# #### Adding Oscar data to the existing dataset :

# In[64]:


OSCAR.head(2)


# In[65]:


OSCAR.shape


# In[66]:


OSCAR.dropna(inplace=True)


# In[67]:


OSCAR.shape


# In[68]:


OSCAR.film.value_counts()


# In[69]:


# Refreneces: [34]
# Calculating the count of nominations received by each film 
oscar_nominations=OSCAR.film.value_counts().sort_values(ascending=False).reset_index(name="oscar_nomination")


# In[70]:


# Refreneces: [34]
# Calculating the count of wins received by each film 
oscar_wins=OSCAR.loc[OSCAR['winner']==True].film.value_counts().sort_values(ascending=False).reset_index(name="oscar_win")


# In[71]:


oscar_nominations.head(2)


# In[72]:


oscar_wins.head(2)


# In[73]:


big_df.head(2)


# In[74]:


big_df.columns


# #### Adding GDP, Inflation, Population , Unemployement, and Internet Use data to the existing collective dataset:

# In[75]:


GDP_ALL.columns


# In[76]:


INFLATION_ALL.columns


# In[77]:


POPULATION_ALL.columns


# In[78]:


#References: [37],[38]
INTERNET.columns


# In[79]:


UNEMPLOYMENT.columns


# In[80]:


#reference : [28]
big_df=big_df.merge(GDP_ALL,how='left', left_on='year_y', right_on='Year')


# In[81]:


#reference : [28]
big_df=big_df.merge(INFLATION_ALL,how='left', left_on='year_y', right_on='year')


# In[82]:


#reference : [28]
big_df=big_df.merge(POPULATION_ALL,how='left', left_on='year_y', right_on='year')


# In[83]:


#reference : [28]
big_df=big_df.merge(INTERNET,how='left', left_on='Year', right_on='int_year')


# In[84]:


#reference : [28]
big_df=big_df.merge(UNEMPLOYMENT,how='left', left_on='Year', right_on='un_year')


# In[85]:


#reference : [28]
big_df=big_df.merge(oscar_nominations,how='left', left_on='title_y', right_on='index')


# In[86]:


#reference : [28]
big_df=big_df.merge(oscar_wins,how='left', left_on='title_y', right_on='index')


# In[87]:


big_df.drop(['index_x','index_y'],axis=1,inplace=True)


# In[88]:


#reference : [39]
big_df.oscar_nomination.fillna(0,inplace=True)


# In[89]:


#reference : [39]
big_df.oscar_win.fillna(0,inplace=True)


# In[90]:


big_df.dropna(inplace=True)


# In[91]:


big_df.drop(['Unnamed: 0','year_y','year_x','int_year'],axis=1,inplace=True)


# In[92]:


big_df.drop(['adult'],axis=1,inplace=True)


# In[93]:


big_df.drop(['un_year'],axis=1,inplace=True)


# In[94]:


#References: [35]
pd.set_option('display.max_columns', None)


# In[95]:


#References: [35]
pd.options.display.max_colwidth = 50


# In[96]:


big_df['cast'].head(2)


# In[97]:


# Deleting the data points that have zero budget
big_df=big_df[big_df['budget']!=0]


# In[98]:


# Deleting the data points that have zero revenue
big_df=big_df[big_df['revenue']!=0]


# In[99]:


big_df.head(1)


# In[100]:


#Reference: [29],[30],[31]
# Extracting the values of the attribute 'name' form the dictionary in the column 'production_companies'
big_df['prod_comp'] = big_df['production_companies'].apply(lambda x: ','.join([i['name'] for i in eval(x)]))


# In[101]:


#Reference: [29],[30],[31]
# Extracting the values of the attribute 'job' form the dictionary in the column 'crew'
big_df['crew_job'] = big_df['crew'].apply(lambda x: ','.join([i['job'] for i in eval(x)]))


# In[102]:


#Reference: [29],[30],[31]
# Extracting the values of the attribute 'name' form the dictionary in the column 'crew'
big_df['crew_name'] = big_df['crew'].apply(lambda x: ','.join([i['name'] for i in eval(x)]))


# In[103]:


#Reference: [29],[30],[31]
# Extracting the values of the attribute 'name' form the dictionary in the column 'spoken_languages'
big_df['Language'] = big_df['spoken_languages'].apply(lambda x: ','.join([i['name'] for i in eval(x)]))


# In[104]:


#Reference: [29],[30],[31]
# Extracting the values of the attribute 'name' form the dictionary in the column 'production_countries'
big_df['country'] = big_df['production_countries'].apply(lambda x: ','.join([i['name'] for i in eval(x)]))


# In[105]:


big_df.drop(['production_companies','production_countries','spoken_languages','cast','crew'],axis=1,inplace=True)


# In[106]:


big_df.drop(['Language'],axis=1,inplace=True)


# In[107]:


#References: [27]
# Convert the text separated by ',' into lists
big_df["actors"]= big_df["actors"].str.split(",", n = 10, expand = False)


# In[108]:


#References: [27]
# Convert the text separated by ',' into lists
big_df["crew_job"]= big_df["crew_job"].str.split(",", n = 10, expand = False)


# In[109]:


#References: [27]
# Convert the text separated by ',' into lists
big_df["crew_name"]= big_df["crew_name"].str.split(",", n = 10, expand = False)


# In[110]:


# Reference: [31]
# To create columns of jobs that contain the names of the employees who occupy those jobs:
# Create a new dataFrame in which each column is a job title and contains the names occupying this job for each film
df_new_name_job = pd.DataFrame(big_df.apply(lambda row: pd.Series({job: name for name, job in zip(row['crew_name'], row['crew_job'])}), axis=1))

# Concatenate the original dataFrame (big_df) with the new dataFrame:
big_df = pd.concat([big_df, df_new_name_job], axis=1)


# In[111]:


# Deleteing the columns that contains job titles that contain more than one job title
# Find column names with a comma
cols_with_comma = [col for col in big_df.columns if ',' in col]

# Drop columns with a comma from the dataFrame
big_df = big_df.drop(columns=cols_with_comma)


# In[112]:


#the same as above but consider the spaces:
# Find column names with a comma
cols_with_comma2 = [col for col in big_df.columns if ', ' in col]

# Drop columns with a comma from the DataFrame
big_df = big_df.drop(columns=cols_with_comma2)


# In[113]:


#References :[40], 
# Convert the lists to multiple columns
df_cast_expanded = big_df['actors'].apply(pd.Series)
#Rename the columns
num_names = len(df_cast_expanded.columns)
new_column_names = {i: f'actor_{i + 1}' for i in range(num_names)}
df_cast_expanded = df_cast_expanded.rename(columns=new_column_names)
# Concatenate the expanded dataFrame with the original dataFrame
big_df = pd.concat([big_df, df_cast_expanded], axis=1)
# Drop the original "actors" column if needed
big_df = big_df.drop(columns='actors')


# In[114]:


big_df.drop(['Director'],axis=1,inplace=True)


# In[115]:


big_df.drop(['crew_job','crew_name'],axis=1,inplace=True)


# In[116]:


big_df.drop(['description'],axis=1,inplace=True)


# In[117]:


#reference : [39]
# filling the NaN values with zeros to keep the shape of the dataframe to be sufficient
big_df.fillna(0,inplace=True)


# In[118]:


#References: [35]
# Displaying all rows
pd.set_option("display.max_rows", None)


# In[119]:


#References: [35]
# Adjusting the display width  
pd.set_option('display.width',20)


# In[120]:


#References: [35]
#Adjusting the column width
pd.options.display.max_colwidth = 50


# In[121]:


big_df.shape


# #### Applying inflation on budget and revenue (equivalent to 2022 values):

# In[122]:


# References: [41]
# Dividing inflation value for each year by the nflation value of the year 2022
INFLATION_ALL['inf_mul'] = INFLATION_ALL['inflation'].iloc[-1] / INFLATION_ALL['inflation']


# In[123]:


inf_mul=INFLATION_ALL.drop('inflation',axis=1)


# In[124]:


#reference : [28]
big_df=big_df.merge(inf_mul,how='left', left_on='Year', right_on='year')


# In[125]:


big_df.drop('year',axis=1,inplace=True)


# In[126]:


# Multiplying each revenue by the inflation multiplier to create a new column 'revenue_adj' 
big_df['revenue_adj']=big_df['revenue']*big_df['inf_mul']


# In[127]:


# Multiplying each budget by the inflation multiplier to create a new column 'budget_adj'
big_df['budget_adj']=big_df['budget']*big_df['inf_mul']


# In[128]:


big_df.drop(['budget','revenue'],axis=1,inplace=True)


# ### Calculating the influence of the indiviuals in the cast of the film on the revenue and adding them to the existing collective dataset: 

# #### Directors power:

# In[129]:


#References :[32],[34]
big_df.groupby('directors')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")


# In[130]:


#references : [28],[32],[34],
#big_df.groupby('directors')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")
director_df=big_df.groupby('directors')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")
director_df['directors_power'] = director_df['direct_total_rev'].rank(ascending=True)
big_df=big_df.merge(director_df,how='left', left_on='directors', right_on='directors')
big_df.drop('direct_total_rev',axis=1,inplace=True)


# ### Writer power:

# In[131]:


#references : [28],[32],[34],[39]
writer_df=big_df.groupby('Writer')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="writer_total_rev")
writer_df=writer_df.loc[writer_df['Writer']!=0]
writer_df['writer_power'] = writer_df['writer_total_rev'].rank(ascending=True)
big_df=big_df.merge(writer_df,how='left', left_on='Writer', right_on='Writer')
big_df.drop('writer_total_rev',axis=1,inplace=True)
big_df['writer_power'].fillna(0,inplace=True)


# ### Actors powers:

# In[132]:


#references : [28],[32],[34],
#big_df.groupby('directors')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")
actor1_df=big_df.groupby('actor_1')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="actor1_total_rev")
actor1_df['actor1_power'] = actor1_df['actor1_total_rev'].rank(ascending=True)
big_df=big_df.merge(actor1_df,how='left', left_on='actor_1', right_on='actor_1')
big_df.drop('actor1_total_rev',axis=1,inplace=True)


# In[133]:


#references : [28],[32],[34],
#big_df.groupby('directors')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")
actor2_df=big_df.groupby('actor_2')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="actor2_total_rev")
actor2_df['actor2_power'] = actor2_df['actor2_total_rev'].rank(ascending=True)
big_df=big_df.merge(actor2_df,how='left', left_on='actor_2', right_on='actor_2')
big_df.drop('actor2_total_rev',axis=1,inplace=True)


# In[134]:


#references : [28],[32],[34],
#actor_8:
#big_df.groupby('directors')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")
actor8_df=big_df.groupby('actor_8')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="actor8_total_rev")
actor8_df=actor8_df.loc[actor8_df['actor_8']!=0]
actor8_df['actor8_power'] = actor8_df['actor8_total_rev'].rank(ascending=True)
big_df=big_df.merge(actor8_df,how='left', left_on='actor_8', right_on='actor_8')
big_df.drop('actor8_total_rev',axis=1,inplace=True)
big_df['actor8_power'].fillna(0,inplace=True)


# In[135]:


#references : [28],[32],[34],[39]
#actor_5:
#big_df.groupby('directors')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")
actor5_df=big_df.groupby('actor_5')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="actor5_total_rev")
actor5_df=actor5_df.loc[actor5_df['actor_5']!=0]
actor5_df['actor5_power'] = actor5_df['actor5_total_rev'].rank(ascending=True)
big_df=big_df.merge(actor5_df,how='left', left_on='actor_5', right_on='actor_5')
big_df.drop('actor5_total_rev',axis=1,inplace=True)
big_df['actor5_power'].fillna(0,inplace=True)


# In[136]:


#references : [28],[32],[34],[39]
#actor_11:
#big_df.groupby('directors')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")
actor11_df=big_df.groupby('actor_11')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="actor11_total_rev")
actor11_df=actor11_df.loc[actor11_df['actor_11']!=0]
actor11_df['actor11_power'] = actor11_df['actor11_total_rev'].rank(ascending=True)
big_df=big_df.merge(actor11_df,how='left', left_on='actor_11', right_on='actor_11')
big_df.drop('actor11_total_rev',axis=1,inplace=True)
big_df['actor11_power'].fillna(0,inplace=True)


# ### Characters makers:

# In[137]:


#references : [28],[32],[34],[39]
#characters:
#big_df.groupby('Characters')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")
Characters_df=big_df.groupby('Characters')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="Characters_total_rev")
Characters_df=Characters_df.loc[Characters_df['Characters']!=0]
Characters_df['Characters_power'] = Characters_df['Characters_total_rev'].rank(ascending=True)
big_df=big_df.merge(Characters_df,how='left', left_on='Characters', right_on='Characters')
big_df.drop('Characters_total_rev',axis=1,inplace=True)
big_df['Characters_power'].fillna(0,inplace=True)


# ### Comic Power:

# In[138]:


#references : [28],[32],[34],[39]
#Comic Book:
#big_df.groupby('directors')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="direct_total_rev")
comic_df=big_df.groupby('Comic Book')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="Comic Book_total_rev")
comic_df=comic_df.loc[comic_df['Comic Book']!=0]
comic_df['Comic Book_power'] = comic_df['Comic Book_total_rev'].rank(ascending=True)
big_df=big_df.merge(comic_df,how='left', left_on='Comic Book', right_on='Comic Book')
big_df.drop('Comic Book_total_rev',axis=1,inplace=True)
big_df['Comic Book_power'].fillna(0,inplace=True)


# In[139]:


big_df.shape


# In[140]:


big_df.isnull().sum()


# In[141]:


#Coverting month and year to strings to get insights related to them
big_df['month']=change_type_to(big_df,'month',str)
big_df['day']=change_type_to(big_df,'day',int)
big_df['Year']=change_type_to(big_df,'Year',str)


# In[142]:


#Saving the dataset
#big_df.to_csv('big2.csv')


# <a id='Exploratory analysis'></a>
# # Exploratory analysis:

# ## Finding insights from the resulting collective dataset

# ### Genres:

# In[143]:


# References: [32],[34]
# Top 5 Genre that produce the highest revenue:
big_df.groupby('genre')['revenue_adj'].sum().sort_values(ascending=False).head(5)
top_genre=big_df.groupby('genre')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="revenue of genre")
top_genre


# The genre that yields the highest revenue is found to be 'Fantasy' genre.

# In[144]:


# References [42]
sns.barplot(x='revenue of genre',y='genre', data=top_genre)
plt.title("Sum of revenue generated by each genre")
plt.show()


# In[145]:


# References :[50]
# the genres corresponding to the revenue each year:
sns.lmplot(y='Year', x='revenue_adj', data=big_df, fit_reg=False, hue='genre', height=5, scatter_kws={"s": 50})
plt.show()


# It is noticed that the genres that get the highest revenue are changed starting from 2020 which may be due to the COVID-19 pandemic.

# ### Year: 

# In[146]:


# References: [32],[34]
# years that yielded highest revenues:
top_year=big_df.groupby('Year')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="revenue of year")
top_year


# It is found that in 2015 ,the highest revenue was generated by films.While the pandemic years (2020,2021,and 2022) are among the lowest.

# In[147]:


# References [42]
plt.figure(figsize=(6, 6))
sns.barplot(x='revenue of year',y='Year', data=top_year)
plt.title("Sum of revenue generated in each year")
plt.show()


# ### Month:

# In[148]:


# References: [32],[34]
# Months that yielded highest revenues
top_month=big_df.groupby('month')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="revenue of month")
top_month


# It is noticed that December ,May ,and November are the months in which the highest revenues are generated by the films.Those months indicates how seasonality affects the revenue like Christmas in December.

# In[149]:


# References [42]
plt.figure(figsize=(5, 5))
sns.barplot(x='revenue of month',y='month', data=top_month)
plt.title("Sum of revenue generated in each month")
plt.show()


# In[150]:


big_df.head(1)


# In[151]:


# References: [32],[34] 
# languages that yielded the highest revenues
top_lang=big_df.groupby('original_language')['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="revenue of language")
top_lang


# In[152]:


# References [42],[43]
plt.figure(figsize=(8, 5))
sns.barplot(x='revenue of language', y='original_language', data=top_lang, orient='h')
# Using log scale to the x axis to be able to visualize the data:
plt.xscale("log")  
plt.title("Sum of revenue generated by each language")
plt.xlabel("Revenue")
plt.ylabel("Original Language")
plt.tight_layout()
plt.show()


# In[153]:


#references: [44]
# Investigating the distributions of the data variables
big_df.hist(figsize=(20,20))
pyplot.show()


# In[154]:


# reference : [68]
# Using density plot to further investigation
big_df.plot(kind='kde', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
pyplot.show()


# In[155]:


big_df.head(1)


# In[156]:


#Distribution of the target variable ("revenue_adj"):
sns.histplot(data=big_df, x="revenue_adj")


# From the above distribution for revenue ,which is the target variable , it is found to be skewed

# In[157]:


# References:[45],[46],[47],[48]
# Create the histogram and density plot  

plt.figure(figsize=(10, 6))  
sns.histplot(data=big_df, x="revenue_adj", bins=20, kde=True)
plt.xlabel("Revenue Adjusted")
plt.ylabel("Frequency")
plt.title("Distribution of Revenue (Adjusted)")
plt.axvline(x=big_df["revenue_adj"].mean(), color='red', linestyle='dashed', linewidth=2, label="Mean")
plt.legend()
# adjusting the spacing: 
plt.tight_layout()  
plt.show()


# In[158]:


#References:[36],[49]
correlation_matrix = big_df.corr()
correlation_matrix


# In[159]:


#References : [49]
plt.figure(figsize=(10,5))
plt.title('Heatmap for correlation between data attributes')
sns.heatmap(correlation_matrix, annot=False)
plt.show()


# Some variables are highly correlated like budget and revenue of the films

# In[160]:


#References : [50]
sns.lmplot(x="budget_adj", y="revenue_adj", data=big_df)


# The relation between the film's budegt and revenue is high postive correlation

# In[161]:


big_df.head(1)


# In[162]:


#References : [50]
sns.lmplot(x="oscar_win", y="revenue_adj", data=big_df)


# Also the win of Oscar prize correlates positively with the revenue

# In[163]:


#References : [50]
sns.lmplot(x="inflation", y="revenue_adj", data=big_df)


# Revenue and inflation are negatively correlated.

# In[164]:


big_df.head(1)


# In[165]:


#References :[32],[34]
big_df.groupby("Producer")['revenue_adj'].sum().sort_values(ascending=False).head(11)


# In[166]:


#References: [32],[34]
top10producers=big_df.groupby("Producer")['revenue_adj'].sum().sort_values(ascending=False).reset_index(name="revenue of producers").head(11)
top10producers=top10producers[top10producers["Producer"]!=0]
top10producers


# In[167]:


#References:[51]
plt.figure(figsize=(7, 7))
plt.pie(top10producers["revenue of producers"], labels=top10producers["Producer"], autopct="%1.1f%%")
plt.title("Producres whose films achived the highest revenues")
plt.show()


# In[168]:


#References:[51]
#actor 1
plt.figure(figsize=(7, 7))
top10actor1=actor1_df.head(10)
plt.pie(top10actor1['actor1_total_rev'], labels=top10actor1['actor_1'], autopct="%1.1f%%")
plt.title("Top 10 Main actors whose films achieved the highest revenues")
plt.show()


# In[169]:


#References:[51]
#director
top10director=director_df.head(10)
plt.figure(figsize=(7, 7))
plt.pie(top10director['direct_total_rev'], labels=top10director['directors'], autopct="%1.1f%%")
plt.title("Top 10 Directors whose films achieved the highets revenues.")
plt.show()


# <a id='Preparing and applying the machine learning models'></a>
# # Preparing and applying the machine learning models : 

# In[170]:


#identifying inputs and target of the model all features:
# big df
x=big_df.drop('revenue_adj',axis=1)
y=big_df[['revenue_adj']]


# In[171]:


x.head(1)


# ### Encoding the categorical features:

# In[172]:


encoded_df_x=labelencoding(x)


# In[173]:


encoded_df_x.head(2)


# <a id="Regression 'value estimation'"></a>
# ## Regression 'value estimation':

# #### Feature Selection using f_regression:

# In[174]:


select_top22_regression_features (x,encoded_df_x,y)


# #### Choosing either all features or pre-release features:

# In[175]:


# Choosing whether to use all the features or only the 'prerelease features':

user=input("If you want to use all features write 'all' or write 'pre' for pre-release features : ").lower()
while user not in ['all','pre']:
    user=input("Please enter a valid answer!")
        
if user=='all':
    #remove actor 11 and put oscar win , also good results of 89.4 % 
    # big df
    print ('all')
    x_regression=big_df[['budget_adj','popularity', 'actor8_power', 'actor5_power','actor11_power','directors_power', 'actor2_power', 'Characters_power','actor1_power','total_votes', 'inf_mul','duration', 'inflation', 'population','int_use','Year','Characters','oscar_nomination','Grip','GDP','oscar_win']]
elif user=='pre':
    #pre release features# big df
    print('pre')
    x_regression=big_df[['budget_adj', 'actor8_power', 'actor5_power','actor11_power','directors_power', 'actor2_power', 'Characters_power','actor1_power', 'inf_mul','duration', 'inflation', 'population','int_use','Year','Characters','Grip','GDP']]


# In[176]:


x_regression.head(2)


# ### Encode the filtered selected features:

# In[177]:


encoded_df_x2=labelencoding(x_regression)


# In[178]:


encoded_df_x2.head(2)


# ### Train-Test Splitting:

# In[179]:


#big_df 90% training
x_enc_train, x_enc_test, y_enc_train, y_enc_test = train_test_split(encoded_df_x2,y, test_size=0.1, random_state=0)


# In[180]:


#big_df 80% training
x_enc_train2, x_enc_test2, y_enc_train2, y_enc_test2 = train_test_split(encoded_df_x2,y, test_size=0.2, random_state=0)


# In[181]:


#big_df 70% training
x_enc_train3, x_enc_test3, y_enc_train3, y_enc_test3 = train_test_split(encoded_df_x2,y, test_size=0.3, random_state=0)


# In[182]:


x_enc_train3.shape ,x_enc_train2.shape ,x_enc_train.shape


# In[183]:


x_enc_test3.shape,x_enc_test2.shape,x_enc_test.shape


# In[184]:


x_enc_train3.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("70% Training",fontsize=60)
pyplot.show()


x_enc_train2.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("80% Training",fontsize=60)
pyplot.show()

x_enc_train.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("90% Training",fontsize=60)
pyplot.show()


# In[185]:


x_enc_test3.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("30% Testing",fontsize=60)
pyplot.show()

x_enc_test2.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("20% Testing",fontsize=60)
pyplot.show()


x_enc_test.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("10% Testing",fontsize=60)
pyplot.show()


# In[186]:


y_enc_train3.shape,y_enc_train2.shape,y_enc_train.shape


# In[187]:


# Distributions of Training sets for the target variable
y_enc_train3.plot(kind='box')

y_enc_train2.plot(kind='box')

y_enc_train.plot(kind='box')


# In[188]:


# Distributions of Testing sets for the target variable

y_enc_test3.plot(kind='box')

y_enc_test2.plot(kind='box')

y_enc_test.plot(kind='box')


# ### Transform the features and target:

# In[189]:


#scale train x and train y
# sacle x :
x_enc_s=scale(x_enc_train, x_enc_test)
x_enc_s_train=x_enc_s[0]
x_enc_s_test=x_enc_s[1]
# sacle y :
y_enc_s=scale(y_enc_train, y_enc_test)
y_enc_s_train=y_enc_s[0]
y_enc_s_test=y_enc_s[1]


# sacle x :
x_enc_s2=scale(x_enc_train2, x_enc_test2)
x_enc_s_train2=x_enc_s2[0]
x_enc_s_test2=x_enc_s2[1]
# sacle y :
y_enc_s2=scale(y_enc_train2, y_enc_test2)
y_enc_s_train2=y_enc_s2[0]
y_enc_s_test2=y_enc_s2[1]


# sacle x :
x_enc_s3=scale(x_enc_train3, x_enc_test3)
x_enc_s_train3=x_enc_s3[0]
x_enc_s_test3=x_enc_s3[1]
# sacle y :
y_enc_s3=scale(y_enc_train3, y_enc_test3)
y_enc_s_train3=y_enc_s3[0]
y_enc_s_test3=y_enc_s3[1]


# In[190]:


# references : [55]
# Scale as a whole data by (StandardScaler) to be used for k fold cross validation
#Scale X:
scaler = StandardScaler().fit(encoded_df_x2)
rescaledX = scaler.fit_transform(encoded_df_x2)
#Scale Y:
scalerY = StandardScaler().fit(y)
rescaledY = scalerY.fit_transform(y)


# In[191]:


encoded_df_x2.head(1)


# # Applying regression "value estimate" machine learning models: 

# In[192]:


# Linear regression: 90% training
linreg(x_enc_s_train, x_enc_s_test,y_enc_s_train,y_enc_s_test,x_enc_train)


# In[193]:


# Linear regression: 80% training
linreg(x_enc_s_train2, x_enc_s_test2,y_enc_s_train2,y_enc_s_test2,x_enc_train2)


# In[194]:


# Linear regression: 70% training
linreg(x_enc_s_train3, x_enc_s_test3,y_enc_s_train3,y_enc_s_test3,x_enc_train3)


# In[195]:


# LightGBM model:90% training
light (x_enc_s_train, x_enc_s_test,y_enc_s_train,y_enc_s_test,rescaledX,rescaledY)


# In[196]:


# LightGBM model:80% training
light (x_enc_s_train2, x_enc_s_test2,y_enc_s_train2,y_enc_s_test2,rescaledX,rescaledY)


# In[197]:


# LightGBM model:70% training
light (x_enc_s_train3, x_enc_s_test3,y_enc_s_train3,y_enc_s_test3,rescaledX,rescaledY)


# In[198]:


# XGBoost regression model:90% training
xgboostmodel (x_enc_s_train,y_enc_s_train, x_enc_s_test,y_enc_s_test,rescaledX,rescaledY)


# In[199]:


# XGBoost regression model:80% training
xgboostmodel (x_enc_s_train2,y_enc_s_train2, x_enc_s_test2,y_enc_s_test2,rescaledX,rescaledY)


# In[200]:


# XGBoost regression model:70% training
xgboostmodel (x_enc_s_train3,y_enc_s_train3, x_enc_s_test3,y_enc_s_test3,rescaledX,rescaledY)


# In[201]:


# Catboost regression model:90%training
catboostmodel (x_enc_s_train,y_enc_s_train, x_enc_s_test,y_enc_s_test,rescaledX,rescaledY)


# In[202]:


# Catboost regression model:80%training
catboostmodel (x_enc_s_train2,y_enc_s_train2, x_enc_s_test2,y_enc_s_test2,rescaledX,rescaledY)


# In[203]:


# Catboost regression model:70%training
catboostmodel (x_enc_s_train3,y_enc_s_train3, x_enc_s_test3,y_enc_s_test3,rescaledX,rescaledY)


# <a id="Binary Classification"></a>
# ## Binary Classification: 

# In[204]:


encoded_df_x.head(2)


# In[205]:


y.median()


# In[206]:


y.mean()


# In[207]:


y.describe()


# In[208]:


y.boxplot()


# In[209]:


# Binarise revenue to high or low revenue
# reference : [53]
#converting the 'revenue_adj' column into two classes ,High revenue (value above the mean) and Low revenue (below the mean)
binarizer = Binarizer(threshold=276904200).fit(y)
y_bin= binarizer.transform(y)
set_printoptions(precision=3)


# ### Feature selection using chi2:

# In[210]:


select_top21_binary_class_features (x,encoded_df_x,y_bin)


# In[211]:


# Choosing whether to use all the features or only the 'prerelease features':

user2=input("If you want to use all features write 'all' or write 'pre' for pre-release features : ").lower()
while user2 not in ['all','pre']:
    user2=input("Please enter a valid answer!")
        
if user2=='all':
    #using above features
    # big df
    x_bin_class=big_df[['GDP','budget_adj','population', 'actor5_power','actor8_power','actor2_power', 'directors_power', 'actor11_power','actor1_power', 'total_votes', 'prod_comp','actor_6','actor_5','Characters_power','actor_8','actor_9','writer_power','actor_7','actor_10','popularity','Writer']]
elif user2=='pre':
    #using only pre released features from above features
    # big df
    x_bin_class=big_df[['GDP','budget_adj','population', 'actor5_power','actor8_power', 'directors_power','actor2_power', 'actor11_power','actor1_power', 'prod_comp','writer_power','actor_5', 'Characters_power', 'actor_8','actor_9','actor_6','Writer','actor_10','actor_7']]


# In[212]:


x_bin_class.head(2)


# ### Encode the selected features: 

# In[213]:


encoded_df_x3=labelencoding(x_bin_class)


# In[214]:


encoded_df_x3.head(2)


# ### Train-Test Splitting:

# In[215]:


#big_df x label ,y bin
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(encoded_df_x3,y_bin, test_size=0.1, random_state=42)


# In[216]:


#big_df x label ,y bin 80% training
X_train_bin2, X_test_bin2, y_train_bin2, y_test_bin2 = train_test_split(encoded_df_x3,y_bin, test_size=0.2, random_state=42)


# In[217]:


#big_df x label ,y bin 70% training 
X_train_bin3, X_test_bin3, y_train_bin3, y_test_bin3 = train_test_split(encoded_df_x3,y_bin, test_size=0.3, random_state=42)


# In[218]:


X_train_bin3.shape,X_train_bin2.shape,X_train_bin.shape


# In[219]:


X_test_bin3.shape,X_test_bin2.shape,X_test_bin.shape


# In[220]:


# Features disributions of trainiing sets with different splits:

X_train_bin3.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("70% Training",fontsize=60)
pyplot.show()

X_train_bin2.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("80% Training",fontsize=60)
pyplot.show()


X_train_bin.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("90% Training",fontsize=60)
pyplot.show()


# In[221]:


# Features disributions of testing sets with different splits:

X_test_bin3.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("30% Testing",fontsize=60)
pyplot.show()

X_test_bin2.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("20% Testing",fontsize=60)
pyplot.show()

X_test_bin.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("10% Testing",fontsize=60)
pyplot.show()


# In[222]:


y_train_bin3.shape,y_train_bin2.shape,y_train_bin.shape


# In[223]:


y_test_bin3.shape,y_test_bin2.shape,y_test_bin.shape


# In[224]:


X_train_bin.head(5)


# In[225]:


X_test_bin.head(5)


# In[226]:


y_test_bin


# ### Transform the features and target:

# In[227]:


# sacle x bin :
x_enc_s33=scale(X_train_bin, X_test_bin)
X_train_bin_scaled=x_enc_s33[0]
X_test_bin_scaled=x_enc_s33[1]


x_enc_s332=scale(X_train_bin2, X_test_bin2)
X_train_bin_scaled2=x_enc_s332[0]
X_test_bin_scaled2=x_enc_s332[1]



x_enc_s333=scale(X_train_bin3, X_test_bin3)
X_train_bin_scaled3=x_enc_s333[0]
X_test_bin_scaled3=x_enc_s333[1]


# In[228]:


# References : [55]
# Scale X as a whole: (Standard,Standard Scaler )
scaler_bin = StandardScaler().fit(encoded_df_x3)
rescaledX_bin = scaler.fit_transform(encoded_df_x3)


# <a id="Applying binary classification machine learning models"></a>
# ### Applying binary classification machine learning models: 

# In[229]:


# Logistic regression model:90% training
LogR (X_train_bin_scaled, y_train_bin,X_test_bin_scaled,y_test_bin,rescaledX_bin,y_bin)


# In[230]:


# Logistic regression model:80% training
LogR (X_train_bin_scaled2, y_train_bin2,X_test_bin_scaled2,y_test_bin2,rescaledX_bin,y_bin)


# In[231]:


# Logistic regression model:70% training 
LogR (X_train_bin_scaled3, y_train_bin3,X_test_bin_scaled3,y_test_bin3,rescaledX_bin,y_bin)


# In[232]:


# Decision tree model:90% training
DT_bin (X_train_bin_scaled, y_train_bin,X_test_bin_scaled,y_test_bin,rescaledX_bin,y_bin)


# In[233]:


# Decision tree model:80% training 
DT_bin (X_train_bin_scaled2, y_train_bin2,X_test_bin_scaled2,y_test_bin2,rescaledX_bin,y_bin)


# In[234]:


# Decision tree model:70% training 
DT_bin (X_train_bin_scaled3, y_train_bin3,X_test_bin_scaled3,y_test_bin3,rescaledX_bin,y_bin)


# In[235]:


# Random forest model:90% training
RF_bin (X_train_bin_scaled, y_train_bin,X_test_bin_scaled,y_test_bin,rescaledX_bin,y_bin)


# In[236]:


# Random forest model:80% training
RF_bin (X_train_bin_scaled2, y_train_bin2,X_test_bin_scaled2,y_test_bin2,rescaledX_bin,y_bin)


# In[237]:


# Random forest model:70% training 
RF_bin (X_train_bin_scaled3, y_train_bin3,X_test_bin_scaled3,y_test_bin3,rescaledX_bin,y_bin)


# In[238]:


# Extreme gradient boosting model (XGBoost):90% training
xgb_bin (X_train_bin_scaled, y_train_bin,X_test_bin_scaled,y_test_bin,rescaledX_bin,y_bin)


# In[239]:


# Extreme gradient boosting model (XGBoost):80% training
xgb_bin (X_train_bin_scaled2, y_train_bin2,X_test_bin_scaled2,y_test_bin2,rescaledX_bin,y_bin)


# In[240]:


# Extreme gradient boosting model (XGBoost): 70% training 
xgb_bin (X_train_bin_scaled3, y_train_bin3,X_test_bin_scaled3,y_test_bin3,rescaledX_bin,y_bin)


# <a id="Multi Classification"></a>
# ## Multi classification:

# In[241]:


# This is to arrange the y to be 1D array for the multi classes preparation
y=big_df['revenue_adj']


# In[242]:


# References: [67] 
# Divide the target variable into 6 classes:  

y_6classes, bins = pd.qcut(y, q=6, labels=False, retbins=True)


# In[243]:


# Ranges for each class
print("Class Ranges:")
for i, bin_edge in enumerate(bins):
    if i < len(bins) - 1:
        print(f"Class {i}: {bin_edge} - {bins[i + 1]}")


# In[244]:


select_top21_binary_class_features (x,encoded_df_x,y_6classes)


# In[245]:


# Choosing whether to use all the features or only the 'prerelease features':

user2=input("If you want to use all features write 'all' or write 'pre' for pre-release features : ").lower()
while user2 not in ['all','pre']:
    user2=input("Please enter a valid answer!")
        
if user2=='all':
    #use above features:
    x_multi=big_df[['GDP','budget_adj','population', 
              'actor5_power','actor8_power', 
              'directors_power','actor2_power', 
              'actor1_power', 'actor11_power', 'total_votes', 
              'prod_comp','Writer','actor_6','writer_power', 
              'Producer','actor_5','actor_8','Screenplay','actor_7', 
              'Executive Producer','Characters_power']]

    
elif user2=='pre':
    #pre release
    #use above features:
    x_multi=big_df[['GDP','budget_adj','population', 
              'actor5_power','actor8_power', 
              'directors_power','actor2_power', 
              'actor1_power', 'actor11_power', 
              'prod_comp','Writer','actor_6','writer_power', 
              'Producer','actor_5','actor_8', 
              'Executive Producer','Characters_power','actor_7',
              'Screenplay']]


# In[246]:


x_multi.head(2)


# ### Encode the selected features:

# In[247]:


encoded_df_x4=labelencoding(x_multi)


# In[ ]:





# ### Train-Test Splitting:

# In[248]:


#6 classes 90% training
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(encoded_df_x4,y_6classes, test_size=0.1, random_state=42)


# In[249]:


#6 classes 80% training
X_train_multi2, X_test_multi2, y_train_multi2, y_test_multi2 = train_test_split(encoded_df_x4,y_6classes, test_size=0.2, random_state=42)


# In[250]:


#6 classes 70% training
X_train_multi3, X_test_multi3, y_train_multi3, y_test_multi3 = train_test_split(encoded_df_x4,y_6classes, test_size=0.3, random_state=42)


# In[251]:


X_train_multi3.shape,X_train_multi2.shape,X_train_multi.shape


# In[252]:


X_test_multi3.shape,X_test_multi2.shape,X_test_multi.shape


# In[253]:


# Features disributions of trainiing and testing sets for 70% training and 30% Testing:

X_train_multi3.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("70% Training set",fontsize=60)
pyplot.show()

X_test_multi3.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("30% Testing set",fontsize=60)
pyplot.show()


# In[254]:


# Features disributions of trainiing and testing sets for 80% training and 20% Testing:
X_train_multi2.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("80% Training set",fontsize=60)
pyplot.show()

X_test_multi2.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("20% Testing set",fontsize=60)
pyplot.show()


# In[255]:


# Features disributions of trainiing and testing sets for 90% training and 10% Testing:
X_train_multi.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("90% Training set",fontsize=60)
pyplot.show()

X_test_multi.plot(kind='box', subplots=True, layout=(6,6), sharex=False,figsize=(20,20))
plt.suptitle("10% Testing set",fontsize=60)
pyplot.show()


# In[256]:


# Target variableble balance in training set
y_train_multi3.value_counts(),y_train_multi2.value_counts(),y_train_multi.value_counts()


# In[257]:


# Target varaible balance in the testing set:
y_test_multi3.value_counts(),y_test_multi2.value_counts(),y_test_multi.value_counts()


# ### Transform the features and target:

# In[258]:


x_enc_s4=scale(X_train_multi, X_test_multi)
X_train_multi_scaled=x_enc_s4[0]
X_test_multi_scaled=x_enc_s4[1]


x_enc_s42=scale(X_train_multi2, X_test_multi2)
X_train_multi_scaled2=x_enc_s42[0]
X_test_multi_scaled2=x_enc_s42[1]



x_enc_s43=scale(X_train_multi3, X_test_multi3)
X_train_multi_scaled3=x_enc_s43[0]
X_test_multi_scaled3=x_enc_s43[1]


# In[259]:


#Scale X as a whole: (Standard,Standard Scaler )
scaler_multi = StandardScaler().fit(encoded_df_x4)
rescaledX_multi = scaler.fit_transform(encoded_df_x4)


# 
# ### Applying multi classification machine learning models: 

# In[260]:


# Logistic regression multi classification:90% training
LogR_multi (X_train_multi_scaled, y_train_multi,X_test_multi_scaled,y_test_multi,rescaledX_multi,y_6classes) 


# In[261]:


# Logistic regression multi classification:80% training
LogR_multi (X_train_multi_scaled2, y_train_multi2,X_test_multi_scaled2,y_test_multi2,rescaledX_multi,y_6classes) 


# In[262]:


# Logistic regression multi classification:70% training
LogR_multi (X_train_multi_scaled3, y_train_multi3,X_test_multi_scaled3,y_test_multi3,rescaledX_multi,y_6classes) 


# In[263]:


# Decision tree multi classification 90% training
DT_multi (X_train_multi_scaled, y_train_multi,X_test_multi_scaled,y_test_multi,rescaledX_multi,y_6classes)


# In[264]:


# Decision tree multi classification 80% training
DT_multi (X_train_multi_scaled2, y_train_multi2,X_test_multi_scaled2,y_test_multi2,rescaledX_multi,y_6classes)


# In[265]:


# Decision tree multi classification  70% training
DT_multi (X_train_multi_scaled3, y_train_multi3,X_test_multi_scaled3,y_test_multi3,rescaledX_multi,y_6classes)


# In[266]:


# Random forest multi classification 90
RF_multi (X_train_multi_scaled, y_train_multi,X_test_multi_scaled,y_test_multi,rescaledX_multi,y_6classes)


# In[267]:


# Random forest multi classification 80
RF_multi (X_train_multi_scaled2, y_train_multi2,X_test_multi_scaled2,y_test_multi2,rescaledX_multi,y_6classes)


# In[268]:


# Random forest multi classification 70
RF_multi (X_train_multi_scaled3, y_train_multi3,X_test_multi_scaled3,y_test_multi3,rescaledX_multi,y_6classes)


# In[269]:


#Extreme gradient boosting multi classification 90% training
xgb_multi (X_train_multi_scaled, y_train_multi,X_test_multi_scaled,y_test_multi,rescaledX_multi,y_6classes)


# In[270]:


#Extreme gradient boosting multi classification 80% training
xgb_multi (X_train_multi_scaled2, y_train_multi2,X_test_multi_scaled2,y_test_multi2,rescaledX_multi,y_6classes)


# In[271]:


#Extreme gradient boosting multi classification 70% training
xgb_multi (X_train_multi_scaled3, y_train_multi3,X_test_multi_scaled3,y_test_multi3,rescaledX_multi,y_6classes)


# <a id='References'></a>
# # References:

# [1] Stackoverflow.(2014).Label encoding across multiple columns in scikit-learn. https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn. [Accessed 26th July 2023].
# 
# [2] A.Provetti.(2023).DSTA_5b_sklearn_classification [colab notebook]. https://colab.research.google.com/drive/1-SAXFfR6ZFtm9tWAsIhNS9M85N0WIFl0#scrollTo=xUXWF_VzNwYh&uniqifier=14. [Accessed 26/07/2023].
# 
# [3] Kaggle.(2023).Simple Linear Regression - plotly visualization.https://www.kaggle.com/code/sandhyakrishnan02/simple-linear-regression-plotly-visualization?scriptVersionId=133830932. [Accessed 26/07/2023].
# 
# [4] Iotespresso.(2021). K-fold cross-validation in Scikit Learn . https://iotespresso.com/k-fold-cross-validation-in-scikit-learn/. [Accessed 26/07/2023].
# 
# [5] Towardsdatascience.(2017).Random Forest in Python. https://towardsdatascience.com/random-forest-in-python-24d0893d51c0.[Accessed 26/07/2023].  
# 
# [6] Machinelearningmastery.(2021).How to Develop a Random Forest Ensemble in Python. https://machinelearningmastery.com/random-forest-ensemble-in-python/. [Accessed 26/07/2023].
# 
# [7] Towardsdatascience.(2018). Hyperparameter Tuning the Random Forest in Python. https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74. [Accessed 26/07/2023]
# 
# [8] Medium.(2020) .How to use Machine Learning Approach to Predict Movie Box-Office Revenue / Success?. https://medium.com/analytics-vidhya/how-to-use-machine-learning-approach-to-predict-movie-box-office-revenue-success-e2e688669972. [Accessed 26/07/2023].
# 
# [9] Geeksforgeeks .(2023) .Support Vector Regression (SVR) using Linear and Non-Linear Kernels in Scikit Learn . https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/. [Accessed 26/07/2023].
# 
# [10] Section.io (2022) . Getting Started with Support Vector Regression in Python .https://www.section.io/engineering-education/support-vector-regression-in-python/#:~:text=the%20kernel%20functions.-,Implementing%20SVR,Basis%20function%20(RBF)%20kernel.&text=As%20we%20can%20see%2C%20the,for%20the%20scaled%20study%20variable. [Accessed 26/07/2023].
# 
# [11] Analyticsvidhya .(2020). Support Vector Regression Tutorial for Machine Learning . https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/#:~:text=Overall%2C%20SVM%20regression%20is%20a,domains%20for%20making%20accurate%20predictions. [Accessed 26/07/2023].
# 
# [12] Michael fuchs .(2021).NN - Multi-layer Perceptron Regressor (MLPRegressor). https://michael-fuchs-python.netlify.app/2021/02/10/nn-multi-layer-perceptron-regressor-mlpregressor/. [Accessed 26/07/2023].
# 
# [13] Github,adrinjalali.test_mlp.py .(2016) . https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neural_network/tests/test_mlp.py. [Accessed 26/07/2023].
# 
# [14] neptune.ai. (2023) . Understanding LightGBM Parameters (and How to Tune Them). https://neptune.ai/blog/lightgbm-parameters-guide. [Accessed 26/07/2023].
# 
# [15] kaggle . (2020) . Easy to implement Sentiment Analysis .https://www.kaggle.com/code/vinaypratap/easy-to-implement-sentiment-analysis. [Accessed 26/07/2023].
# 
# [16] scikit-learn .(2023).sklearn.neighbors.KNeighborsRegressor. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html. [Accessed 26/07/2023].
# 
# [17] scikit-learn.(2023). sklearn.metrics.r2_score . https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html. [Accessed 26/07/2023].
# 
# [18] Machine learning mastery . How to Develop Ridge Regression Models in Python .https://machinelearningmastery.com/ridge-regression-with-python/. [Accessed 26/07/2023].
# 
# [19] Kaggle. (2018) . Using XGBoost with Scikit-learn. https://www.kaggle.com/code/stuarthallows/using-xgboost-with-scikit-learn . [Accessed 02/09/2023]. 
# 
# [20] educative.io .(2023) . Regression using XGBoost in Python . https://www.educative.io/answers/regression-using-xgboost-in-python. [Accessed 02/09/2023]. 
# 
# [21] Analytics Vidhya.(2022). A Brief Introduction To Yandex-Catboost Regressor. https://www.analyticsvidhya.com/blog/2021/12/a-brief-introduction-to-yandex-catboost-regressor/. [Accessed 03/09/2023].
# 
# [22] scikit-learn.(2023).sklearn.metrics.mean_squared_error. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html .[Accessed 03/09/2023].
# 
# [23] Vijay Choubey .(2020).How to evaluate the performance of a machine learning model.https://vijay-choubey.medium.com/how-to-evaluate-the-performance-of-a-machine-learning-model-d12ce920c365. [Accessed 03/09/2023].
# 
# [24] scikit-learn .(2023) . sklearn.metrics.mean_absolute_error.https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html. [Accessed 03/09/2023].
# 
# [25] pandas.pydata.(2023).pandas.read_csv.https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html. [Accessed 04/09/2023].
# 
# [26] pandas.pydata.(2023).pandas.DataFrame.astype.https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html. [Accessed 04/09/2023].
# 
# [27] pandas.pydata.(2023).pandas.Series.str.split.https://pandas.pydata.org/docs/reference/api/pandas.Series.str.split.html. [Accessed 04/09/2023]. 
# 
# [28] Pandas.DataFrame.merge,available online at https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html.
# [Accessed 04/09/2023].
# 
# [29] Real Python . Python eval(): Evaluate Expressions Dynamically.https://realpython.com/python-eval-function/ . [Accessed 04/09/2023].
# 
# [30] geeksforgeeks.(2023).eval in Python.https://www.geeksforgeeks.org/eval-in-python/ . [Accessed 04/09/2023].
# 
# [31] sparkbyexamples.Pandas apply() with Lambda Examples.https://sparkbyexamples.com/pandas/pandas-apply-with-lambda-examples/. [Accessed 04/09/2023].
# 
# [32] https://stackoverflow.com/questions/29836477/pandas-create-new-column-with-count-from-groupby
# 
# [33] https://www.researchgate.net/publication/293008955_How_to_Measure_the_Power_of_Actors_and_Film_Directors
# 
# [34] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html
# 
# [35] stackoverfolw. (2023) . How do I expand the output display to see more columns of a Pandas DataFrame?.https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe. [Accessed 04/09/2023].
# 
# [36] pandas pydata. (2023) . pandas.DataFrame.corr. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html . [Accessed 04/09/2023].
# 
# [37] https://data.worldbank.org/indicator/IT.NET.USER.ZS
# 
# [38] https://www.statista.com/statistics/273018/number-of-internet-users-worldwide/#:~:text=As%20of%202022%2C%20the%20estimated,66%20percent%20of%20global%20population. 
# 
# [39] pandas pydata. (2023).pandas.DataFrame.fillna.https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html. [Accessed 04/09/2023].
# 
# [40] pandas pydata.(2023) .pandas.Series.rename.https://pandas.pydata.org/docs/reference/api/pandas.Series.rename.html. [Accessed 04/09/2023].
# 
# [41] Jake Huneycutt.(2018).Adjusting Prices for Inflation in Python with Pandas Merge. https://towardsdatascience.com/adjusting-prices-for-inflation-in-pandas-daaaa782cd89. [Accessed 04/09/2023].
# 
# [42] python basics.(2021). seaborn barplot.https://pythonbasics.org/seaborn-barplot/. [Accessed 04/09/2023].
# 
# [43] https://www.andiamo.co.uk/resources/iso-language-codes/
# 
# [44] https://www.geeksforgeeks.org/pandas-dataframe-hist-function-in-python/
# 
# [45] https://www.freecodecamp.org/news/matplotlib-figure-size-change-plot-size-in-python/
# 
# [46] https://seaborn.pydata.org/generated/seaborn.histplot.html
# 
# [47] https://www.geeksforgeeks.org/matplotlib-pyplot-axvline-in-python/
# 
# [48] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html
# 
# [49] https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
# 
# [50] https://seaborn.pydata.org/generated/seaborn.lmplot.html
# 
# [51] https://www.geeksforgeeks.org/plot-a-pie-chart-in-python-using-matplotlib/
# 
# [52] https://www.w3schools.com/python/pandas/ref_df_select_dtypes.asp
# 
# [53] https://www.andrewgurung.com/2018/12/26/preprocessing-data-binarization-using-scikit-learn/
# 
# [54] https://www.datatechnotes.com/2021/02/seleckbest-feature-selection-example-in-python.html
# 
# [55] https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python
# 
# [56] https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
# 
# [57] https://www.datacamp.com/tutorial/decision-tree-classification-python
# 
# [58] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# 
# [59] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
# 
# [60] https://saturncloud.io/blog/python-numpy-logistic-regression-a-comprehensive-guide-for-data-scientists/
# 
# [61] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
# 
# [62] https://luisvalesilva.com/datasimple/random_forests.html
# 
# [63] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
# 
# [64] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# [65] https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/#:~:text=look%20at%20each.-,One%2DVs%2DRest%20for%20Multi%2DClass%20Classification,into%20multiple%20binary%20classification%20problems.
# 
# [66] https://link.springer.com/article/10.1007/s10479-020-03804-4
# 
# [67] https://note.nkmk.me/en/python-pandas-cut-qcut-binning/
# 
# [68] https://analyticsindiamag.com/a-hands-on-introduction-to-visualizing-data-with-pandas/
# 
# [69] https://www.freecodecamp.org/news/matplotlib-figure-size-change-plot-size-in-python/
# 
# [70] https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
# 
# [71] https://matplotlib.org/stable/tutorials/introductory/pyplot.html
# 
# [72] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html
# 
# [73] P.Yoo.(2022). FEATURE SELECTION AND RESAMPLING [PDF,lab material]. Available:https://moodle.bbk.ac.uk/course/view.php?id=38405
# 
