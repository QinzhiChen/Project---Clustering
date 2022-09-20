#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from env import user, password, host
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import os
import csv
import wrangle_final
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


# the target will be taxamount


# In[3]:


zillow_train,zillow_validate,zillow_test=wrangle_final.wrangled_file()


# In[ ]:





# In[5]:


x_train, y_train = zillow_train.drop(columns='logerror'),zillow_train.logerror
x_validate, y_validate = zillow_validate.drop(columns='logerror'),zillow_validate.logerror
x_test, y_test = zillow_test.drop(columns='logerror'),zillow_test.logerror


# In[6]:


x_train=x_train.drop(columns=['propertylandusedesc','county','propertycountylandusecode'])
x_validate=x_validate.drop(columns=['propertylandusedesc','county','propertycountylandusecode'])
x_test=x_test.drop(columns=['propertylandusedesc','county','propertycountylandusecode'])             


# In[7]:


scaler=MinMaxScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
x_validate_scaled = pd.DataFrame(scaler.fit_transform(x_validate), index=x_validate.index, columns=x_validate.columns)
x_test_scaled = pd.DataFrame(scaler.fit_transform(x_test), index=x_test.index, columns=x_test.columns)




# In[9]:


# We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
y_train = pd.DataFrame(y_train)
y_validate = pd.DataFrame(y_validate)
y_test=pd.DataFrame(y_test)

# predict mean
y_train['baseline'] = y_train['logerror'].mean()
y_validate['baseline'] = y_validate['logerror'].mean()

# predict median
y_train['logerror_med'] = y_train['logerror'].median()
y_validate['logerror_med'] = y_validate['logerror'].median()

y_test['baseline'] = y_test['logerror'].mean()
y_test['logerror_med'] = y_test['logerror'].median()

def baseline():
# RMSE of mean
    rmse_train = mean_squared_error(y_train.logerror, y_train.baseline)**(1/2)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.baseline)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2),
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    # RMSE of median
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_med)**(1/2)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_med)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2),
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))




def x_y_split(zillow_train,zillow_validate,zillow_test):
    x_train, y_train = zillow_train.select_dtypes('float').drop(columns='logerror'),zillow_train.logerror
    x_validate, y_validate = zillow_validate.select_dtypes('float').drop(columns='logerror'),zillow_validate.logerror
    x_test, y_test = zillow_test.select_dtypes('float').drop(columns='logerror'),zillow_test.logerror
    return x_train, y_train,x_validate,y_validate,x_test,y_test


# In[ ]:


def models(y_train):
    models= pd.DataFrame(
    [
        {
            'model': 'baseline',
            'rmse': mean_squared_error(zillow_train['logerror'], y_train.baseline,squared=False),
            'r^2': explained_variance_score(zillow_train['logerror'], y_train.baseline)

        }
    ])
    return models


# In[ ]:


def modeling_train(model, 
                  x_train, 
                  y_train, 
                  x_validate, 
                  y_validate, 
                  scores=models):
    model.fit(x_train, y_train.logerror)
    in_sample_pred = model.predict(x_train)
    out_sample_pred = model.predict(x_validate)
    model_name = input('model name?')
    y_train[model_name] = in_sample_pred
    y_validate[model_name] = out_sample_pred
    rmse_val = mean_squared_error(
    y_train.logerror, in_sample_pred, squared=False)
    r_squared_val = explained_variance_score(
        y_train.logerror, in_sample_pred)
    return pd.DataFrame([{
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    
    }])



# In[ ]:


def modeling_validate(model, 
                  x_train, 
                  y_train, 
                  x_validate, 
                  y_validate, 
                  scores=models):
    model.fit(x_train, y_train.logerror)
    in_sample_pred = model.predict(x_train)
    out_sample_pred = model.predict(x_validate)
    model_name = input('model name?')
    y_train[model_name] = in_sample_pred
    y_validate[model_name] = out_sample_pred
    rmse_val = mean_squared_error(
    y_validate.logerror, out_sample_pred, squared=False)
    r_squared_val = explained_variance_score(
        y_validate.logerror, out_sample_pred)
    return pd.DataFrame([{
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    
    }])


# In[25]:


def modeling_test(model, 
                  x_test, 
                  y_test,  
                  scores=models):
    model.fit(x_test, y_test.logerror)
    in_sample_pred = model.predict(x_test)
    model_name = input('model name?')
    y_test[model_name] = in_sample_pred
    rmse_val = mean_squared_error(
    y_test.logerror, in_sample_pred, squared=False)
    r_squared_val = explained_variance_score(
        y_test.logerror, in_sample_pred)
    return pd.DataFrame([{
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val}])
    


# In[

# In[ ]:





