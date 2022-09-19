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
import wrangle_zillow
import prepare
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures


# In[5]:


# the target will be taxamount


# In[6]:

def x_y_split(zillow_train,zillow_validate,zillow_test):
    x_train, y_train = zillow_train.select_dtypes('float').drop(columns='taxvalue'),zillow_train.taxvalue
    x_validate, y_validate = zillow_validate.select_dtypes('float').drop(columns='taxvalue'),zillow_validate.taxvalue
    x_test, y_test = zillow_test.select_dtypes('float').drop(columns='taxvalue'),zillow_test.taxvalue
    return x_train, y_train,x_validate,y_validate,x_test,y_test
def models(y_train):
    models= pd.DataFrame(
    [
        {
            'model': 'baseline',
            'rmse': mean_squared_error(y_train['taxvalue'], y_train.baseline,squared=False),
            'r^2': explained_variance_score(y_train['taxvalue'], y_train.baseline)

        }
    ])
    return models

def modeling_train(model, 
                  x_train, 
                  y_train, 
                  x_validate, 
                  y_validate, 
                  scores=models):
    model.fit(x_train, y_train.taxvalue)
    in_sample_pred = model.predict(x_train)
    out_sample_pred = model.predict(x_validate)
    model_name = input('model name?')
    y_train[model_name] = in_sample_pred
    y_validate[model_name] = out_sample_pred
    rmse_val = mean_squared_error(
    y_train.taxvalue, in_sample_pred, squared=False)
    r_squared_val = explained_variance_score(
        y_train.taxvalue, in_sample_pred)
    return pd.DataFrame([{
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    
    }])


def modeling_validate(model, 
                  x_train, 
                  y_train, 
                  x_validate, 
                  y_validate, 
                  scores=models):
    model.fit(x_train, y_train.taxvalue)
    in_sample_pred = model.predict(x_train)
    out_sample_pred = model.predict(x_validate)
    model_name = input('model name?')
    y_train[model_name] = in_sample_pred
    y_validate[model_name] = out_sample_pred
    rmse_val = mean_squared_error(
    y_validate.taxvalue, out_sample_pred, squared=False)
    r_squared_val = explained_variance_score(
        y_validate.taxvalue, out_sample_pred)
    return pd.DataFrame([{
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    
    }])

def modeling_test(model, 
                  x_test, 
                  y_test,  
                  scores=models):
    model.fit(x_test, y_test.taxvalue)
    in_sample_pred = model.predict(x_test)
    model_name = input('model name?')
    y_test[model_name] = in_sample_pred
    rmse_val = mean_squared_error(
    y_test.taxvalue, in_sample_pred, squared=False)
    r_squared_val = explained_variance_score(
        y_test.taxvalue, in_sample_pred)
    return pd.DataFrame([{
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    
    }])


# %%
