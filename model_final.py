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
import wrangle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


# the target will be taxamount


# In[3]:


zillow_train,zillow_validate,zillow_test=wrangle.wrangled_file()


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


# In[8]:


y_test


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


# In[10]:


y_train


# In[11]:


models = pd.DataFrame(
[
    {
        'model': 'baseline',
        'rmse': mean_squared_error(y_train['logerror'], y_train.baseline,squared=False),
        'r^2': explained_variance_score(y_train['logerror'], y_train.baseline)
    
    }
])
models


# In[12]:


def modeling(model, 
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
    print(y_validate.shape)
    print(out_sample_pred.shape)
    rmse_val = mean_squared_error(
    y_validate.logerror, out_sample_pred, squared=False)
    r_squared_val = explained_variance_score(
        y_validate.logerror, out_sample_pred)
    return models.append({
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    
    }, ignore_index=True)


# In[13]:


models = modeling(LinearRegression(normalize=True), 
                  x_train_scaled, 
                  y_train, 
                  x_validate_scaled, 
                  y_validate, 
                  scores=models)


# In[14]:


models = modeling(LassoLars(alpha=1.0), 
                  x_train_scaled, 
                  y_train, 
                  x_validate_scaled, 
                  y_validate, 
                  scores=models)


# In[15]:


polyfeats = PolynomialFeatures(degree=2)
x_train_quad = polyfeats.fit_transform(x_train_scaled)
x_val_quad = polyfeats.transform(x_validate_scaled)
models = modeling(LinearRegression(), 
                  x_train_quad, 
                  y_train, 
                  x_val_quad, 
                  y_validate, 
                  scores=models)


# In[16]:


models = modeling(TweedieRegressor(power=0, alpha=0), 
                  x_train_scaled, 
                  y_train, 
                  x_validate_scaled, 
                  y_validate, 
                  scores=models)


# In[17]:


models


# In[18]:


x_train_scaled


# In[19]:


kbest = SelectKBest(f_regression, k=3)
kbest.fit(x_train_scaled, y_train.logerror)
mask = x_train_scaled.columns[kbest.get_support()].to_list()


# In[20]:


models = modeling(LinearRegression(), 
                  x_train_scaled[mask], 
                  y_train, 
                  x_validate_scaled[mask], 
                  y_validate, 
                  scores=models)


# In[21]:


models


# In[22]:


x_test.shape,y_test.shape


# In[23]:


kbest = SelectKBest(f_regression, k=20)
kbest.fit(x_test_scaled, y_test.logerror)


# In[26]:


mask = x_train_scaled.columns[kbest.get_support()].to_list()
models = modeling_test(LinearRegression(), 
                  x_test_scaled[mask], 
                  y_test, 
                  scores=models)


# In[27]:


models


# In[ ]:


polyfeats = PolynomialFeatures(degree=2)
x_test_quad = polyfeats.fit_transform(x_test_scaled)


# In[ ]:


def modeling(model, 
                  x_test, 
                  y_test, 
                  scores=models):
    model.fit(x_test, y_test)
    in_sample_pred = model.predict(x_test)
    model_name = input('model_name?')
    y_test[model_name] = in_sample_pred
    rmse_val = mean_squared_error(
    y_test, in_sample_pred, squared=False)**(1/2)
    r_squared_val = explained_variance_score(
        y_test, in_sample_pred)
    return models.append({
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    
    }, ignore_index=True)


# In[ ]:





# In[ ]:


modeling(TweedieRegressor(power=0, alpha=0), 
                  x_test_scaled,
                  y_test,
                  scores=models)


# In[ ]:


polyfeats = PolynomialFeatures(degree=2)
modeltest=LinearRegression()
modeltest.fit(x_test_scaled, y_test)
x_test = polyfeats.fit_transform(x_test_scaled)


# In[ ]:


rmse_val = mean_squared_error(
    y_test, modeltest.predict(x_test_scaled), squared=False)
r_squared_val = explained_variance_score(
        y_test, modeltest.predict(x_test_scaled))


# In[ ]:


rmse_val,r_squared_val


# In[ ]:


y_train = pd.DataFrame(y_train)
y_validate = pd.DataFrame(y_validate)

# predict mean
y_train['baseline'] = y_train['logerror'].mean()
y_validate['baseline'] = y_validate['logerror'].mean()

# predict median
y_train['logerror_med'] = y_train['logerror'].median()
y_validate['logerror_med'] = y_validate['logerror'].median()

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


# In[ ]:


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
    


# In[ ]:


x_train,y_train,x_val,y_val,x_test,y_test=x_y_split(zillow_train,zillow_validate,zillow_test)


# In[ ]:


models=models(y_train)


# In[ ]:


modeling_train(LinearRegression(normalize=True), 
                  x_train, 
                  y_train, 
                  x_validate, 
                  y_validate, 
                  scores=models)


# In[ ]:





