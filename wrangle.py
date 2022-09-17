#!/usr/bin/env python
# coding: utf-8

# In[1]:


# setting up the environment
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from env import user, password, host
import env
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[2]:


# create a function for the acquisition 
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[3]:


# acquire required tables into one dataframe
def acquire_zillow():
    file='zillow_df.csv'
    if os.path.isfile(file):
        return pd.read_csv(file)
    else:
        zillow2017_df = pd.read_sql(('''SELECT
    prop.*,
    predictions_2017.logerror,
    predictions_2017.transactiondate,
    air.airconditioningdesc,
    arch.architecturalstyledesc,
    build.buildingclassdesc,
    heat.heatingorsystemdesc,
    landuse.propertylandusedesc,
    story.storydesc,
    construct.typeconstructiondesc
FROM properties_2017 prop
JOIN (
    SELECT parcelid, MAX(transactiondate) max_transactiondate
    FROM predictions_2017
    GROUP BY parcelid
) pred USING(parcelid)
JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                      AND pred.max_transactiondate = predictions_2017.transactiondate
LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
LEFT JOIN storytype story USING (storytypeid)
LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
WHERE prop.latitude IS NOT NULL
  AND prop.longitude IS NOT NULL
  AND predictions_2017.transactiondate like "2017%%" '''), get_connection('zillow'))
        zillow2017_df.to_csv(file,index=False)
    return zillow2017_df


# In[10]:


def zillow_info():
    zillow_df=acquire_zillow()
    print('The basic stats about zillow dataframe is: ','\n',zillow_df.describe())
    print('The shape of the zillow dataframe is: ','\n',zillow_df.info())


# ## Key takeaway
# There are 67 columns, a lot of columns are missing over 70000 values.
# There are 66 float, 2 int and 11 object
# The total count is 77375 rows retrieved

# In[12]:


# ## Takeaway
# - A lot of columns have missing over 90% of the values. the problem is that whether we should decided to drop the null values
# - Some of missing values can be dropped, which like ids. However, some values we can imputed

# In[13]:


# function for null counts 
def null_counts():
    zilloow_df=acquire_zillow()
    sumdf=pd.DataFrame(zillow_df.isnull().sum(axis=0).rename('num_rows_missing'))
    meandf=pd.DataFrame(zillow_df.isnull().mean(axis=0).rename('pct_rows_missing'))
    print(pd.concat([sumdf,meandf],axis=1))
    


# In[14]:


# clean the column and create county columns
def clean_column():
    zillow2017_df=acquire_zillow()
    zillow2017_df.rename(columns={'bedroomcnt':'bedroom','bathroomcnt':'bathroom','calculatedfinishedsquarefeet':'sqtft','taxvaluedollarcnt':'taxvalue','garagecarcnt':'garage','lotsizesquarefeet':'lots','poolcnt':'pool','regionidzip':'zipcode'},inplace=True)
    zillow2017_df['fips']= zillow2017_df['fips'].astype(object)
    value=[]
    for row in zillow2017_df['fips']:
        if row ==6037.0: value.append('Los Angeles County, CA')
        elif row == 6059.0: value.append("Orange County, CA")
        elif row == 6111.0: value.append('Ventura County, CA')
        else:
            value.append('Undetermined')
    zillow2017_df['county']=value
    zillow2017_df['zipcode']=zillow2017_df['zipcode'].astype(object)
    zillow2017_df=df = zillow2017_df[zillow2017_df.columns.drop(list(zillow2017_df.filter(regex='id')))]
    return zillow2017_df


# In[17]:


# create function that will drop null when exceed certain amount of null values
def handle_missing_values(df, prop_required_column, prop_required_row):
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df


# In[31]:


def wrangle_zillow():
    zillow_df=clean_column()
    zillow2017_df=handle_missing_values(zillow_df,.4,.5)
    train_validate, zillow_test = train_test_split(zillow2017_df, test_size=.2, random_state=123)
    zillow_train, zillow_validate = train_test_split(train_validate, test_size=.3, random_state=123)
    zillow_train['month']=pd.DatetimeIndex(zillow_train['transactiondate']).month
    zillow_train=zillow_train.drop(columns=['transactiondate'])
    zillow_validate['month']=pd.DatetimeIndex(zillow_validate['transactiondate']).month
    zillow_validate=zillow_validate.drop(columns=['transactiondate'])
    zillow_test['month']=pd.DatetimeIndex(zillow_test['transactiondate']).month
    zillow_test=zillow_test.drop(columns=['transactiondate'])
    return zillow_train, zillow_validate, zillow_test


# In[ ]:





# In[33]:


def wrangled_file():
    zillow_train,zillow_validate,zillow_test=wrangle_zillow()
    zillow_col=zillow_train.columns.copy()
    imp = SimpleImputer(strategy="most_frequent")
    zillow_train=imp.fit_transform(zillow_train)
    zillow_validate=imp.fit_transform(zillow_validate)
    zillow_test=imp.fit_transform(zillow_test)
    zillow_train=pd.DataFrame(zillow_train,columns=zillow_col)
    zillow_validate=pd.DataFrame(zillow_validate,columns=zillow_col)
    zillow_test=pd.DataFrame(zillow_test,columns=zillow_col)
    zillow_train['age']=2022.0-zillow_train.yearbuilt.astype(float)
    zillow_validate['age']=2022.0-zillow_validate.yearbuilt.astype(float)
    zillow_test['age']=2022.0-zillow_test.yearbuilt.astype(float)
    return zillow_train,zillow_validate,zillow_test


# In[ ]:





# In[ ]:




