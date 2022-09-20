import wrangle_final

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
from sklearn.cluster import KMeans


# building and X to start clustering
def cluster(target):
    X = target[['sqtft_scaled']]
    with plt.style.context('seaborn-whitegrid'):
    #graph size
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')
        kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    kmeans.predict(X)
    target['cluster']= kmeans.predict(X)
    
def cluster2(target):
    X = target[['bathroom','bedroom']]
    with plt.style.context('seaborn-whitegrid'):
    #graph size
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')
        kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)

    kmeans.predict(X)
    target['cluster2']= kmeans.predict(X)
    
def cluster3(target):
    X = target[["age","month"]]
    with plt.style.context('seaborn-whitegrid'):
    #graph size
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')
        kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    kmeans.predict(X)
    target['cluster3']= kmeans.predict(X)


