#!/usr/bin/env python
# coding: utf-8

# In[1]:




#standard imports
import pandas as pd
import numpy as np


import env
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,LassoLars,TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,explained_variance_score




def scale_data(zillow_train,zillow_validate,zillow_test,cols):
    #make the scaler
    scaler = RobustScaler()
    #fit the scaler at train data only
    scaler.fit(zillow_train[cols])
    #tranforrm train, validate and test
    zillow_train_scaled = scaler.transform(zillow_train[cols])
    zillow_validate_scaled = scaler.transform(zillow_validate[cols])
    zillow_test_scaled = scaler.transform(zillow_test[cols])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in cols]
    
    #concatenate with orginal train, validate and test
    scaled_train = pd.concat([zillow_train.reset_index(drop = True),pd.DataFrame(zillow_train_scaled,columns = scaled_columns)],axis = 1)
    scaled_validate = pd.concat([zillow_validate.reset_index(drop = True),pd.DataFrame(zillow_validate_scaled, columns = scaled_columns)], axis = 1)
    scaled_test= pd.concat([zillow_test.reset_index(drop = True),pd.DataFrame(zillow_test_scaled,columns = scaled_columns)],axis = 1)
    
    return scaled_train,scaled_validate,scaled_test



#calculate rmse using actual and baseline mean
def get_baseline(y_train,y_validate):
    RMSE_train_mean=mean_squared_error(y_train.logerror,y_train.baseline_mean, squared = False)
    RMSE_validate_mean=mean_squared_error(y_validate.logerror,y_validate.baseline_mean, squared = False)

    print("RMSE using Mean on \nTrain: ", round(RMSE_train_mean,8), "\nValidate: ", round(RMSE_validate_mean,8))
    print()

#calculate rmse using actual and baseline mean
    RMSE_train_median= mean_squared_error(y_train.logerror,y_train.baseline_median, squared = False)
    RMSE_validate_median= mean_squared_error(y_validate.logerror,y_validate.baseline_median, squared = False)

    print("RMSE using Median on \nTrain: ", round(RMSE_train_median,8), "\nValidate: ", round(RMSE_validate_median,8))



def linear_regression(X_train,y_train,X_validate,y_validate):

    # create the model object
    lm = LinearRegression(normalize = True)
    
    # Fit the model
    lm.fit(X_train, y_train.logerror)
    
    # Predict y on train
    y_train['logerror_pred_lm'] = lm.predict(X_train)
    # predict validate
    y_validate['logerror_pred_lm'] = lm.predict(X_validate)
    
    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm) ** (1/2)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm) ** (1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample", round(rmse_train, 8),
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 8))


def lasso_lars(X_train, y_train, X_validate, y_validate, alpha):
    
    # create the model object
    lars = LassoLars(alpha)

    # fit the model.
    lars.fit(X_train, y_train.logerror)

    # predict train
    y_train['logerror_pred_lars'] = lars.predict(X_train)
    # predict validate
    y_validate['logerror_pred_lars'] = lars.predict(X_validate)
    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lars)**(1/2)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars, alpha = ", alpha, "\nTraining/In-Sample: ", round(rmse_train, 8),
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 8))



def Tweedie_regressor(X_train, y_train, X_validate, y_validate, power, alpha):

    # create the model object
    glm = TweedieRegressor(power=power, alpha=alpha)

    # fit the model to our training data.
    glm.fit(X_train, y_train.logerror)

    # predict train
    y_train['logerror_pred_glm'] = glm.predict(X_train)
    # predict validate
    y_validate['logerror_pred_glm'] = glm.predict(X_validate)
    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_glm)**(1/2)
    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=", power, " & alpha=", alpha,
        "\nTraining/In-Sample: ", round(rmse_train, 8),
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 8))


def polynomial_regression(X_train, y_train, X_validate, y_validate, degree):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree= degree)
    
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    

    # transform X_validate_scaled
    X_validate_degree2 = pf.transform(X_validate)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train,
    # since we have converted it to a dataframe from a series!
    lm2.fit(X_train_degree2, y_train.logerror)

    # predict train
    y_train['logerror_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.logerror, y_train.logerror_pred_lm2)**(1/2)

    # predict validate
    y_validate['logerror_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.logerror_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=", degree, "\nTraining/In-Sample: ", round(rmse_train,8),
        "\nValidation/Out-of-Sample: ", round(rmse_validate,8))



def model_performance(y_validate):
    plt.figure(figsize=(16,8))
    plt.xlim(-.25, .25)
    plt.ylim(-.25, .25)
    plt.plot(y_validate.logerror, y_validate.baseline_mean, alpha=.5, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (0.012, 0.012 ))
    plt.plot(y_validate.logerror, y_validate.logerror, alpha=.5, color="blue", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.10, .12), rotation=25)


    plt.scatter(y_validate.logerror, y_validate.logerror_pred_lm2,
            alpha=.5, color="green", s=100, label="Model 2nd degree Polynomial")
    plt.scatter(y_validate.logerror, y_validate.logerror_pred_lm,
            alpha=.5, color="red", s=100, label="Model: LinearRegression")
    plt.scatter(y_validate.logerror, y_validate.logerror_pred_lars,
            alpha=.5, color="blue", s=100, label="Model: LassoLars")



    plt.legend()
    plt.xlabel("Actual Margin of Error")
    plt.ylabel("Predicted Margin of Error")
    plt.title("Where are predictions more extreme? More modest?")
    plt.show()



def test_prediction(X_train,y_train,X_test,y_test,degree):
   
    lars = LassoLars(alpha=0)
    lars.fit(X_test, y_test.logerror)
    # fit the model to our training data. We must specify the column in y_train,
    # since we have converted it to a dataframe from a series!
    # predict test
    y_test['logerror_pred_lars'] = lars.predict(X_test)
    # evaluate: test rmse
    rmse_test = mean_squared_error(y_test.logerror, y_test.logerror_pred_lars)**(1/2)

    print("RMSE for Linear Regression Model,","\ntest: ", rmse_test, "\nr^2: ", explained_variance_score(y_test.logerror,
                                           y_test.logerror_pred_lars))
