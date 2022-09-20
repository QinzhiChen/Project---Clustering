# Predicting Logerror in Zillow data

# Project Goal
> - The goal of this project was to identify factors affecting errors in the Zestimate used by Zillow to estimate single family home prices in 2017. The results of this project will be used to improve the model used to estimate prices and guide future data collection efforts


# Project Description
> - We will conduct an in depth analysis of Zillow property data from 2017. We will use exploratory analysis and clustering techniques to identify the key drivers of error in Zillow's predictions, then use machine learning algorithms to create a model capable of predicting the error. If we know log errors are higher for certain home characteristics, we can investigate the reasons for higher errors, develop a mitigation plan, and improve model predictions. We used statistical testing, clustering and regression methods to provide insight into what affects log error.


# Initial Questions
> - Whether there are correlation between sqtft and logerror
> - Whether the bedroom, bathroom, and logerror have significant relationship
> - whether the county has similar logerror
> - Whether the logerror and age could have significant group


# Data Dictionary

|Feature                               |	Description                              
|--------------------------------------|-------------------------------------------------------|
|logerror|The difference in log of Zestimate and log of sales price|
|bathroomcnt| Number of bathrooms in home including fractional bathrooms|
 |bedroomcnt|Number of bedrooms in home |
 |calculatedbathnbr| Number of bathrooms in home including fractional bathroom|
 |finishedsquarefeet12|Finished living area|
 |fips| Federal Information Processing Standard code|
 |fullbathcnt| Number of full bathrooms (sink, shower + bathtub, and toilet) present in home|
|latitude| Latitude of the middle of the parcel multiplied by 10e6|
 |longitude| Longitude of the middle of the parcel multiplied by 10e6|
 |propertycountylandusecode| County land use code i.e. it's zoning at the county level|
 |rawcensustractandblock| Census tract and block ID combined - also contains blockgroup assignment by extension|
 |censustractandblock| Census tract and block ID combined - also contains blockgroup assignment by extension|
 |yearbuilt|The Year the principal residence was built |
 |landtaxvaluedollarcnt|The assessed value of the land area of the parcel
 |taxamount|The total property tax assessed for that assessment year|
 |assessmentyear|The year of the property tax assessment |
 taxvaluedollarcnt|The total tax assessed value of the parcel|
 |calculatedfinishedsquarefeet| Calculated total finished living area of the home |
 |regionidzip| Zip code in which the property is located|
 |structuretaxvaluedollarcnt|	The assessed value of the built structure on the parcel|
 |propertyzoningdesc| Description of the allowed land uses (zoning) for that property|
   
 








## Steps to Reproduce

> -  To clone this repo, use this command in your terminal https://github.com/QinzhiChen/Project---Clustering
> -  You will need login credentials for MySQL database hosted at data.codeup.com
> -  You will need an env.py file that contains hostname,username and password
> -  The env.py should also contain a function named get_db_url() that establishes the string value of the database url.
> -  Store that env file locally in the repository.


## The plan

> - We set up our initial questions during this phase. Wr made the outline of possible exploration techniques and hypothesis testing that we can use.

##  Acquisition

> - We obtanied Zillow data by using SQL query via MySQL database and saved the file locally as a csv. We used the code created at wrangle.py.

## Preparation

A total of 77380 rows and 68 columns were retrieved.
 These are the steps taken for data clean up and split
 > - feature engineered taxrate where taxrate = taxamount/taxvalue
 > - feature engineered age column, age = present year - year built
 > - renamed columns (bedroomcnt':'bedroom','bathroomcnt':'bathroom','calculatedfinishedsquarefeet':'sqtft', 'taxvaluedollarcnt':'taxvalue','garagecarcnt':'garage','lotsizesquarefeet':'lots','poolcnt':'pool','regionidzip':'zipcode')
 > - nulls: removed columns with more than 40% nulls and rows with more than 50% nulls.
 > - nulls: dropped all other remaining null
 > - created a new column called county with county names corresponding to the fips.
 >- created a new column called month corresponding to the month in column transactiondate
 > - dropped columns: transactiondate,heatingorsystemdesc,unitcnt,propertyzoningdesc,lots
 > - outliers: dropped bathroom count over 5, bedroom count over 8, tax rate over 0.20 and sqtft over 4000

  Finally, split data into train (56%), validate (24%), test(20%)

##  Exploration

 We conducted an initial exploration of the data by examining correlations between each of the potential features and the target. We also explored further by using K-means clustering to see relationship of our features with the target

##  Modeling

We scaled our top features using MinMaxScaler. We used RMSE using mean for the baseline and used Linear Regression(OLS),kbest, Lassor + Lars, General Linear Regression(Tweedier Regressor) and Polynomial Regression to evalute our model. The modeling we selected is the polynominal modeling with two degree, which achieved 0.15748 RMSE and .02091 Rsquare.

## Prediction delivery

Using our top model, we able to predict the logerror on our data.

## Key Takeaways and Recommendations

All models performed better than the baseline, even if not by much. We do not recommend to use this model. Also we recommend that we need better data. Roughly half of all features are lost due to missing too many values which reduces the number of potential drivers of logerror we could analyze. Clustering was shown to be a useful exercise, so additional clustering exploration is recommended in an attempt to find clusters that serve as larger drivers.








