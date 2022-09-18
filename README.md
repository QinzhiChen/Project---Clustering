# Project---Clustering

# Project Goal
> - The goal of this project was to identify factors affecting errors in the Zestimate used by Zillow to estimate single family home prices in 2017. The results of this project will be used to improve the model used to estimate prices and guide future data collection efforts


# Project Description
> - We will conduct an in depth analysis of Zillow property data from 2017. We will use exploratory analysis and clustering techniques to identify the key drivers of error in Zillow's predictions, then use machine learning algorithms to create a model capable of predicting the error. If we know log errors are higher for certain home characteristics, we can investigate the reasons for higher errors, develop a mitigation plan, and improve model predictions. We used statistical testing, clustering and regression methods to provide insight into what affects log error.


# Initial Questions
> - Is there relationship between LogError and Age?
> - Is there relationship between taxrate and logerror?
> - Is there a significant different between logerror and the bathroom and bedroom counts?
> - Can we achieve lower logerror when seperate orange county out of overall?


# Data Dictionary

|Feature                               |	Description                              
|--------------------------------------|-------------------------------------------------------|
|airconditioningtypeid                | Type of cooling system present in the home            |
|architecturalstyletypeid.            |  Architectural style of the home (i.e. ranch, colonial, split-level, etc…)|
|basementsqft| Finished living area below or partially below ground level|
|bathroomcnt| Number of bathrooms in home including fractional bathrooms|
|bedroomcnt||
|buildingqualitytypeid|
|buildingclasstypeid |
|calculatedbathnbr|
|decktypeid|
|threequarterbathnbr|
|finishedfloor1squarefeet|
|calculatedfinishedsquarefeet|
|finishedsquarefeet6|
|finishedsquarefeet12|
|finishedsquarefeet13|
|finishedsquarefeet15|
|finishedsquarefeet50|
|fips|
|fireplacecnt|
|fireplaceflag|
|fullbathcnt|
|garagecarcnt|
|garagetotalsqft|
|hashottuborspa|
|heatingorsystemtypeid|
|latitude|
|longitude|
|lotsizesquarefeet|
|numberofstories|
|parcelid|
|poolcnt|
|poolsizesum|
|pooltypeid10|
|pooltypeid2|
|pooltypeid7|
|propertycountylandusecode|
|propertylandusetypeid|
|propertyzoningdesc|
|rawcensustractandblock|
|censustractandblock|
|regionidcounty|
|regionidcity|
|regionidzip|
|regionidneighborhood|
|roomcnt|
|storytypeid|
|typeconstructiontypeid|
|unitcnt|
|yardbuildingsqft17|
|yardbuildingsqft26|
|yearbuilt|
|taxvaluedollarcnt|
|landtaxvaluedollarcnt|
|taxamount|
|assessmentyear|
|taxdelinquencyflag|
|taxdelinquencyyear|
                                                                 |








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

##  Exploration

##  Modeling

## Prediction delivery

## Key Takeaways and Recommendations







