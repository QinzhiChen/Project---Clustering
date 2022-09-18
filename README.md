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
|bedroomcnt|Number of bedrooms in home |
|buildingqualitytypeid|Number of bedrooms in home |
|buildingclasstypeid | Overall assessment of condition of the building from best (lowest) to worst (highest)|
|calculatedbathnbr| Number of bathrooms in home including fractional bathroom|
|decktypeid|Type of deck (if any) present on parcel|
|threequarterbathnbr| Number of 3/4 bathrooms in house (shower + sink + toilet)|
|finishedfloor1squarefeet|  Size of the finished living area on the first (entry) floor of the home|
|calculatedfinishedsquarefeet| Calculated total finished living area of the home |
|finishedsquarefeet6|Base unfinished and finished area|
|finishedsquarefeet12|Finished living area|
|finishedsquarefeet13|Perimeter  living area|
|finishedsquarefeet15|Total area|
|finishedsquarefeet50| Size of the finished living area on the first (entry) floor of the home|
|fips| Federal Information Processing Standard code|
|fireplacecnt| Number of fireplaces in a home (if any)|
|fireplaceflag| Is a fireplace present in this home |
|fullbathcnt| Number of full bathrooms (sink, shower + bathtub, and toilet) present in home|
|garagecarcnt| Total number of garages on the lot including an attached garage|
|garagetotalsqft| Total number of square feet of all garages on lot including an attached garage|
|hashottuborspa| Does the home have a hot tub or spa|
|heatingorsystemtypeid| Type of home heating system|
|latitude| Latitude of the middle of the parcel multiplied by 10e6|
|longitude| Longitude of the middle of the parcel multiplied by 10e6|
|lotsizesquarefeet| Area of the lot in square feet|
|numberofstories| Number of stories or levels the home has|
|parcelid| Unique identifier for parcels (lots) |
|poolcnt| Number of pools on the lot (if any)|
|poolsizesum| Total square footage of all pools on property|
|pooltypeid10| Spa or Hot Tub|
|pooltypeid2| Pool with Spa/Hot Tub|
|pooltypeid7| Pool without hot tub|
|propertycountylandusecode| County land use code i.e. it's zoning at the county level|
|propertylandusetypeid| Type of land use the property is zoned for|
|propertyzoningdesc| Description of the allowed land uses (zoning) for that property|
|rawcensustractandblock| Census tract and block ID combined - also contains blockgroup assignment by extension|
|censustractandblock| Census tract and block ID combined - also contains blockgroup assignment by extension|
|regionidcounty|County in which the property is located|
|regionidcity| City in which the property is located (if any)|
|regionidzip| Zip code in which the property is located|
|regionidneighborhood|Neighborhood in which the property is located|
|roomcnt| Total number of rooms in the principal residence|
|storytypeid| Type of floors in a multi-story house (i.e. basement and main level, split-level, attic|
|typeconstructiontypeid| What type of construction material was used to construct the home|
|unitcnt|Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...)|
|yardbuildingsqft17|Patio in  yard|
|yardbuildingsqft26|Storage shed/building in yard|
|yearbuilt|The Year the principal residence was built |
|taxvaluedollarcnt|The total tax assessed value of the parcel
|landtaxvaluedollarcnt|The assessed value of the land area of the parcel
|taxamount|The total property tax assessed for that assessment year|
|assessmentyear|The year of the property tax assessment |
|taxdelinquencyflag|Property taxes for this parcel are past due as of 2015|
|taxdelinquencyyear|Year for which the unpaid propert taxes were due |
   
   
 








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







