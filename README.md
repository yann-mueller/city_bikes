# Manual
Additional notes on the code used to solve the tasks in the Data Science challenge. 
Data transformation files are stored in the `./01_build`-folder while the analysis files
are stored in the `./02_analysis`-folder.

### Instructions

Using PostgreSQL to initialize databases.

#### 1) Initialize Databases
`bash ./01_build/init_db_rides.sh`

`bash ./01_build/init_db_accident.sh`

#### 2) Download Data and import into Databases
`python ./01_build/build_rides_data.py`

`python ./01_build/build_accident_data.py`

#### 3) Analyses
`python ./02_analysis/descriptive_statistics_ride.py`

`python ./02_analysis/descriptive_statistics_accidents.py`

`python ./02_analysis/insurance_bundles.py`

`python ./02_analysis/individual_pricing.py`


### Data Sources

Ride Data: [CitiBike](https://s3.amazonaws.com/tripdata/index.html)

Accident Data: [NYPD](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/data_preview)

Weather Data: [Open-meteo](https://open-meteo.com/)

NYC ZIP Code Shapefile: [NYC OpenData](https://data.cityofnewyork.us/Health/Modified-Zip-Code-Tabulation-Areas-MODZCTA-/pri4-ifjk)

NYC Transportation Network Shapefile: [NYC OpenData](https://data.cityofnewyork.us/City-Government/Centerline/inkn-q76z/about_data)