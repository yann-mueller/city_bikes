import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from datetime import datetime
from sqlalchemy import create_engine


##############
### CONFIG ###
##############

#%% Create Bike List first
# Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/nypd")

# Column type information
query = """
SELECT 
    column_name, 
    data_type 
FROM 
    information_schema.columns 
WHERE 
    table_name = 'collisions'
ORDER BY 
    ordinal_position;
"""

temp = pd.read_sql(query, engine)
print(temp)

# Unique contributing factors
query = """
SELECT DISTINCT vehicle_type_code_1 AS factor FROM collisions
UNION
SELECT DISTINCT vehicle_type_code_2 FROM collisions
UNION
SELECT DISTINCT vehicle_type_code_3 FROM collisions
UNION
SELECT DISTINCT vehicle_type_code_4 FROM collisions
UNION
SELECT DISTINCT vehicle_type_code_5 FROM collisions
ORDER BY factor;
"""

temp = pd.read_sql(query, engine)
print(temp)

# Filter for vehicle types containing 'bike' or 'bicycle'
bike_strings = temp[temp['factor'].str.contains('bike|bicycle', case=False, na=False)]

# Convert to list
bike_strings_list = bike_strings['factor'].tolist()

print(bike_strings_list)

# Entries to remove
to_remove = ['Dart bike', 'dirt bike', 'Dirt Bike', 'DIRT BIKE', 'dirtbike', 'Dirtbike', 'DIRTBIKE', 'Moped bike',
             'gas bike', 'Minibike', 'Motorbike', 'motorbike', 'PEDAL BIKE', 'Dirt Bike']

# Filter the list
bike_strings_list = [item for item in bike_strings_list if item not in to_remove]

print(bike_strings_list)

# Create a tuple of strings for SQL IN clause
bike_tuple = tuple(bike_strings_list)


##############################################
### Accidents/Ride per Precipitation Level ###
##############################################

#%% NY coordinates
latitude = 40.7128
longitude = -74.0060

# Download weather data from Open-Meteo
# API URL
url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date=2024-01-01&end_date=2024-12-31&daily=precipitation_sum&timezone=America/New_York"

# API request
response = requests.get(url)
data = response.json()

df = pd.DataFrame({
    "date": data["daily"]["time"],
    "precipitation_sum_mm": data["daily"]["precipitation_sum"]
})

#%% Add Rides per Day
# Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/citibike")

temp = pd.read_sql("""
    SELECT 
        DATE(started_at) AS ride_date,
        COUNT(*) AS num_rides
    FROM trips
    WHERE DATE_PART('year', started_at::timestamp) = 2024
    GROUP BY ride_date
    ORDER BY ride_date;
""", con=engine)

# Prepare rides df
temp['ride_date'] = pd.to_datetime(temp['ride_date'])

# Prepare weather df
df['date'] = pd.to_datetime(df['date'])

# Merge rides into weather df
df = df.merge(
    temp,
    how='left',
    left_on='date',
    right_on='ride_date'
)

# rop the duplicate 'ride_date' column after merge
df = df.drop(columns=['ride_date'])

# Fill days with no rides (e.g., missing data) with 0 rides
df['num_rides'] = df['num_rides'].fillna(0).astype(int)

#%% Add Accidents per Day
# Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/nypd")

# Query accidents per day
temp = pd.read_sql(f"""
    SELECT 
        crash_date AS accident_date,
        COUNT(*) AS num_bike_accidents
    FROM collisions
    WHERE crash_date >= '2024-01-01' 
      AND crash_date < '2025-01-01'
      AND (
          vehicle_type_code_1 IN {bike_tuple} OR
          vehicle_type_code_2 IN {bike_tuple} OR
          vehicle_type_code_3 IN {bike_tuple} OR
          vehicle_type_code_4 IN {bike_tuple} OR
          vehicle_type_code_5 IN {bike_tuple}
      )
    GROUP BY crash_date
    ORDER BY crash_date;
""", con=engine)

# Prepare accident df
temp['accident_date'] = pd.to_datetime(temp['accident_date'])

# Merge accidents into weather df
df = df.merge(
    temp,
    how='left',
    left_on='date',
    right_on='accident_date'
)

# Drop the duplicate 'accident_date' column after merge
df = df.drop(columns=['accident_date'])

# Fill missing accident days with 0
df['num_bike_accidents'] = df['num_bike_accidents'].fillna(0).astype(int)

# Create Accidents/Ride column
df['accidents_per_ride'] = df['num_bike_accidents'] / df['num_rides']

#%% Scatterplot
# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(df['precipitation_sum_mm'], df['accidents_per_ride'], alpha=0.7)

# Regression
sns.regplot(
    x='precipitation_sum_mm',
    y='accidents_per_ride',
    data=df,
    scatter=False,     # Don't double-plot scatter points
    ci=95,              # 95% confidence interval
    color='red',
    line_kws={'label': 'Lineare Regression'}
)

# Labels and title
plt.xlabel("Niederschlag (mm)", fontsize=14)
plt.ylabel("Unfälle pro Fahrt", fontsize=14)
plt.title("Zusammenhang zwischen Regen und Unfallrate (2024)", fontsize=16)

plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(left=0)

# Save
plt.tight_layout()
plt.savefig("02_analysis/plots/scatter_precipitation_accidents_per_ride.png")

#%% Scatterplot Percentiles
# Calculate percentiles
df['precipitation_percentile'] = df['precipitation_sum_mm'].rank(pct=True)
df['accidents_per_ride_percentile'] = df['accidents_per_ride'].rank(pct=True)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df['precipitation_percentile'], df['accidents_per_ride_percentile'], alpha=0.7)

# Regression
reg_line = sns.regplot(
    x='precipitation_percentile',
    y='accidents_per_ride_percentile',
    data=df,
    scatter=False,
    ci=95,
    color='red',
    line_kws={'label': 'Lineare Regression'}
)

# Manually add label to the regression line
reg_line.lines[0].set_label('Lineare Regression')


# Labels and title
plt.xlabel("Niederschlags-Perzentil", fontsize=14)
plt.ylabel("Unfallrate-Perzentil", fontsize=14)
plt.title("Zusammenhang zwischen Regen und Unfallrate (Perzentile)", fontsize=16)

plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("02_analysis/plots/scatter__precipitation_accidents_percentile.png")


####################################
### Accidents/Ride per Time Slot ###
####################################



