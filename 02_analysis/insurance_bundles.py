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
url = f"https://archive-api.open-meteo.com/v1/archive?" \
      f"latitude={latitude}&longitude={longitude}" \
      f"&start_date=2024-01-01&end_date=2024-12-31" \
      f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,snowfall_sum,windspeed_10m_max" \
      f"&timezone=America/New_York"

# API request
response = requests.get(url)
data = response.json()

df = pd.DataFrame({
    "date": pd.to_datetime(data["daily"]["time"]),
    "precipitation_sum_mm": data["daily"]["precipitation_sum"],
    "temperature_max_c": data["daily"]["temperature_2m_max"],
    "temperature_min_c": data["daily"]["temperature_2m_min"],
    "snowfall_sum_cm": data["daily"]["snowfall_sum"],
    "windspeed_max_kmh": data["daily"]["windspeed_10m_max"]
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

#%% Scatterplot Precipitation
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

#%% Scatterplot Windspeed
# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(df['windspeed_max_kmh'], df['accidents_per_ride'], alpha=0.7)

# Regression
sns.regplot(
    x='windspeed_max_kmh',
    y='accidents_per_ride',
    data=df,
    scatter=False,     # Don't double-plot scatter points
    ci=95,              # 95% confidence interval
    color='red',
    line_kws={'label': 'Lineare Regression'}
)

# Labels and title
plt.xlabel("Max. Windgeschwindigkeit (kmh)", fontsize=14)
plt.ylabel("Unfälle pro Fahrt", fontsize=14)
plt.title("Zusammenhang zwischen Windgeschwindigkeit und Unfallrate (2024)", fontsize=16)

plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(left=0)

# Save
plt.tight_layout()
plt.savefig("02_analysis/plots/scatter_wind_accidents_per_ride.png")


####################################
### Accidents/Ride per Time Slot ###
####################################

#%% Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/citibike")

# Query rides per 3-hour slot
temp = pd.read_sql("""
    SELECT 
        FLOOR(EXTRACT(hour FROM started_at::timestamp) / 3) * 3 AS time_slot,
        COUNT(*) AS num_rides
    FROM trips
    WHERE DATE_PART('year', started_at::timestamp) = 2024
    GROUP BY time_slot
    ORDER BY time_slot;
""", con=engine)

# Prepare rides df
df_time_slots = temp.copy()

#%% Bike Accidents per 3-Hour Slot
# Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/nypd")

# Query bike accidents per 3-hour slot
temp = pd.read_sql(f"""
    SELECT 
        FLOOR(EXTRACT(hour FROM crash_time::time) / 3) * 3 AS time_slot,
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
    GROUP BY time_slot
    ORDER BY time_slot;
""", con=engine)

# Merge bike accidents into rides df
df_time_slots = df_time_slots.merge(
    temp,
    how='left',
    left_on='time_slot',
    right_on='time_slot'
)

# Fill missing accident slots with 0
df_time_slots['num_bike_accidents'] = df_time_slots['num_bike_accidents'].fillna(0).astype(int)

# Create accidents per ride
df_time_slots['accidents_per_ride'] = df_time_slots['num_bike_accidents'] / df_time_slots['num_rides']

# Optional: Create readable label for time slots
df_time_slots['time_slot_label'] = df_time_slots['time_slot'].apply(lambda x: f"{str(int(x)).zfill(2)}:00–{str((int(x)+3)%24).zfill(2)}:00")

# Sort by time slot
df_time_slots = df_time_slots.sort_values('time_slot')

# Accidents per 10.000 rides
df_time_slots['accidents_per_ride'] = df_time_slots['accidents_per_ride'] * 100000

#%% Plot: Accidents/Ride per Time Slot
# Bar Chart: Accidents per Ride by 3-Hour Time Slot
plt.figure(figsize=(10, 6))
plt.bar(df_time_slots['time_slot_label'], df_time_slots['accidents_per_ride'], width=0.6)

# Labels and Title
plt.xlabel("Zeitfenster", fontsize=14, labelpad=10)
plt.ylabel("Unfälle pro 10.000 Fahrten", fontsize=14)
plt.title("Unfallrate per 3-Stunden-Fenster (2024)", fontsize=16)

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig("02_analysis/plots/accidents_per_ride_time_slot.png")