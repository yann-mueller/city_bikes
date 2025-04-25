import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg' if you have PyQt installed

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

# Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/citibike")

#%% Column type information
query = """
SELECT 
    column_name, 
    data_type 
FROM 
    information_schema.columns 
WHERE 
    table_name = 'trips'
ORDER BY 
    ordinal_position;
"""

temp = pd.read_sql(query, engine)
print(temp)

#%% Number of unique stations
temp = pd.read_sql("SELECT COUNT(DISTINCT start_station_id) AS unique_start_stations FROM trips", con=engine)
print(temp)

#%% Number of observations
temp = pd.read_sql("SELECT COUNT(*) AS total_rows FROM trips", con=engine)
print(temp)

#%% Rides per month & Plot
query = """
SELECT 
    DATE_TRUNC('month', started_at::timestamp) AS month,
    COUNT(*) AS num_rides
FROM trips
WHERE DATE_PART('year', started_at::timestamp) = 2024
GROUP BY month
ORDER BY month;
"""

temp = pd.read_sql(query, engine)
print(temp)

# Plot: Rides per Month
temp['month_label'] = temp['month'].dt.strftime('%B')  # e.g., "January"
temp['num_rides_millions'] = temp['num_rides'] / 1_000_000

# Plot
plt.figure(figsize=(10, 5))
plt.bar(temp['month_label'], temp['num_rides_millions'], color="#1f77b4")

plt.xlabel("Monat", fontsize=16)
plt.ylabel("Anzahl Fahrten (in Millionen)", fontsize=16, labelpad=15)
plt.title("Monatliche CitiBike Fahrten – 2024", fontsize=18, pad=20)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, fontsize=14, ha='right')
plt.yticks(fontsize=14)

plt.tight_layout()

plt.savefig("02_analysis/plots/monthly_rides_2024.png")

#%% Check first and last ride
temp = pd.read_sql("""
    SELECT MIN(started_at) AS first_ride, MAX(started_at) AS last_ride
    FROM trips
""", con=engine)
print(temp)

#%% Top 10 stations
temp = pd.read_sql("""
    SELECT start_station_name, COUNT(*) AS ride_count
    FROM trips
    GROUP BY start_station_name
    ORDER BY ride_count DESC
    LIMIT 10
""", con=engine)
print(temp)

#%% Overview member status
temp = pd.read_sql("""
    SELECT member_casual, COUNT(*) AS num_rides
    FROM trips
    GROUP BY member_casual
""", con=engine)
print(temp)


