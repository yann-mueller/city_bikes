import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from datetime import datetime
from sqlalchemy import create_engine


# NY coordinates
latitude = 40.7128
longitude = -74.0060


#%% Download weather data from Open-Meteo
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
temp = pd.read_sql("""
    SELECT 
        crash_date AS accident_date,
        COUNT(*) AS num_accidents
    FROM collisions
    WHERE crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
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
df['num_accidents'] = df['num_accidents'].fillna(0).astype(int)

# Create Accidents/Ride column
df['accidents_per_ride'] = df['num_accidents'] / df['num_rides']

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