import requests
import pandas as pd
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