import pandas as pd
from sqlalchemy import create_engine

# Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/citibike")


#%%
# Query the schema for the 'trips' table
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

df = pd.read_sql(query, engine)
print(df)


#%%
df = pd.read_sql("SELECT COUNT(*) AS total_rows FROM trips", con=engine)
print(df)


#%%
df = pd.read_sql("""
    SELECT MIN(started_at) AS first_ride, MAX(started_at) AS last_ride
    FROM trips
""", con=engine)
print(df)

#%%
df = pd.read_sql("""
    SELECT start_station_name, COUNT(*) AS ride_count
    FROM trips
    GROUP BY start_station_name
    ORDER BY ride_count DESC
    LIMIT 10
""", con=engine)
print(df)

#%%
df = pd.read_sql("""
    SELECT member_casual, COUNT(*) AS num_rides
    FROM trips
    GROUP BY member_casual
""", con=engine)
print(df)


