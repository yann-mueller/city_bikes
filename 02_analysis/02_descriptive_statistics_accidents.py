import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg' if you have PyQt installed

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

# Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/nypd")

#%% Column type information
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

#%% Total Accidents in 2024
temp = pd.read_sql("SELECT COUNT(*) AS total_accidents FROM collisions;", engine)
print(temp)

#%% Total accidents per month
query = """
SELECT 
    DATE_TRUNC('month', crash_date) AS month,
    COUNT(*) AS total_accidents
FROM 
    collisions
WHERE 
    crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
GROUP BY 
    month
ORDER BY 
    month;
"""

temp = pd.read_sql(query, engine)
print(temp)

#%% Unique contributing factors
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
to_remove = ['Dirt Bike', 'DIRTBIKE', 'gas bike', 'Minibike', 'Motorbike']

# Filter the list
bike_strings_list = [item for item in bike_strings_list if item not in to_remove]

print(bike_strings_list)
