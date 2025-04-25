import matplotlib
matplotlib.use('TkAgg')

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import importlib.util

# Path to the file you want to import from
file_path = os.path.join(os.getcwd(), "02_analysis", "subroutines", "sub_create_map.py")

# Load the module manually
spec = importlib.util.spec_from_file_location("sub_create_map", file_path)
sub_create_map = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub_create_map)

#%%
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
to_remove = ['Dirt Bike', 'DIRTBIKE', 'gas bike', 'Minibike', 'Motorbike']

# Filter the list
bike_strings_list = [item for item in bike_strings_list if item not in to_remove]

print(bike_strings_list)

# Create a tuple of strings for SQL IN clause
bike_tuple = tuple(bike_strings_list)

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


#%% Summarizing table & Plot
# SQL query to get both total and bike-related accidents per month
query = f"""
WITH monthly_totals AS (
    SELECT 
        DATE_TRUNC('month', crash_date) AS month,
        COUNT(*) AS total_accidents
    FROM collisions
    WHERE crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
    GROUP BY month
),
bike_related AS (
    SELECT 
        DATE_TRUNC('month', crash_date) AS month,
        COUNT(*) AS bike_accidents
    FROM collisions
    WHERE crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
      AND (
          vehicle_type_code_1 IN {bike_tuple} OR
          vehicle_type_code_2 IN {bike_tuple} OR
          vehicle_type_code_3 IN {bike_tuple} OR
          vehicle_type_code_4 IN {bike_tuple} OR
          vehicle_type_code_5 IN {bike_tuple}
      )
    GROUP BY month
)
SELECT 
    t.month,
    t.total_accidents,
    COALESCE(b.bike_accidents, 0) AS bike_accidents,
    ROUND(COALESCE(b.bike_accidents, 0)::decimal / t.total_accidents, 4) AS bike_share
FROM 
    monthly_totals t
LEFT JOIN 
    bike_related b ON t.month = b.month
ORDER BY 
    t.month;
"""

temp = pd.read_sql(query, engine)
print(temp)


print(f"Average Bike Accidents per Month: {temp['bike_accidents'].mean():.2f}")


# Make sure 'month' is datetime and sorted
temp['month'] = pd.to_datetime(temp['month'])
temp = temp.sort_values('month')

# Create month labels
temp['month_label'] = temp['month'].dt.strftime('%B')

# Plot
plt.figure(figsize=(10, 5))
plt.bar(temp['month_label'], temp['bike_accidents'], color="#ff7f0e")

# Dashed average line
plt.axhline(temp['bike_accidents'].mean(), color="darkred", linestyle="--", linewidth=2, label=f"Durchschnitt ({round(temp['bike_accidents'].mean(), 0):.0f})")

# Add label above the line
plt.text(
    x=0.1,  # X position in axis coordinates (0 = left, 1 = right)
    y=temp['bike_accidents'].mean() + 15,  # Y position in data coordinates, slightly above the line
    s=f"Durchschnitt: {round(temp['bike_accidents'].mean(), 0):.0f}",
    color='darkred',
    fontsize=12,
    ha='center',
    va='bottom',
    transform=plt.gca().get_yaxis_transform()  # Keep x in axis coords, y in data coords
)

plt.xlabel("Monat", fontsize=16)
plt.ylabel("Unfälle mit Fahrradbeteiligung", fontsize=16, labelpad=15)
plt.title("Monatliche Fahrradunfälle – 2024", fontsize=18, pad=20)

plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.ylim(0, 1000)

plt.tight_layout()

# Save
plt.savefig("02_analysis/plots/monthly_bike_accidents_2024.png")

#%% Fatal Bike Accidents
# Updated SQL query to include injured/fatal bike-related accidents
query = f"""
WITH bike_accidents AS (
    SELECT 
        DATE_TRUNC('month', crash_date) AS month,
        COUNT(*) AS bike_accidents
    FROM collisions
    WHERE crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
      AND (
          vehicle_type_code_1 IN {bike_tuple} OR
          vehicle_type_code_2 IN {bike_tuple} OR
          vehicle_type_code_3 IN {bike_tuple} OR
          vehicle_type_code_4 IN {bike_tuple} OR
          vehicle_type_code_5 IN {bike_tuple}
      )
    GROUP BY month
),
injured_bike_accidents AS (
    SELECT 
        DATE_TRUNC('month', crash_date) AS month,
        COUNT(*) AS injured_bike_accidents
    FROM collisions
    WHERE crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
      AND (
          vehicle_type_code_1 IN {bike_tuple} OR
          vehicle_type_code_2 IN {bike_tuple} OR
          vehicle_type_code_3 IN {bike_tuple} OR
          vehicle_type_code_4 IN {bike_tuple} OR
          vehicle_type_code_5 IN {bike_tuple}
      )
      AND (
          number_of_cyclist_injured >= 1 OR
          number_of_pedestrians_injured >= 1
      )
    GROUP BY month
),
fatal_bike_accidents AS (
    SELECT 
        DATE_TRUNC('month', crash_date) AS month,
        COUNT(*) AS fatal_bike_accidents
    FROM collisions
    WHERE crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
      AND (
          vehicle_type_code_1 IN {bike_tuple} OR
          vehicle_type_code_2 IN {bike_tuple} OR
          vehicle_type_code_3 IN {bike_tuple} OR
          vehicle_type_code_4 IN {bike_tuple} OR
          vehicle_type_code_5 IN {bike_tuple}
      )
      AND (
          number_of_cyclist_killed >= 1 OR
          number_of_pedestrians_killed >= 1
      )
    GROUP BY month
)
SELECT 
    b.month,
    b.bike_accidents,
    COALESCE(i.injured_bike_accidents, 0) AS injured_bike_accidents,
    COALESCE(f.fatal_bike_accidents, 0) AS fatal_bike_accidents,
    ROUND(COALESCE(i.injured_bike_accidents, 0)::decimal / NULLIF(b.bike_accidents, 0), 4) AS injured_share,
    ROUND(COALESCE(f.fatal_bike_accidents, 0)::decimal / NULLIF(b.bike_accidents, 0), 4) AS fatal_share
FROM 
    bike_accidents b
LEFT JOIN 
    injured_bike_accidents i ON b.month = i.month
LEFT JOIN 
    fatal_bike_accidents f ON b.month = f.month
ORDER BY 
    b.month;
"""

temp = pd.read_sql(query, engine)
print(temp)


#%% Accidents by 3-Hour Time Slots
query = f"""
WITH all_collisions AS (
    SELECT 
        (EXTRACT(hour FROM crash_time)::int / 3) * 3 AS hour_bin,
        vehicle_type_code_1,
        vehicle_type_code_2,
        vehicle_type_code_3,
        vehicle_type_code_4,
        vehicle_type_code_5,
        number_of_persons_injured,
        number_of_cyclist_injured,
        number_of_pedestrians_injured,
        number_of_cyclist_killed,
        number_of_pedestrians_killed
    FROM collisions
    WHERE crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
),
bike_crashes AS (
    SELECT 
        hour_bin,
        COUNT(*) AS bike_accidents,
        SUM(
            CASE 
                WHEN number_of_persons_injured >= 1 OR number_of_cyclist_injured >= 1 OR number_of_pedestrians_injured >= 1 
                THEN 1 ELSE 0 
            END
        ) AS injured_bike_accidents,
        SUM(
            CASE 
                WHEN number_of_cyclist_killed >= 1 OR number_of_pedestrians_killed >= 1 
                THEN 1 ELSE 0 
            END
        ) AS fatal_bike_accidents
    FROM all_collisions
    WHERE 
        vehicle_type_code_1 IN {bike_tuple} OR
        vehicle_type_code_2 IN {bike_tuple} OR
        vehicle_type_code_3 IN {bike_tuple} OR
        vehicle_type_code_4 IN {bike_tuple} OR
        vehicle_type_code_5 IN {bike_tuple}
    GROUP BY hour_bin
)
SELECT 
    hour_bin,
    bike_accidents,
    injured_bike_accidents,
    fatal_bike_accidents,
    ROUND(injured_bike_accidents::decimal / NULLIF(bike_accidents, 0), 4) AS injured_share,
    ROUND(fatal_bike_accidents::decimal / NULLIF(bike_accidents, 0), 4) AS fatal_share
FROM 
    bike_crashes
ORDER BY 
    hour_bin;
"""

temp = pd.read_sql(query, engine)

# Optional: Label the time slots
temp['time_slot'] = temp['hour_bin'].apply(lambda h: f"{h:02d}:00–{(h + 3) % 24:02d}:00")

# Reorder columns
temp = temp[['time_slot', 'bike_accidents', 'injured_bike_accidents', 'injured_share', 'fatal_bike_accidents', 'fatal_share']]
print(temp)

#%% Accidents by Borough
query = f"""
WITH bike_collisions AS (
    SELECT 
        borough,
        number_of_persons_injured,
        number_of_cyclist_injured,
        number_of_pedestrians_injured,
        number_of_cyclist_killed,
        number_of_pedestrians_killed,
        vehicle_type_code_1,
        vehicle_type_code_2,
        vehicle_type_code_3,
        vehicle_type_code_4,
        vehicle_type_code_5
    FROM collisions
    WHERE crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
),
bike_crashes AS (
    SELECT 
        borough,
        COUNT(*) AS bike_accidents,
        SUM(
            CASE 
                WHEN number_of_persons_injured >= 1 OR number_of_cyclist_injured >= 1 OR number_of_pedestrians_injured >= 1 
                THEN 1 ELSE 0 
            END
        ) AS injured_bike_accidents,
        SUM(
            CASE 
                WHEN number_of_cyclist_killed >= 1 OR number_of_pedestrians_killed >= 1 
                THEN 1 ELSE 0 
            END
        ) AS fatal_bike_accidents
    FROM bike_collisions
    WHERE 
        vehicle_type_code_1 IN {bike_tuple} OR
        vehicle_type_code_2 IN {bike_tuple} OR
        vehicle_type_code_3 IN {bike_tuple} OR
        vehicle_type_code_4 IN {bike_tuple} OR
        vehicle_type_code_5 IN {bike_tuple}
    GROUP BY borough
)
SELECT 
    borough,
    bike_accidents,
    injured_bike_accidents,
    fatal_bike_accidents,
    ROUND(injured_bike_accidents::decimal / NULLIF(bike_accidents, 0), 4) AS injured_share,
    ROUND(fatal_bike_accidents::decimal / NULLIF(bike_accidents, 0), 4) AS fatal_share
FROM 
    bike_crashes
ORDER BY 
    bike_accidents DESC;
"""

temp = pd.read_sql(query, engine)
print(temp)


#%% Accidents by Zip Code
query = f"""
WITH bike_collisions AS (
    SELECT 
        zip_code,
        number_of_persons_injured,
        number_of_cyclist_injured,
        number_of_pedestrians_injured,
        number_of_cyclist_killed,
        number_of_pedestrians_killed,
        vehicle_type_code_1,
        vehicle_type_code_2,
        vehicle_type_code_3,
        vehicle_type_code_4,
        vehicle_type_code_5
    FROM collisions
    WHERE crash_date >= '2024-01-01' AND crash_date < '2025-01-01'
),
bike_crashes AS (
    SELECT 
        zip_code,
        COUNT(*) AS bike_accidents,
        SUM(
            CASE 
                WHEN number_of_persons_injured >= 1 OR number_of_cyclist_injured >= 1 OR number_of_pedestrians_injured >= 1 
                THEN 1 ELSE 0 
            END
        ) AS injured_bike_accidents,
        SUM(
            CASE 
                WHEN number_of_cyclist_killed >= 1 OR number_of_pedestrians_killed >= 1 
                THEN 1 ELSE 0 
            END
        ) AS fatal_bike_accidents
    FROM bike_collisions
    WHERE 
        vehicle_type_code_1 IN {bike_tuple} OR
        vehicle_type_code_2 IN {bike_tuple} OR
        vehicle_type_code_3 IN {bike_tuple} OR
        vehicle_type_code_4 IN {bike_tuple} OR
        vehicle_type_code_5 IN {bike_tuple}
    GROUP BY zip_code
)
SELECT 
    zip_code,
    bike_accidents,
    injured_bike_accidents,
    fatal_bike_accidents,
    ROUND(injured_bike_accidents::decimal / NULLIF(bike_accidents, 0), 4) AS injured_share,
    ROUND(fatal_bike_accidents::decimal / NULLIF(bike_accidents, 0), 4) AS fatal_share
FROM 
    bike_crashes
ORDER BY 
    bike_accidents DESC;
"""

temp = pd.read_sql(query, engine)
print(temp)

## Plot Accidents by ZIP Code
# Drop NAs
temp = temp.dropna(subset=["zip_code"])

# Convert zip_code to string (important for matching shapefile)
temp["zip_code"] = temp["zip_code"].astype(int).astype(str)

# 3. Extract values
zip_codes = temp["zip_code"].tolist()
bike_accidents = temp["bike_accidents"].tolist()

# 4. Call the map function
sub_create_map.plot_zip_map(
    zip_codes=zip_codes,
    values=bike_accidents,
    output_name='bike_accidents_zip',
    value_label='bike_accidents',
    legend_label='Bike Accidents (2024)',
    plot_title='Bike Accidents by ZIP Code (2024)'
)
