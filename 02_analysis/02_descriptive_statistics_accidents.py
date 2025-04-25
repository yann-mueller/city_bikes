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

#%% Summarizing table & Plot
# Create a tuple of strings for SQL IN clause
bike_tuple = tuple(bike_strings_list)

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