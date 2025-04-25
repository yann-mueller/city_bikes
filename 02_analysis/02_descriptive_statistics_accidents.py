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

