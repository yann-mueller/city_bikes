### Individual Pricing Strategy
import os
import requests
import zipfile
import pandas as pd
import shutil
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine, text


#%%
##################################
### 2D Fast Marching Algorithm ###
##################################
# Load New York Map
csv_path = "02_analysis/subroutines/input/map_nyc/Modified_Zip_Code_Tabulation_Areas_MODZCTA_20250425.csv"
df = pd.read_csv(csv_path)

# Detect WKT geometry column
geometry_col = None
for col in df.columns:
    if df[col].astype(str).str.startswith('MULTI').any() or df[col].astype(str).str.startswith('POLYGON').any():
        geometry_col = col
        break
if not geometry_col:
    raise ValueError("No WKT geometry column found in CSV.")

df[geometry_col] = df[geometry_col].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col], crs="EPSG:4326")

#%% Load New York Road Map
url = "https://data.cityofnewyork.us/api/views/inkn-q76z/rows.csv?accessType=DOWNLOAD"  # Example URL for centerlines CSV
local_path = "99_temp/nyc_centerlines.csv"

# Download
response = requests.get(url)
with open(local_path, 'wb') as f:
    f.write(response.content)

print(f"Downloaded file to {local_path}")

# Read the .csv file
df_roads = pd.read_csv(local_path)

# Delete the .csv file
os.remove(local_path)
print(f"Deleted file {local_path}")

#%% Detect WKT geometry column
geometry_col = 'the_geom'
for col in df_roads.columns:
    if df_roads[col].astype(str).str.startswith('MULTI').any() or df_roads[col].astype(str).str.startswith('POLYGON').any():
        geometry_col = col
        break
if not geometry_col:
    raise ValueError("No WKT geometry column found in CSV.")

df_roads[geometry_col] = df_roads[geometry_col].apply(wkt.loads)
gdf_roads = gpd.GeoDataFrame(df_roads, geometry=df_roads[geometry_col], crs="EPSG:4326")

#%% Create plot
fig, ax = plt.subplots(figsize=(12, 12))

gdf.boundary.plot(ax=ax, color='black', linewidth=0.5)
gdf_roads.plot(ax=ax, linewidth=0.2, color='red', alpha=0.7)

# Styling
ax.set_title("NYC Road Network", fontsize=16)
ax.set_axis_off()

plt.tight_layout()
plt.savefig("02_analysis/plots/nyc_zip_roads.png")
plt.close()