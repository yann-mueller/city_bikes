import zipfile
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shutil
from shapely import wkt


# Path to the CSV you want to visualize
csv_path = r"02_analysis\subroutines\input\map_nyc\Modified_Zip_Code_Tabulation_Areas__MODZCTA__20250425.csv"

#%% Load the CSV
df = pd.read_csv(csv_path)

# Try to detect if a WKT-style geometry column exists
geometry_col = None
for col in df.columns:
    if df[col].astype(str).str.startswith('MULTI').any() or df[col].astype(str).str.startswith('POLYGON').any():
        geometry_col = col
        break

if geometry_col:
    # If WKT geometry exists, convert to GeoDataFrame
    df[geometry_col] = df[geometry_col].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col], crs="EPSG:4326")
else:
    # If no geometry, fall back to lat/lng columns
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()), None)

    if lat_col is None or lon_col is None:
        raise ValueError("Could not find geometry or lat/lon columns in the CSV.")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )

# Save the map to plots folder
plot_path = r"02_analysis\plots\nyc_map_from_csv.png"
os.makedirs(os.path.dirname(plot_path), exist_ok=True)

gdf.plot(figsize=(10, 10), edgecolor='black')
plt.title("NYC Map from CSV Geometry")
plt.axis('off')
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"🖼️ Map saved to: {plot_path}")

#%%
import os
print(os.getcwd())

