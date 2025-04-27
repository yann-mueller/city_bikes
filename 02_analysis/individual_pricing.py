### Individual Pricing Strategy
import os
import requests
import zipfile
import pandas as pd
import shutil
import skfmm
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from affine import Affine
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine, text

#%%
########################
### NYC Road Network ###
########################
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
fig, ax = plt.subplots(figsize=(8, 8))

# Plot boundaries and roads
gdf.boundary.plot(ax=ax, color='black', linewidth=0.5)
gdf_roads.plot(ax=ax, linewidth=0.2, color='blue', alpha=0.7)

start = (-74.0145498363001, 40.71253895687167)  # AXA XL Office
end = (-73.78037574173788, 40.59253271869781)   # Rockaway Beach

# Define coordinates of the two points (longitude, latitude)
points = [start, end]
labels = ['AXA XL Office  ', '  Rockaway\nBeach']
alignments = [{'ha':'right', 'va':'center', 'xytext':(-5, 0)},
              {'ha':'center', 'va':'top', 'xytext':(0, -10)}]

# Plot points with adjusted labels
for (lon, lat), label, align in zip(points, labels, alignments):
    ax.scatter(lon, lat, color='red', s=50, zorder=5)
    ax.annotate(label,
                xy=(lon, lat),
                xytext=align['xytext'],
                textcoords='offset points',
                fontsize=12,
                color='red',
                weight='bold',
                ha=align['ha'],
                va=align['va'])

# Styling adjustments
ax.set_title("NYC Road Network", fontsize=20)
ax.set_axis_off()

plt.tight_layout()
plt.savefig("02_analysis/plots/nyc_zip_roads.png")
plt.close()

#%%
##################################
### 2D Fast Marching Algorithm ###
##################################
#%% Build Raster Mask
def build_raster_mask(gdf_streets, resolution=200):
    xmin, ymin, xmax, ymax = gdf_streets.total_bounds

    width = resolution
    height = resolution
    x_res = (xmax - xmin) / width
    y_res = (ymax - ymin) / height

    transform = Affine.translation(xmin, ymin) * Affine.scale(x_res, y_res)

    # Create shapes for rasterization
    shapes = ((geom.buffer(0.0005), 1) for geom in gdf_streets.geometry if geom is not None)

    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=100,    # Cells without road
        dtype=np.float32,
        all_touched=True  # Consider any touched cell as "on road"
    )

    # Create coordinate grid (X,Y)
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)

    print("Raster mask ready.")

    return mask, X, Y

#%% Compute fast marching algorithm
def fast_marching_plotting(mask, X, Y, start_coord, end_coord):
    """
    mask: 2D array with speed values
    X, Y: 2D meshgrid arrays (lon, lat)
    start_coord: (lon, lat) tuple
    end_coord: (lon, lat) tuple
    """
    resolution = mask.shape[0]

    x = np.linspace(X.min(), X.max(), resolution)
    y = np.linspace(Y.min(), Y.max(), resolution)

    # Locate start index
    start_x, start_y = start_coord
    start_idx = (np.argmin(np.abs(x - start_x)), np.argmin(np.abs(y - start_y)))

    # Initialize the front
    phi = np.ones_like(mask)
    phi[start_idx[1], start_idx[0]] = -1

    # Fast Marching
    travel_time = skfmm.travel_time(phi, speed=mask)

    # Locate end index
    end_x, end_y = end_coord
    end_idx = (np.argmin(np.abs(x - end_x)), np.argmin(np.abs(y - end_y)))

    # Now backtrack path
    print(f"Backtracking path...")
    path = [end_idx]
    current = end_idx

    # Search 8-neighborhood (diagonals allowed)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for _ in range(10000):  # Max steps to avoid infinite loop
        neighbors = []
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < resolution and 0 <= ny < resolution:
                neighbors.append(((nx, ny), travel_time[ny, nx]))

        if not neighbors:
            break

        # Move to neighbor with lowest travel time
        next_idx, next_time = min(neighbors, key=lambda t: t[1])

        if next_idx == current:  # Stuck
            break

        path.append(next_idx)
        current = next_idx

        if current == start_idx:
            break

    print(f"Path found with {len(path)} steps.")

    # Convert grid indices to lon/lat
    path_coords = [(x[i], y[j]) for i, j in path]

    return travel_time, path_coords

#%% Build mask
mask, X, Y = build_raster_mask(gdf_roads, resolution=200)

plt.figure(figsize=(8, 8))
plt.pcolormesh(X, Y, mask, shading='auto', cmap='gray_r')

# Remove ticks and labels
plt.xticks([])
plt.yticks([])
plt.gca().set_axis_off()

# Title only
plt.title('Raster Mask Visualization', fontsize=20)

plt.tight_layout()
plt.savefig("02_analysis/plots/baseline_raster.png")
plt.close()

#%% Run 2D Fast Marching Algorithm for illustration
start = (-74.0145498363001, 40.71253895687167)  # AXA XL Office
end = (-73.78037574173788, 40.59253271869781)    # Rockaway Beach

# Invert mask (different to 2D Fast Marching Packages in R)
mask = np.where(mask == 1, 100, 1)  # roads=100, non-roads=1

#%%
travel_time, path_coords = fast_marching_plotting(mask, X, Y, start, end)

plt.figure(figsize=(8, 8))

# Plot the mask
plt.pcolormesh(X, Y, mask, shading='auto', cmap='gray_r')

# Extract path coordinates into separate longitude and latitude lists
path_lon, path_lat = zip(*path_coords)

# Plot traveled path clearly in red
plt.plot(path_lon, path_lat, color='red', linewidth=2)

# Optional: clearly mark start and end points
plt.scatter([start[0], end[0]], [start[1], end[1]], color='red', s=70, zorder=5)

# Remove ticks and labels for clean visualization
plt.xticks([])
plt.yticks([])
plt.gca().set_axis_off()

# Title
plt.title('Optimal Path via Fast Marching', fontsize=20)

plt.tight_layout()
plt.savefig("02_analysis/plots/path_on_raster.png")
plt.close()

