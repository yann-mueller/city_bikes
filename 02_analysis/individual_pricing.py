### Individual Pricing Strategy
import os
import json
import requests
import zipfile
import shutil
import skfmm
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import pandas as pd
import xgboost as xgb
import importlib.util
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from rasterio.features import rasterize
from scipy.stats import rankdata
from affine import Affine
from xgboost import plot_importance
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine, text

# Path to the file you want to import from
file_path = os.path.join(os.getcwd(), "02_analysis", "subroutines", "sub_create_map.py")

# Load the module manually
spec = importlib.util.spec_from_file_location("sub_create_map", file_path)
sub_create_map = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub_create_map)

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
ax.set_title("NYC Transportation Network", fontsize=20)
ax.set_axis_off()

plt.tight_layout()
plt.savefig("02_analysis/plots/nyc_zip_roads.png")
plt.close()

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

#%% Assign ZIP Codes to Fastest Path
def build_zipcode_raster(gdf_zipcodes, X, Y):
    xmin, ymin, xmax, ymax = gdf_zipcodes.total_bounds

    resolution_x = X.shape[1]
    resolution_y = X.shape[0]

    x_res = (xmax - xmin) / resolution_x
    y_res = (ymax - ymin) / resolution_y
    transform = Affine.translation(xmin, ymin) * Affine.scale(x_res, y_res)

    shapes = [(geom, zip_code) for geom, zip_code in zip(gdf_zipcodes.geometry, gdf_zipcodes["MODZCTA"])]

    zipcode_raster = rasterize(
        shapes,
        out_shape=(resolution_y, resolution_x),
        transform=transform,
        fill=-1,  # Fill non-covered cells with -1
        dtype=int,
        all_touched=True
    )

    return zipcode_raster

zipcode_raster = build_zipcode_raster(gdf, X, Y)

#%% Find ZIP Codes along the optimal path
def coords_to_raster_indices(path_coords, X, Y):
    x = X[0, :]
    y = Y[:, 0]
    indices = []
    for lon, lat in path_coords:
        idx_x = np.argmin(np.abs(x - lon))
        idx_y = np.argmin(np.abs(y - lat))
        indices.append((idx_y, idx_x))
    return indices

# Get raster indices for path
path_indices = coords_to_raster_indices(path_coords, X, Y)

# Extract ZIP codes along path
zipcodes_along_path = [zipcode_raster[y, x] for y, x in path_indices]

# Remove invalid ZIPs (if path goes outside ZIP areas)
zipcodes_along_path = [zipcode for zipcode in zipcodes_along_path if zipcode != -1]

# Convert to pandas series
zipcodes_along_path = pd.Series([int(z) for z in zipcodes_along_path if z != 99999]).drop_duplicates().reset_index(drop=True)

# Unique ZIP codes on path
unique_zipcodes = list(set(zipcodes_along_path))

print("ZIP codes on optimal path:", unique_zipcodes)

#%% New Plot ZIP codes along the Optimal Path
# Select ZIPs from GeoPandas dataframe
selected_zipcodes = [int(z) for z in unique_zipcodes if z != 99999]  # Make sure they're normal integers
gdf_selected = gdf[gdf["MODZCTA"].astype(int).isin(selected_zipcodes)]

plt.figure(figsize=(8, 8))

# Plot the mask
plt.pcolormesh(X, Y, mask, shading='auto', cmap='gray_r')

# Plot the ZIP code boundaries you passed
gdf_selected.boundary.plot(ax=plt.gca(), edgecolor='green', linewidth=1.5)

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
plt.savefig("02_analysis/plots/path_on_raster_zip.png")
plt.close()

#%% Create new database table with the optimal path for every route
# Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/citibike")
#%%
create_table_query = """
DROP TABLE IF EXISTS station_pairs;

CREATE TABLE station_pairs AS
SELECT 
    LEAST(start_station_name, end_station_name) AS station_start_name,
    GREATEST(start_station_name, end_station_name) AS station_end_name,
    COUNT(*) AS rides_count,
    MIN(start_lat) AS start_lat,
    MIN(start_lng) AS start_lng,
    MIN(end_lat) AS end_lat,
    MIN(end_lng) AS end_lng,
    '[]'::text AS zip_codes
FROM trips
WHERE 
    start_station_name IS NOT NULL
    AND end_station_name IS NOT NULL
GROUP BY 
    station_start_name,
    station_end_name
ORDER BY 
    rides_count DESC;
"""

with engine.begin() as connection:
    connection.execute(text(create_table_query))

#%% Safety checks
query = "SELECT * FROM station_pairs LIMIT 10;"
temp = pd.read_sql(query, engine)

print(temp)

# Sum up the total number of rides
query = "SELECT SUM(rides_count) AS total_rides FROM station_pairs;"
temp = pd.read_sql(query, engine)

print("\nTotal number of rides:")
print(temp)

# Number of rows in the new table
query = "SELECT COUNT(*) AS num_pairs FROM station_pairs;"
temp = pd.read_sql(query, engine)

print("\nNumber of station pairs (observations):")
print(temp)

#%% Add ID to database table
with engine.begin() as conn:
    conn.execute(text("""
        ALTER TABLE station_pairs
        ADD COLUMN id SERIAL PRIMARY KEY;
    """))

#%% Find Optimal path for every route in our dataset
with engine.begin() as conn:
    # Only select first 10 rows for testing
    result = conn.execute(text("""
        SELECT id, station_start_name, station_end_name, start_lat, start_lng, end_lat, end_lng
        FROM station_pairs
        WHERE zip_codes = '[]'
        ORDER BY id;
    """))

    counter = 0

    for row in result:
        counter += 1

        # Extract coordinates
        start_lat = row.start_lat
        start_lng = row.start_lng
        end_lat = row.end_lat
        end_lng = row.end_lng

        # Skip rows with missing coordinates
        if None in (start_lat, start_lng, end_lat, end_lng):
            print(f"Skipping row {row.id} because coordinates are missing.")
            continue

        start_coord = (start_lng, start_lat)  # (lon, lat)
        end_coord = (end_lng, end_lat)

        # Your travel path and zip code extraction
        travel_time, path_coords = fast_marching_plotting(mask, X, Y, start_coord, end_coord)
        path_indices = coords_to_raster_indices(path_coords, X, Y)
        zipcodes_along_path = [zipcode_raster[y, x] for y, x in path_indices]
        zipcodes_along_path = [zipcode for zipcode in zipcodes_along_path if zipcode != -1]

        zipcodes_along_path = pd.Series(
            [int(z) for z in zipcodes_along_path if z != 99999]).drop_duplicates().reset_index(drop=True)
        unique_zipcodes = list(set(zipcodes_along_path))
        zipcodes_json = json.dumps(unique_zipcodes)

        # Update statement
        update_query = text("""
            UPDATE station_pairs
            SET zip_codes = :zipcodes
            WHERE id = :id
        """)

        conn.execute(update_query, {
            'zipcodes': zipcodes_json,
            'id': row.id
        })

        print(f"Processed station pair #{counter}")

#%% Table Check
query = "SELECT * FROM station_pairs LIMIT 100;"
temp = pd.read_sql(query, engine)

print(temp)

#%% Safety Check: Longest Ride
query = """
SELECT 
    id, 
    zip_codes,
    -- Count commas and add 1 (elements = commas + 1)
    LENGTH(zip_codes) - LENGTH(REPLACE(zip_codes, ',', '')) + 1 AS num_zipcodes
FROM station_pairs
WHERE zip_codes != '[]'
ORDER BY num_zipcodes DESC
LIMIT 1;
"""
result = pd.read_sql(query, engine)
print(result)

#%% Count Rides per ZIP Code Area
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS zip_code_counts (
            zip_code INTEGER PRIMARY KEY,
            count INTEGER
        );
    """))

    conn.execute(text("""
        WITH all_zip_codes AS (
            SELECT 
                TRIM(BOTH '[]' FROM zip_codes) AS cleaned_zipcodes
            FROM station_pairs
            WHERE zip_codes != '[]'
        ), exploded AS (
            SELECT 
                unnest(string_to_array(cleaned_zipcodes, ','))::INTEGER AS zip_code
            FROM all_zip_codes
        )
        INSERT INTO zip_code_counts (zip_code, count)
        SELECT 
            zip_code,
            COUNT(*) AS count
        FROM exploded
        GROUP BY zip_code
        ORDER BY zip_code;
    """))

#%% Table Check
query = "SELECT * FROM zip_code_counts;"
zip_rides = pd.read_sql(query, engine)

print(zip_rides)

#%% Add new Bike Accidents column
with engine.begin() as conn:
    conn.execute(text("""
        ALTER TABLE zip_code_counts
        ADD COLUMN bike_accidents_2024 INTEGER DEFAULT 0;
    """))

#%% Get Bike Accidents
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/nypd")

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
to_remove = ['Dart bike', 'dirt bike', 'Dirt Bike', 'DIRT BIKE', 'dirtbike', 'Dirtbike', 'DIRTBIKE', 'Moped bike',
             'gas bike', 'Minibike', 'Motorbike', 'motorbike', 'PEDAL BIKE', 'Dirt Bike']

# Filter the list
bike_strings_list = [item for item in bike_strings_list if item not in to_remove]

print(bike_strings_list)

# Create a tuple of strings for SQL IN clause
bike_tuple = tuple(bike_strings_list)


#%% Connect to the NYPD database
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
),
bike_crashes AS (
    SELECT 
        zip_code,
        COUNT(*) AS bike_accidents
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
    bike_accidents
FROM 
    bike_crashes
ORDER BY 
    bike_accidents DESC;
"""

bike_accidents_2024 = pd.read_sql(query, engine)
bike_accidents_2024 = bike_accidents_2024.dropna(subset=["zip_code"])

bike_accidents_2024["zip_code"] = bike_accidents_2024["zip_code"].astype(int)
print(bike_accidents_2024.head())

#%%
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/citibike")
zip_rides = pd.read_sql("SELECT * FROM zip_code_counts;", engine)
print(zip_rides.head())

#%% Merge Rides and Accidents
zip_rides = zip_rides.merge(
    bike_accidents_2024,
    how='left',
    on='zip_code'
)

# Fill missing accident counts with 0
zip_rides['bike_accidents_2024'] = zip_rides['bike_accidents_2024'].fillna(0).astype(int)
zip_rides['bike_accidents_2024'] = zip_rides['bike_accidents'].fillna(0).astype(int)
zip_rides = zip_rides.drop(columns=['bike_accidents'])

print(zip_rides.head())

# Store in database
zip_rides.to_sql('zip_code_counts_with_bikeaccidents', engine, if_exists='replace', index=False)

#%% Compute share and percentiles
zip_rides['bike_share_2024'] = zip_rides['bike_accidents_2024'] / zip_rides['count']

# Counts percentiles
zip_rides['count_percentile'] = rankdata(zip_rides['count'], method='average') / len(zip_rides['count']) * 100

# Bike share percentiles
zip_rides['bike_share_percentile'] = rankdata(zip_rides['bike_share_2024'], method='average') / len(zip_rides['bike_share_2024']) * 100

zip_rides['count_percentile'] = zip_rides['count_percentile'].round(2)
zip_rides['bike_share_percentile'] = zip_rides['bike_share_percentile'].round(2)


# Store in database
zip_rides.to_sql('zip_code_counts_with_bikeaccidents', engine, if_exists='replace', index=False)

#%%
temp = pd.read_sql("SELECT * FROM zip_code_counts_with_bikeaccidents;", engine)
print(temp.head())

#%% Plot Rides and Risk by ZIP Code
# Convert zip_code to string (important for matching shapefile)
zip_rides["zip_code"] = zip_rides["zip_code"].astype(int).astype(str)

# Extract values
zip_codes = zip_rides["zip_code"].tolist()
rides = zip_rides["count"].tolist()
accidents_rides_share = zip_rides["bike_share_2024"].tolist()
rides_percentile = zip_rides["count_percentile"].tolist()
accidents_rides_share_percentile = zip_rides["bike_share_percentile"].tolist()

sub_create_map.plot_zip_map(
    zip_codes=zip_codes,
    values=rides_percentile,
    output_name='rides_percentile_zip',
    value_label='rides_percentile',
    legend_label='Anzahl Fahrten (Perzentil)',
    plot_title='Fahrten nach ZIP Code (Perzentil, 2024)',
    fill_missing_with_neighbors = True
)

sub_create_map.plot_zip_map(
    zip_codes=zip_codes,
    values=accidents_rides_share_percentile,
    output_name='accidents_rides_share_percentile_zip',
    value_label='accidents_rides_share_percentile',
    legend_label='Unfallrate (Perzentil)',
    plot_title='Unfallrate nach ZIP Code (Perzentil, 2024)',
    fill_missing_with_neighbors = True
)

#%%
#######################
### RIDE PREDICTION ###
#######################
# Database connection
engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/citibike")

query = "SELECT * FROM trips LIMIT 10;"
temp = pd.read_sql(query, engine)

print(temp)

#%% Load Name to ZIP matching
query = "SELECT * FROM stations;"
temp_stations = pd.read_sql(query, engine)

temp_stations = temp_stations.drop_duplicates(subset=['station_name'], keep='first')

print(temp_stations)

#%% Load rides data
query = """
SELECT *
FROM trips;
"""
sample_df = pd.read_sql(query, engine)

#%% Load weather data
# NY coordinates
latitude = 40.7128
longitude = -74.0060

# Download weather data from Open-Meteo
# API URL
url = f"https://archive-api.open-meteo.com/v1/archive?" \
      f"latitude={latitude}&longitude={longitude}" \
      f"&start_date=2024-01-01&end_date=2024-12-31" \
      f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,snowfall_sum,windspeed_10m_max" \
      f"&timezone=America/New_York"

# API request
response = requests.get(url)
data = response.json()

weather_df = pd.DataFrame({
    "date": pd.to_datetime(data["daily"]["time"]),
    "precipitation_sum_mm": data["daily"]["precipitation_sum"],
    "temperature_max_c": data["daily"]["temperature_2m_max"],
    "temperature_min_c": data["daily"]["temperature_2m_min"],
    "snowfall_sum_cm": data["daily"]["snowfall_sum"],
    "windspeed_max_kmh": data["daily"]["windspeed_10m_max"]
})

sample_df['started_at'] = pd.to_datetime(sample_df['started_at'], format='mixed')

sample_df['date'] = sample_df['started_at'].dt.date

weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date

sample_df = sample_df.merge(weather_df, on='date', how='left')

sample_df = sample_df.drop(columns=['date'])

### Data Transformation
#%% Drop unnecessary columns
sample_df = sample_df.drop(columns=['ride_id', 'start_station_id', 'end_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng'])

#%% Create new e_bike dummy column
sample_df['e_bike'] = (sample_df['rideable_type'] == 'electric_bike').astype(int)
sample_df = sample_df.drop(columns=['rideable_type'])

#%% Create new member dummy column
sample_df['member'] = (sample_df['member_casual'] == 'member').astype(int)
sample_df = sample_df.drop(columns=['member_casual'])

#%% Create start date columns
sample_df['started_at'] = pd.to_datetime(sample_df['started_at'], format='mixed')

# Monthly dummies
for month in range(1, 13):
    sample_df[f'started_at_month_{month}'] = (sample_df['started_at'].dt.month == month).astype(int)

# Weekday dummies
for weekday in range(7):
    sample_df[f'started_at_weekday_{weekday}'] = (sample_df['started_at'].dt.weekday == weekday).astype(int)

# Hourly dummies
sample_df['started_at_hour'] = sample_df['started_at'].dt.hour
for hour in range(24):
    sample_df[f'started_at_hour_{hour}'] = (sample_df['started_at_hour'] == hour).astype(int)

# Drop helper columns
sample_df = sample_df.drop(columns=['started_at', 'started_at_hour'])

#%% Create end date columns
sample_df['ended_at'] = pd.to_datetime(sample_df['ended_at'], format='mixed')

# Monthly dummies
for month in range(1, 13):
    sample_df[f'ended_at_month_{month}'] = (sample_df['ended_at'].dt.month == month).astype(int)

# Weekday dummies
for weekday in range(7):
    sample_df[f'ended_at_weekday_{weekday}'] = (sample_df['ended_at'].dt.weekday == weekday).astype(int)

# Hourly dummies
sample_df['ended_at_hour'] = sample_df['ended_at'].dt.hour
for hour in range(24):
    sample_df[f'ended_at_hour_{hour}'] = (sample_df['ended_at_hour'] == hour).astype(int)

# Drop helper columns
sample_df = sample_df.drop(columns=['ended_at', 'ended_at_hour'])

#%% Replace station names with correspond ZIP code areas
sample_df = sample_df.merge(
    temp_stations.rename(columns={'station_name': 'start_station_name', 'zip_code': 'start_zip'}),
    on='start_station_name',
    how='left'
)

sample_df = sample_df.merge(
    temp_stations.rename(columns={'station_name': 'end_station_name', 'zip_code': 'end_zip'}),
    on='end_station_name',
    how='left'
)

sample_df = sample_df.dropna(subset=['start_zip', 'end_zip'])
sample_df = sample_df.reset_index(drop=True)

# Drop the station name columns
sample_df = sample_df.drop(columns=['start_station_name', 'end_station_name'])

# Get the current column order
cols = sample_df.columns.tolist()

# Reorder: start_zip, end_zip first, then the rest
new_order = ['start_zip', 'end_zip'] + [col for col in cols if col not in ['start_zip', 'end_zip']]
sample_df = sample_df[new_order]

#%% Prepare features and target
X = sample_df.drop('end_zip', axis=1)  # Features: everything except destination
y = sample_df['end_zip']               # Target: destination zip

#%% Label recoding
le = LabelEncoder()

# Fit encoder and transform y
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#%% Set up XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softprob',   # very important! -> probability output
    num_class=len(y.unique()),    # number of destination zip codes
    eval_metric='mlogloss',       # for multiclass classification
    use_label_encoder=False,      # newer versions of XGBoost
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    tree_method="hist"            # faster if your data is large
)

#%% Train model
model.fit(X_train, y_train)

#%% Predict probabilities for test set
y_pred_proba = model.predict_proba(X_test)

#%% Predict most likely destination
all_labels = list(range(len(le.classes_)))

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_pred_proba, labels=all_labels))

# Compute Top-N Hit Rates
top_ns = [1, 3, 5, 10, 20]
hit_rates = []

for n in top_ns:
    topn_pred = np.argsort(y_pred_proba, axis=1)[:, -n:]
    topn_hits = [y_test[i] in topn_pred[i] for i in range(len(y_test))]
    topn_hit_rate = np.mean(topn_hits)
    hit_rates.append(topn_hit_rate)
    print(f"Top-{n} Hit Rate: {topn_hit_rate:.4f}")

# Plot Hit Rates
plt.figure(figsize=(12, 6))
plt.bar([f"Top-{n}" for n in top_ns], hit_rates, color='royalblue')

plt.ylim(0, 1)
plt.ylabel("Hit Rate", fontsize=18)  # correct: fontsize not size
plt.title("Top-N Hit Rates", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("02_analysis/plots/hit_rates.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("02_analysis/plots/confusion_matrix.png")
plt.close()

# Feature Importance
fig, ax = plt.subplots(figsize=(10, 12))

plot_importance(
    model,
    ax=ax,
    importance_type='gain',
    max_num_features=20,
    title="Feature Importance (Top 20)",
    xlabel="Importance (Gain)",
    show_values=False,
    grid=False
)

ax.title.set_fontsize(16)
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig("02_analysis/plots/plot_importance.png")
plt.close()
#%% Prediction Illustration
# Create an empty DataFrame with the same columns as your model expects
X_example = pd.DataFrame(columns=X_train.columns)

X_example.loc[0] = 0

# Start zip code
X_example.at[0, 'start_zip'] = 110281  # AXA XL Office ZIP code

# Weather data
X_example.at[0, 'precipitation_sum_mm'] = 0.0
X_example.at[0, 'temperature_max_c'] = 25
X_example.at[0, 'temperature_min_c'] = 18
X_example.at[0, 'snowfall_sum_cm'] = 0.0
X_example.at[0, 'windspeed_max_kmh'] = 10

# Member status
X_example.at[0, 'member'] = 1  # 1 if member, 0 if casual

# Time features
# Set specific month, weekday and hour manually:
X_example.at[0, 'started_at_month_4'] = 1  # April
X_example.at[0, 'started_at_weekday_2'] = 1  # Wednesday
X_example.at[0, 'started_at_hour_12'] = 1  # 12AM

X_example.at[0, 'ended_at_month_4'] = 1  # April
X_example.at[0, 'ended_at_weekday_2'] = 1  # Wednesday
X_example.at[0, 'ended_at_hour_12'] = 1  # 12AM

y_pred_proba_example = model.predict_proba(X_example)[0]  # get the first (and only) row

# Map class indices back to zip codes
predicted_zip_codes = le.inverse_transform(np.arange(len(y_pred_proba_example)))

# Prepare lists (make sure to use *_example everywhere)
zip_codes_list = [str(z) for z in predicted_zip_codes]
cleaned_zip_codes_list = [str(int(float(z))) for z in zip_codes_list]

probabilities_list = (np.array(y_pred_proba_example) * 100).tolist()

# Plot
sub_create_map.plot_zip_map(
    zip_codes=cleaned_zip_codes_list,
    values=probabilities_list,
    output_name='predicted_destination_probs',
    value_label='probabilities_list',
    legend_label='Wahrscheinlichkeit (%)',
    plot_title='Prediction ZIP Code Endstation',
    extent=(-74.05, -73.93, 40.68, 40.81),
    dot_color='red'
)