### Download and build rides database
import os
import requests
import zipfile
import pandas as pd
import shutil
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from sqlalchemy import create_engine, text
#%%
# Configuration
# URLs of the 2024 data
ZIP_URLS = [
    "https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.csv.zip",
    "https://s3.amazonaws.com/tripdata/202402-citibike-tripdata.csv.zip",
    "https://s3.amazonaws.com/tripdata/202403-citibike-tripdata.csv.zip",
    "https://s3.amazonaws.com/tripdata/202404-citibike-tripdata.csv.zip",
    "https://s3.amazonaws.com/tripdata/202405-citibike-tripdata.zip",
    "https://s3.amazonaws.com/tripdata/202406-citibike-tripdata.zip",
    "https://s3.amazonaws.com/tripdata/202407-citibike-tripdata.zip",
    "https://s3.amazonaws.com/tripdata/202408-citibike-tripdata.zip",
    "https://s3.amazonaws.com/tripdata/202409-citibike-tripdata.zip",
    "https://s3.amazonaws.com/tripdata/202410-citibike-tripdata.zip",
    "https://s3.amazonaws.com/tripdata/202411-citibike-tripdata.zip",
    "https://s3.amazonaws.com/tripdata/202412-citibike-tripdata.zip"
]

DEST_DIR = "./99_temp"

# Database config credentials
DB_USER = "postgres"
DB_PASS = "axa_datascience"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "citibike"

# Setup database connection
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'trips'
    """))
    db_columns = [row[0] for row in result]

#%%
# Ensure temp directory exists
os.makedirs(DEST_DIR, exist_ok=True)

# Define size of chunks in order to process the .csv files
chunksize = 1000

# Process each .csv file from CitiBike
for ZIP_URL in ZIP_URLS:
    zip_filename = os.path.basename(ZIP_URL)
    ZIP_PATH = os.path.join(DEST_DIR, zip_filename)

    print(f"\nDownloading {zip_filename}...")
    r = requests.get(ZIP_URL)
    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)
    print(f"Downloaded to {ZIP_PATH}")

    # Extract all .csv files
    print("Extracting .csv files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DEST_DIR)

    # First: Check for .csv files directly in DEST_DIR (For some months the .csv files are located in a subfolder)
    csv_files = [os.path.join(DEST_DIR, f) for f in os.listdir(DEST_DIR) if f.endswith(".csv")]

    # If none found, search recursively in subfolders
    if not csv_files:
        print("No .csv files found directly in DEST_DIR. Searching in subfolders...")
        csv_files = [os.path.join(root, file)
                     for root, _, files in os.walk(DEST_DIR)
                     for file in files if file.endswith(".csv")]

    print(f"Found {len(csv_files)} .csv files:", csv_files)

    # Load all .csv files into database
    for csv in csv_files:
        print(f"Loading {csv} into PostgreSQL in chunks...")

        dtype_spec = {
            'start_station_id': str,
            'end_station_id': str
        }

        with open(csv) as f:
            total_lines = sum(1 for _ in f) - 1
        total_chunks = (total_lines // chunksize) + 1

        chunks = pd.read_csv(csv, chunksize=chunksize, low_memory=False, dtype=dtype_spec)
        total_chunks = sum(1 for _ in pd.read_csv(csv, chunksize=chunksize, low_memory=False))  # estimate total chunks

        for i, chunk in enumerate(pd.read_csv(csv, chunksize=chunksize, low_memory=False, dtype=dtype_spec), start=1):
            num_params = chunk.notna().sum().sum()
            print(f"Chunk {i} → Parameters: {num_params}")

            # Normalize column names
            chunk.columns = [col.lower().strip().replace(" ", "_") for col in chunk.columns]

            # Drop duplicate columns
            chunk = chunk.loc[:, ~chunk.columns.duplicated()]

            # Remove any 'unnamed' columns
            chunk = chunk.loc[:, ~chunk.columns.str.contains("^unnamed", case=False)].copy()

            # Enforce datetime conversion
            chunk['started_at'] = pd.to_datetime(chunk['started_at'], errors='coerce')
            chunk['ended_at'] = pd.to_datetime(chunk['ended_at'], errors='coerce')

            # Explicitly convert columns 5 and 7 (0-based index) to string safely using .loc and column names
            col_list = chunk.columns.tolist()
            if len(col_list) > 7:
                chunk.loc[:, col_list[5]] = chunk[col_list[5]].astype(str)
                chunk.loc[:, col_list[7]] = chunk[col_list[7]].astype(str)

            # Append to database
            try:
                # Keep only columns that exist in the database
                chunk = chunk[[col for col in chunk.columns if col in db_columns]]
                chunk.to_sql("trips", engine, if_exists="append", index=False, method="multi")
            except Exception as e:
                print(f"Chunk {i} failed! Params: {num_params}, shape: {chunk.shape}")
                print(e)
                # Save for debugging
                chunk.to_csv(f"debug_failed_chunk_{i}.csv", index=False)
                break

        print("All chunks loaded.")

    del chunks

    # Remove temporary .csv files
    print("Cleaning up...")
    os.remove(ZIP_PATH)
    for item in os.listdir(DEST_DIR):
        path = os.path.join(DEST_DIR, item)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    print("Cleanup complete for this file.")

print("\nAll files processed successfully.")

#%% Remove potential duplicate observations
with engine.begin() as conn:
    print("Removing duplicates from 'trips' based on ride_id...")

    # Create a deduplicated version of the table
    conn.execute(text("""
        CREATE TABLE trips_clean AS
        SELECT DISTINCT ON (ride_id) *
        FROM trips
        ORDER BY ride_id, started_at;
    """))

    # Drop the original 'trips' table
    conn.execute(text("DROP TABLE trips;"))

    # Rename the cleaned table back to 'trips'
    conn.execute(text("ALTER TABLE trips_clean RENAME TO trips;"))

    # Re-create index to improve performance
    conn.execute(text("CREATE INDEX idx_ride_id ON trips(ride_id);"))

    print("Duplications removed and table updated.")

#%% Add ZIP codes to every observation
# Create dataframe with all CitiBike stations and longitude and latitude
# Average lng and lat for every station due to slight GPS variations
query = """
SELECT 
    station_id,
    station_name,
    AVG(latitude) AS avg_latitude,
    AVG(longitude) AS avg_longitude
FROM (
    SELECT 
        start_station_id AS station_id,
        start_station_name AS station_name,
        start_lat AS latitude,
        start_lng AS longitude
    FROM trips

    UNION ALL

    SELECT 
        end_station_id AS station_id,
        end_station_name AS station_name,
        end_lat AS latitude,
        end_lng AS longitude
    FROM trips
) AS all_stations
WHERE station_id IS NOT NULL
GROUP BY station_id, station_name
ORDER BY station_name;
"""

stations = pd.read_sql(query, engine)
print(stations)

#%% Diagnostics
stations.isna().sum()

stations[stations['station_id'] == 'nan']

# Replace string 'nan' with real np.nan
stations.replace('nan', pd.NA, inplace=True)

# Drop all rows where station_name or station_id is missing
stations = stations.dropna(subset=['station_name', 'station_id']).reset_index(drop=True)

# Check for duplicate stations
stations['station_name'].duplicated().sum()

stations = stations.drop(columns=['station_id'])

#%% Read shapefile
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

#%% Add ZIP Code to stations df
stations['geometry'] = stations.apply(lambda row: Point(row['avg_longitude'], row['avg_latitude']), axis=1)
stations_gdf = gpd.GeoDataFrame(stations, geometry='geometry', crs="EPSG:4326")

# Spatial join with ZIP code polygons
stations_with_zip = gpd.sjoin(stations_gdf, gdf[['MODZCTA', 'geometry']], how='left', predicate='within')
stations['zip_code'] = stations_with_zip['MODZCTA'].values

print(stations.head())

#%% Add ZIP Codes to database
stations = stations[['station_name', 'zip_code']]

stations.to_sql('stations', con=engine, if_exists='replace', index=False)
