### Download and build rides database
import os
import requests
import zipfile
import pandas as pd
import shutil
from sqlalchemy import create_engine, text
#%%
# Configuration
# URLs of the 2024 data
ZIP_URLS = [
    #"https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.csv.zip",
    #"https://s3.amazonaws.com/tripdata/202402-citibike-tripdata.csv.zip",
    #"https://s3.amazonaws.com/tripdata/202403-citibike-tripdata.csv.zip",
    #"https://s3.amazonaws.com/tripdata/202404-citibike-tripdata.csv.zip",
    #"https://s3.amazonaws.com/tripdata/202405-citibike-tripdata.zip",
    #"https://s3.amazonaws.com/tripdata/202406-citibike-tripdata.zip",
    #"https://s3.amazonaws.com/tripdata/202407-citibike-tripdata.zip",
    #"https://s3.amazonaws.com/tripdata/202408-citibike-tripdata.zip",
    #"https://s3.amazonaws.com/tripdata/202409-citibike-tripdata.zip",
    "https://s3.amazonaws.com/tripdata/202410-citibike-tripdata.zip",
    #"https://s3.amazonaws.com/tripdata/202411-citibike-tripdata.zip",
    #"https://s3.amazonaws.com/tripdata/202412-citibike-tripdata.zip"
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

            # Remove any 'unnamed' columns that sneak in from CSV index, and make a fresh copy to avoid SettingWithCopyWarning
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
    print("🧹 Cleaning up...")
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

