### Download and build rides database

import os
import requests
import zipfile
import pandas as pd
import shutil
from sqlalchemy import create_engine
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

# Ensure temp directory exists
os.makedirs(DEST_DIR, exist_ok=True)

# Process each .csv file from CitiBike
for ZIP_URL in ZIP_URLS:
    zip_filename = os.path.basename(ZIP_URL)
    ZIP_PATH = os.path.join(DEST_DIR, zip_filename)

    print(f"\n📥 Downloading {zip_filename}...")
    r = requests.get(ZIP_URL)
    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)
    print(f"✅ Downloaded to {ZIP_PATH}")

    # === Extract all CSVs ===
    print("📦 Extracting CSV(s)...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DEST_DIR)
    csv_files = [os.path.join(DEST_DIR, f) for f in os.listdir(DEST_DIR) if f.endswith(".csv")]
    print(f"🗃️ Found {len(csv_files)} CSV file(s):", csv_files)

    # Load all CSVs into database (append mode)
    for csv in csv_files:
        print(f"🚀 Loading {csv} into PostgreSQL in chunks...")

        dtype_spec = {
            'start_station_id': str,
            'end_station_id': str
        }

        chunks = pd.read_csv(csv, chunksize=chunksize, low_memory=False, dtype=dtype_spec)
        total_chunks = sum(1 for _ in pd.read_csv(csv, chunksize=chunksize, low_memory=False))  # estimate total chunks

        for i, chunk in enumerate(chunks, start=1):
            print(f"📊 Processing chunk {i}/{total_chunks}...")

            # Normalize column names
            chunk.columns = [col.lower().strip().replace(" ", "_") for col in chunk.columns]

            # Enforce types for start/end time
            chunk['started_at'] = pd.to_datetime(chunk['started_at'], errors='coerce')
            chunk['ended_at'] = pd.to_datetime(chunk['ended_at'], errors='coerce')

            # Explicitly convert columns 5 and 7 (0-based index) to strings
            if len(chunk.columns) > 7:  # make sure the columns exist
                chunk.iloc[:, 5] = chunk.iloc[:, 5].astype(str)
                chunk.iloc[:, 7] = chunk.iloc[:, 7].astype(str)

            # Append to database
            chunk.to_sql("trips", engine, if_exists="append", index=False, method="multi")

        print("✅ All chunks loaded.")

    # Remove temporary .csv files
    print("🧹 Cleaning up...")
    os.remove(ZIP_PATH)
    for item in os.listdir(DEST_DIR):
        path = os.path.join(DEST_DIR, item)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    print("✅ Cleanup complete for this file.")

print("\n🎉 All files processed successfully.")


