### Download and build accident database
import os
import requests
import pandas as pd
from sqlalchemy import create_engine
import shutil

# Download configuration
url = "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD"
DEST_DIR = "./99_temp"
FILENAME = "nyc_motor_vehicle_collisions.csv"
FILE_PATH = os.path.join(DEST_DIR, FILENAME)

# DB config (use NYPD database)
DB_USER = "postgres"
DB_PASS = "axa_datascience"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "nypd"

# Setup database engine
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Ensure temp directory exists
os.makedirs(DEST_DIR, exist_ok=True)

# Download the file
print("Downloading NYPD collision data...")
response = requests.get(url)
if response.status_code == 200:
    with open(FILE_PATH, "wb") as f:
        f.write(response.content)
    print(f"Data successfully saved to {FILE_PATH}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
    exit()

# Load into database
chunksize = 100_000
print(f"Loading {FILENAME} into PostgreSQL database 'nypd' in chunks...")

chunks = pd.read_csv(FILE_PATH, chunksize=chunksize, low_memory=False)
total_chunks = sum(1 for _ in pd.read_csv(FILE_PATH, chunksize=chunksize, low_memory=False))  # for progress display

for i, chunk in enumerate(chunks, start=1):
    print(f"Processing chunk {i}/{total_chunks}...")

    # Normalize column names
    chunk.columns = [col.lower().strip().replace(" ", "_") for col in chunk.columns]

    # Attempt to parse date/time fields
    if "crash_date" in chunk.columns:
        chunk["crash_date"] = pd.to_datetime(chunk["crash_date"], errors='coerce')
    if "crash_time" in chunk.columns:
        chunk["crash_time"] = pd.to_datetime(chunk["crash_time"], format="%H:%M", errors='coerce')

    # Insert into database (table name = collisions)
    chunk.to_sql("collisions", engine, if_exists="append", index=False, method="multi")

print("All chunks loaded.")

# Cleanup
print("Cleaning up...")
os.remove(FILE_PATH)
print(f"Removed file: {FILE_PATH}")

print("\nNYPD data loaded and cleaned up successfully.")
