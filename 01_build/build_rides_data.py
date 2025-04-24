### Download and build rides database

import os
import requests
import zipfile
import pandas as pd
import shutil
from sqlalchemy import create_engine
#%%

# === CONFIGURATION ===
ZIP_URL = "https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.csv.zip"
DEST_DIR = "./99_temp"
ZIP_PATH = os.path.join(DEST_DIR, "202401.zip")

DB_USER = "postgres"
DB_PASS = "axa_datascience"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "citibike"

# Setup database connection
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

#%%
# Download CityBike files
os.makedirs(DEST_DIR, exist_ok=True)
print("📥 Downloading...")
r = requests.get(ZIP_URL)
with open(ZIP_PATH, "wb") as f:
    f.write(r.content)
print(f"✅ Downloaded to {ZIP_PATH}")


#%%
## Extract CityBike files
# Structure: One or many .csv files in every .zip-file
print("📦 Extracting CSV(s)...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DEST_DIR)
csv_files = [os.path.join(DEST_DIR, f) for f in os.listdir(DEST_DIR) if f.endswith(".csv")]
print(f"🗃️ Found {len(csv_files)} CSV file(s):", csv_files)


# Load CSVs into database
for csv in csv_files:
    print(f"🚀 Loading {csv} into PostgreSQL in chunks...")

    chunksize = 100_000
    for chunk in pd.read_csv(csv, chunksize=chunksize):
        chunk.columns = [col.lower().strip().replace(" ", "_") for col in chunk.columns]
        chunk.to_sql("trips", engine, if_exists="append", index=False, method="multi")

    print("✅ Loaded in chunks.")

# Remove .zip File
os.remove(ZIP_PATH)

# Remove all extracted content (files + folders)
for item in os.listdir(DEST_DIR):
    path = os.path.join(DEST_DIR, item)
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


