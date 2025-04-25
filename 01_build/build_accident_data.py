### Download and build accident database
import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text
import shutil

#%% Download configuration
url = "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD"
DEST_DIR = "./99_temp/"
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
#%%
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

#%% Load into database
chunksize = 1_000
print(f"Loading {FILENAME} into PostgreSQL database 'nypd' in chunks...")

chunks = pd.read_csv(FILE_PATH, chunksize=chunksize, low_memory=False)
total_chunks = sum(1 for _ in pd.read_csv(FILE_PATH, chunksize=chunksize, low_memory=False))  # for progress display

for i, chunk in enumerate(chunks, start=1):
    num_params = chunk.notna().sum().sum()
    print(f"Chunk {i}/{total_chunks} → Parameters: {num_params}")

    # Normalize column names
    chunk.columns = [col.lower().strip().replace(" ", "_") for col in chunk.columns]

    # Parse datetime fields
    if "crash_date" in chunk.columns:
        chunk["crash_date"] = pd.to_datetime(chunk["crash_date"], errors='coerce')

    if "crash_time" in chunk.columns:
        chunk["crash_time"] = pd.to_datetime(chunk["crash_time"], format="%H:%M", errors='coerce')

    # Filter for year 2024 only (Observations for some earlier years made problems and are not needed anyways right now)
    if "crash_date" in chunk.columns:
        chunk = chunk[chunk["crash_date"].dt.year >= 2010]
        print(f"Keeping {len(chunk)} rows from years > 2010")

    # Skip insert if chunk is now empty
    if chunk.empty:
        print(f"Chunk {i} had no >2010 entries, skipping.")
        continue

    # Insert into database (table name = collisions)
    try:
        chunk.to_sql("collisions", engine, if_exists="append", index=False, method="multi")
    except Exception as e:
        print(f"Chunk {i} failed! Params: {num_params}, shape: {chunk.shape}")
        print(e)
        # Save for debugging
        chunk.to_csv(f"debug_failed_chunk_{i}.csv", index=False)
        break

print("All chunks loaded.")


#%% Cleanup
print("Cleaning up...")
os.remove(FILE_PATH)
print(f"Removed file: {FILE_PATH}")

print("\nNYPD data loaded and cleaned up successfully.")

#%%
#with engine.connect() as conn:
#    conn.execute(text("TRUNCATE TABLE collisions;"))
#    conn.commit()

########################
### DEBUGING SCRIPTS ###
########################

#%%
#df = pd.read_csv("debug_failed_chunk_1572.csv", low_memory=False)
#print(df.shape)
#df.head()

#%% Find rows with many non-null fields (suspiciously "full" rows)
#row_non_nulls = df.notna().sum(axis=1)
#print(row_non_nulls.sort_values(ascending=False).head())

#%%
# Look for weird strings (null bytes, etc.)
#weird_strings = df.applymap(lambda x: isinstance(x, str) and '\x00' in x)
#print("Null bytes found in any cell?", weird_strings.any().any())


#%%
#engine = create_engine("postgresql://postgres:axa_datascience@localhost:5432/nypd")

#df = pd.read_csv("debug_failed_chunk_1572.csv", low_memory=False)
#df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]  # make sure it's normalized

#for idx, row in df.iterrows():
#    try:
#        print(f"Row {idx}/1000")
#        row_df = pd.DataFrame([row])
#        row_df.to_sql("collisions", engine, if_exists="append", index=False, method="multi")
#    except Exception as e:
#        print(f"❌ Row {idx} failed:")
#        print(e)
#        print(row)
#        break

#%%
# SQL query to inspect column types
#query = """
#SELECT column_name, data_type
#FROM information_schema.columns
#WHERE table_name = 'collisions';
#"""

# Run the query and show result
#temp = pd.read_sql(query, engine)
#print(temp)
