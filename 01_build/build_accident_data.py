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

# List of chunks that created problems and could not load into the database
problematic_chunks = []

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

    # Filter for year 2010–2024
    if "crash_date" in chunk.columns:
        chunk = chunk[(chunk["crash_date"].dt.year >= 2010) & (chunk["crash_date"].dt.year <= 2024)].copy()
        print(f"Keeping {len(chunk)} rows from years 2010 to 2024")

    # Skip empty chunks
    if chunk.empty:
        print(f"Chunk {i} had no valid rows, skipping.")
        continue

    # Columns that must be numeric
    numeric_cols = [
        "number_of_persons_injured", "number_of_persons_killed",
        "number_of_pedestrians_injured", "number_of_pedestrians_killed",
        "number_of_cyclist_injured", "number_of_cyclist_killed",
        "number_of_motorist_injured", "number_of_motorist_killed",
        "latitude", "longitude"
    ]

    # Clean numeric columns
    chunk[numeric_cols] = chunk[numeric_cols].replace(r'^\s*$', pd.NA, regex=True)
    chunk[numeric_cols] = chunk[numeric_cols].apply(pd.to_numeric, errors='coerce')
    chunk.dropna(subset=numeric_cols, inplace=True)

    if chunk.empty:
        print(f"Chunk {i} had no valid numeric rows, skipping.")
        continue

    # Insert into DB
    try:
        chunk.to_sql("collisions", engine, if_exists="append", index=False, method="multi")
    except Exception as e:
        print(f"Chunk {i} failed! Params: {num_params}, shape: {chunk.shape}")
        print(e)
        chunk.to_csv(f"debug_failed_chunk_{i}.csv", index=False)
        problematic_chunks.append(i)
        continue

print("All chunks loaded.")
if problematic_chunks:
    print(f"The following chunks failed and were skipped: {problematic_chunks}")
else:
    print("No problematic chunks.")




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
