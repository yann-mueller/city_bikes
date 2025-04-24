### Download and build accident database
import os
import requests

# Define the URL for the CSV export
url = "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD"
#%%
# Define the output directory and file path
output_dir = "99_temp"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "nyc_motor_vehicle_collisions.csv")

# Download and save the CSV
response = requests.get(url)
if response.status_code == 200:
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Data successfully downloaded and saved to {output_path}")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")
