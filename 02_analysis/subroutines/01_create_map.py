import zipfile
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import shutil


# Path to ZIP and extract dir (same directory)
zip_path = r"02_analysis\subroutines\input\map_nyc\nycb2020wi_25a.zip"
extract_dir = os.path.dirname(zip_path)

#%% Extract ZIP contents and remember what was extracted
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    extracted_files = zip_ref.namelist()  # store names of files inside the ZIP
    zip_ref.extractall(extract_dir)

# Find shapefile among extracted files
shapefile_path = next(
    (os.path.join(extract_dir, f) for f in extracted_files if f.endswith(".shp")),
    None
)

if not shapefile_path or not os.path.exists(shapefile_path):
    raise FileNotFoundError("No .shp file found in the extracted ZIP!")

# Load and plot
gdf = gpd.read_file(shapefile_path)
gdf.plot(figsize=(10, 10), edgecolor='black')

# Save the plot
plot_path = r"02_analysis\plots\nyc_map.png"
os.makedirs(os.path.dirname(plot_path), exist_ok=True)

plt.title("New York City Map")
plt.axis('off')
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"🖼️ Map saved to: {plot_path}")


# Cleanup only extracted files, keep the ZIP
for file in extracted_files:
    file_path = os.path.join(extract_dir, file)

    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

print("Extracted shapefiles deleted. ZIP file remains intact.")
