import zipfile
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shutil
from shapely import wkt



#%% Load the CSV
def plot_zip_map(
        zip_codes: list,
        values: list,
        output_name: str,
        csv_path: str = "02_analysis\subroutines\input\map_nyc\Modified_Zip_Code_Tabulation_Areas_MODZCTA_20250425.csv",
        zip_column: str = "MODZCTA",
        value_label: str = "value"
):
    """
    Plot a choropleth map of NYC zip code areas based on a CSV with geometry.

    Parameters:
    - csv_path: Path to CSV with zip geometries.
    - zip_codes: List of zip codes to color.
    - values: List of values to use for color scale.
    - output_path: Path to save the output PNG.
    - zip_column: Column in the CSV corresponding to ZIP codes.
    - value_label: Name of the value column (for legend).
    """

    # Load CSV
    df = pd.read_csv(csv_path)
    df[value_label] = pd.NA
    output_path = f"02_analysis/plots/{output_name}.png",

    # Detect WKT geometry column
    geometry_col = None
    for col in df.columns:
        if df[col].astype(str).str.startswith('MULTI').any() or df[col].astype(str).str.startswith('POLYGON').any():
            geometry_col = col
            break

    if not geometry_col:
        raise ValueError("No WKT geometry column found in CSV.")

    # Convert geometry
    df[geometry_col] = df[geometry_col].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=df[geometry_col], crs="EPSG:4326")

    # Merge with provided values
    zip_value_df = pd.DataFrame({zip_column: zip_codes, value_label: values})
    merged = gdf.merge(zip_value_df, on=zip_column, how="left")

    # Create output folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Plot with color scale
    ax = merged.plot(
        column=value_label,
        cmap="viridis",
        figsize=(10, 10),
        edgecolor="black",
        legend=True,
        missing_kwds={"color": "lightgrey", "label": "No data"}
    )

    plt.title("NYC Zip Code Map (colored by value)")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"🖼️ Map saved to: {output_path}")

