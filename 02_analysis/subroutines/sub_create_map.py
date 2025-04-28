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
        csv_path: str = "02_analysis/subroutines/input/map_nyc/Modified_Zip_Code_Tabulation_Areas_MODZCTA_20250425.csv",
        zip_column: str = "MODZCTA",
        value_label: str = "value",
        plot_title: str = "NYC Zip Code Map (colored by value)",
        legend_label: str = "Value",
        extent: tuple = None,
        dot_color: str = None,
        fill_missing_with_neighbors: bool = False
):
    """
    Plot a choropleth map of NYC zip code areas based on a CSV with geometry.

    Parameters:
    - zip_codes: List of zip codes to color.
    - values: List of values to use for color scale.
    - output_name: Filename (without .png extension) for saving the plot.
    - csv_path: Path to ZIP geometry CSV.
    - zip_column: Column in the shapefile for ZIP code.
    - value_label: Column name for coloring.
    - plot_title: Title displayed above the map.
    """

    # Load and parse
    df = pd.read_csv(csv_path)
    output_path = f"02_analysis/plots/{output_name}.png"

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

    # Merge with input values
    gdf[zip_column] = gdf[zip_column].astype(str)
    zip_value_df = pd.DataFrame({zip_column: zip_codes, value_label: values})
    zip_value_df[zip_column] = zip_value_df[zip_column].astype(str)
    merged = gdf.merge(zip_value_df, on=zip_column, how="left")

    # 🆕 Fill missing ZIP codes based on neighbors
    if fill_missing_with_neighbors:
        print("Filling missing ZIP codes based on neighbors...")

        # Create a spatial index to speed up neighbor search
        gdf_sindex = gdf.sindex

        for idx, row in merged[merged[value_label].isna()].iterrows():
            # Find neighboring geometries
            possible_neighbors_idx = list(gdf_sindex.intersection(row.geometry.bounds))
            possible_neighbors = gdf.iloc[possible_neighbors_idx]

            # True neighbors = geometries that actually touch
            true_neighbors = possible_neighbors[possible_neighbors.geometry.touches(row.geometry)]

            # Only neighbors that have a known value
            neighbors_with_values = merged.loc[true_neighbors.index]
            neighbors_with_values = neighbors_with_values[neighbors_with_values[value_label].notna()]

            if not neighbors_with_values.empty:
                # Calculate mean value of the neighbors
                avg_value = neighbors_with_values[value_label].mean()
                merged.at[idx, value_label] = avg_value
                print(f"Filled ZIP {row[zip_column]} with neighbor average {avg_value:.2f}")
            else:
                print(f"No valid neighbors found for ZIP {row[zip_column]} — remains missing.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 10))
    color_map = merged.plot(
        column=value_label,
        cmap="viridis",
        edgecolor="black",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "No data"}
    )

    # Title
    ax.set_title(plot_title, fontsize=18)
    ax.axis('off')

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])  # xmin, xmax
        ax.set_ylim(extent[2], extent[3])  # ymin, ymax

    if dot_color is not None:
        start = (-74.0145498363001, 40.71253895687167)  # AXA XL Office
        end = (-73.78037574173788, 40.59253271869781)  # Rockaway Beach

        # Define coordinates of the two points (longitude, latitude)
        points = [start, end]
        labels = ['AXA XL Office  ', '  Rockaway\nBeach']
        alignments = [{'ha': 'right', 'va': 'center', 'xytext': (-5, 0)},
                      {'ha': 'center', 'va': 'top', 'xytext': (0, -10)}]

        # Plot points with adjusted labels
        for (lon, lat), label, align in zip(points, labels, alignments):
            ax.scatter(lon, lat, color=dot_color, s=50, zorder=5)
            ax.annotate(label,
                        xy=(lon, lat),
                        xytext=align['xytext'],
                        textcoords='offset points',
                        fontsize=12,
                        color=dot_color,
                        weight='bold',
                        ha=align['ha'],
                        va=align['va'])

    # Adjust colorbar
    cbar = color_map.get_figure().get_axes()[-1]
    cbar.set_ylabel(legend_label, fontsize=14)
    cbar.tick_params(labelsize=12)

    # Shrink colorbar height by 50%
    box = cbar.get_position()
    cbar.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.5])

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Map saved to: {output_path}")



