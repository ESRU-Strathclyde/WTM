import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

# ===================================================
# USER SETTINGS
# ===================================================

INPUT_FILE = "classified_points.txt"
OUTPUT_FILE = "landcover_classification.tif"

CRS = "EPSG:32723"  # UTM Zone 23S


# ===================================================
# READ CLASSIFICATION FILE
# ===================================================

print("Reading classification file...")

df = pd.read_csv(
    INPUT_FILE,
    sep=r"\s+",
    engine="python"
)

x = df["X[m]"].values
y = df["Y[m]"].values
classification = df["Classification"].values

print(f"Number of points: {len(df):,}")

# ===================================================
# GRID DEFINITION
# ===================================================

x_unique = np.sort(np.unique(x))
y_unique = np.sort(np.unique(y))

nx = len(x_unique)
ny = len(y_unique)

print(f"Grid size: {nx} x {ny}")

if nx < 2 or ny < 2:
    raise ValueError(
        "Unable to determine grid spacing. "
        "Need at least 2 unique X values and 2 unique Y values."
    )

dx = np.min(np.diff(x_unique))
dy = np.min(np.diff(y_unique))

print(f"Cell size X = {dx:.3f} m")
print(f"Cell size Y = {dy:.3f} m")

# ===================================================
# LOOKUP TABLES
# ===================================================

x_lookup = {
    value: index
    for index, value in enumerate(x_unique)
}

y_lookup = {
    value: index
    for index, value in enumerate(y_unique)
}

# ===================================================
# RGB ARRAYS
# ===================================================

red = np.zeros(
    (ny, nx),
    dtype=np.uint8
)

green = np.zeros(
    (ny, nx),
    dtype=np.uint8
)

blue = np.zeros(
    (ny, nx),
    dtype=np.uint8
)

# ===================================================
# ASSIGN COLOURS
# ===================================================

for xx, yy, cls in zip(
    x,
    y,
    classification
):

    col = x_lookup[xx]

    # flip rows so north is up
    row = ny - 1 - y_lookup[yy]

    cls = str(cls).strip().lower()

    if cls == "urban":

        red[row, col] = 255

    elif cls == "water":

        blue[row, col] = 255

    else:

        green[row, col] = 255

# ===================================================
# GEOTRANSFORM
# ===================================================

xmin = x_unique.min() - dx / 2
ymax = y_unique.max() + dy / 2

transform = from_origin(
    xmin,
    ymax,
    dx,
    dy
)

# ===================================================
# WRITE GEOTIFF
# ===================================================

print("Writing GeoTIFF...")

with rasterio.open(
    OUTPUT_FILE,
    "w",
    driver="GTiff",
    height=ny,
    width=nx,
    count=3,
    dtype=np.uint8,
    crs=CRS,
    transform=transform
) as dst:

    dst.write(red, 1)
    dst.write(green, 2)
    dst.write(blue, 3)

print()
print(f"GeoTIFF saved as:")
print(OUTPUT_FILE)

print()
print("Classification summary:")
print(df["Classification"].value_counts())