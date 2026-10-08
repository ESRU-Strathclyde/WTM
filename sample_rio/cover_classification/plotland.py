import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

INPUT_FILE = "classified_points.txt"
OUTPUT_TIF = "landcover_map.tif"

CRS = "EPSG:32723"   # UTM Zone 23S


# --------------------------------------------------
# READ DATA
# --------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    sep=r"\s+",
    engine="python"
)

x = df["X[m]"].values
y = df["Y[m]"].values
cls = df["Classification"].values

# --------------------------------------------------
# BUILD GRID
# --------------------------------------------------

x_unique = np.sort(np.unique(x))
y_unique = np.sort(np.unique(y))

nx = len(x_unique)
ny = len(y_unique)

print(f"Grid size: {nx} x {ny}")

# Estimate cell size
dx = np.min(np.diff(x_unique))
dy = np.min(np.diff(y_unique))

print(f"Cell size X = {dx}")
print(f"Cell size Y = {dy}")

# Lookup tables
x_lookup = {v: i for i, v in enumerate(x_unique)}
y_lookup = {v: i for i, v in enumerate(y_unique)}

# RGB bands
red = np.zeros((ny, nx), dtype=np.uint8)
green = np.zeros((ny, nx), dtype=np.uint8)
blue = np.zeros((ny, nx), dtype=np.uint8)

# --------------------------------------------------
# ASSIGN COLOURS
# --------------------------------------------------

for xx, yy, cc in zip(x, y, cls):

    col = x_lookup[xx]

    # flip y-axis so north is up
    row = ny - 1 - y_lookup[yy]

    if str(cc).lower() == "urban":

        red[row, col] = 255

    elif str(cc).lower() == "water":

        blue[row, col] = 255

    else:

        green[row, col] = 255

# --------------------------------------------------
# GEOTRANSFORM
# --------------------------------------------------

xmin = x_unique.min() - dx / 2
ymax = y_unique.max() + dy / 2

transform = from_origin(
    xmin,
    ymax,
    dx,
    dy
)

# --------------------------------------------------
# WRITE GEOTIFF
# --------------------------------------------------

with rasterio.open(
    OUTPUT_TIF,
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

print(f"GeoTIFF written to: {OUTPUT_TIF}")