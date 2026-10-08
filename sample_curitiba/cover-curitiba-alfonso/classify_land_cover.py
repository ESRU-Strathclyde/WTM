from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


# =====================================================
# USER SETTINGS
# =====================================================

INPUT_FILE = Path("input.txt")

WORLDCOVER_FILE = Path(
    "ESA_WorldCover_10m_2021_v200_S27W051_Map.tif"
)

OUTPUT_FILE = Path(
    "classified_points.txt"
)

#adapt to your location
INPUT_CRS = "EPSG:32722"

URBAN_THRESHOLD = 20.0
WATER_THRESHOLD = 50.0


# =====================================================
# WORLD COVER CLASSES
# =====================================================

BUILT_UP = 50
WATER = 80


# =====================================================
# READ WINDSTATION FILE
# =====================================================

def read_grid_size(filename):

    with open(filename, "r", encoding="cp1252") as f:

        f.readline()

        line = f.readline()

    values = line.split()

    nx = int(values[0])
    ny = int(values[1])

    return nx, ny


def read_points(filename):

    df = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=3,
        header=None,
        encoding="cp1252",
        usecols=[0, 1]
    )

    df.columns = ["X", "Y"]

    return df


# =====================================================
# CLASSIFICATION RULE
# =====================================================

def classify(urban_pct, water_pct):

    if urban_pct >= URBAN_THRESHOLD:
        return "Urban"

    if water_pct >= WATER_THRESHOLD:
        return "Water"

    return "Other"


# =====================================================
# MAIN
# =====================================================

def main():

    print("Reading input file...")

    nx, ny = read_grid_size(INPUT_FILE)

    df = read_points(INPUT_FILE)

    print(f"Grid: {nx} x {ny}")

    xmin = df["X"].min()
    xmax = df["X"].max()

    ymin = df["Y"].min()
    ymax = df["Y"].max()

    print()
    print("Domain extent")
    print(f"Xmin = {xmin}")
    print(f"Xmax = {xmax}")
    print(f"Ymin = {ymin}")
    print(f"Ymax = {ymax}")

    # -------------------------------------------------
    # CELL SIZE
    # -------------------------------------------------

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    print()
    print(f"Cell size dx = {dx:.3f} m")
    print(f"Cell size dy = {dy:.3f} m")

    # -------------------------------------------------
    # OPEN WORLDCOVER
    # -------------------------------------------------

    with rasterio.open(WORLDCOVER_FILE) as raster:

        print()
        print("WorldCover information")
        print(raster.crs)
        print(raster.bounds)

        transformer = Transformer.from_crs(
            INPUT_CRS,
            raster.crs,
            always_xy=True
        )

        results = []

        total_points = len(df)

        for counter, row in enumerate(df.itertuples()):

            if counter % 500 == 0:

                print(
                    f"{counter}/{total_points}"
                )

            x = row.X
            y = row.Y

            # -----------------------------------------
            # CELL BOUNDARIES IN UTM
            # -----------------------------------------

            cell_xmin = x - dx / 2
            cell_xmax = x + dx / 2

            cell_ymin = y - dy / 2
            cell_ymax = y + dy / 2

            # -----------------------------------------
            # CONVERT TO LAT/LON
            # -----------------------------------------

            lon1, lat1 = transformer.transform(
                cell_xmin,
                cell_ymin
            )

            lon2, lat2 = transformer.transform(
                cell_xmax,
                cell_ymax
            )

            west = min(lon1, lon2)
            east = max(lon1, lon2)

            south = min(lat1, lat2)
            north = max(lat1, lat2)

            # -----------------------------------------
            # PIXEL WINDOW
            # -----------------------------------------

            row_min, col_min = raster.index(
                west,
                north
            )

            row_max, col_max = raster.index(
                east,
                south
            )

            row_start = min(row_min, row_max)
            row_end = max(row_min, row_max)

            col_start = min(col_min, col_max)
            col_end = max(col_min, col_max)

            try:

                data = raster.read(
                    1,
                    window=(
                        (
                            row_start,
                            row_end + 1
                        ),
                        (
                            col_start,
                            col_end + 1
                        )
                    )
                )

            except Exception:

                results.append(
                    [
                        x,
                        y,
                        0,
                        0,
                        100,
                        "Outside raster"
                    ]
                )

                continue

            total_pixels = data.size

            if total_pixels == 0:

                results.append(
                    [
                        x,
                        y,
                        0,
                        0,
                        100,
                        "Outside raster"
                    ]
                )

                continue

            urban_pixels = np.sum(
                data == BUILT_UP
            )

            water_pixels = np.sum(
                data == WATER
            )

            urban_pct = (
                100.0 *
                urban_pixels /
                total_pixels
            )

            water_pct = (
                100.0 *
                water_pixels /
                total_pixels
            )

            other_pct = (
                100.0 -
                urban_pct -
                water_pct
            )

            cls = classify(
                urban_pct,
                water_pct
            )

            results.append(
                [
                    x,
                    y,
                    round(urban_pct, 1),
                    round(water_pct, 1),
                    round(other_pct, 1),
                    cls
                ]
            )

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------

    output = pd.DataFrame(
        results,
        columns=[
            "X[m]",
            "Y[m]",
            "Urban[%]",
            "Water[%]",
            "Other[%]",
            "Classification"
        ]
    )

    output.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")

    print()
    print(
        output["Classification"]
        .value_counts()
    )


if __name__ == "__main__":
    main()