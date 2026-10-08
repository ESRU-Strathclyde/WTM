from pathlib import Path

import pandas as pd
import rasterio
from pyproj import Transformer


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE = Path("input.txt")

WORLDCOVER_FILE = Path(
    "ESA_WorldCover_10m_2021_v200_S24W045_Map.tif"
)

OUTPUT_FILE = Path("classified_points.txt")

# Rio de Janeiro is in WGS84 / UTM Zone 23 South.
INPUT_CRS = "EPSG:32723"


# ============================================================
# WORLDCOVER CLASSIFICATION
# ============================================================

def classify_worldcover(value, nodata=None):
    """
    Convert an ESA WorldCover class into one of the required
    classifications: Urban, Water, Other, or Unknown.

    ESA WorldCover:
        50 = Built-up
        80 = Permanent water bodies
    """

    if value is None:
        return "Unknown"

    if nodata is not None and value == nodata:
        return "Unknown"

    if value == 50:
        return "Urban"

    if value == 80:
        return "Water"

    return "Other"


# ============================================================
# READ THE WINDSTATION FILE
# ============================================================

def read_input_file(filename):
    """
    Read the WindStation-style input file.

    The sample structure contains:
        Line 1: column headings
        Line 2: grid dimensions
        Line 3 onwards: numerical data

    Only the first two numerical columns are retained.
    """

    data = pd.read_csv(
        filename,
        sep=r"\s+",
        skiprows=3,
        header=None,
        usecols=[0, 1],
        names=["X[m]", "Y[m]"],
        encoding="cp1252"
    )

    # Ensure coordinates are numeric.
    data["X[m]"] = pd.to_numeric(data["X[m]"], errors="coerce")
    data["Y[m]"] = pd.to_numeric(data["Y[m]"], errors="coerce")

    # Identify invalid rows rather than silently processing them.
    invalid_rows = data[
        data["X[m]"].isna() | data["Y[m]"].isna()
    ]

    if not invalid_rows.empty:
        print(
            f"Warning: {len(invalid_rows)} row(s) contain invalid "
            "coordinates and will be removed."
        )

        data = data.dropna(subset=["X[m]", "Y[m]"])

    return data


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE.resolve()}"
        )

    if not WORLDCOVER_FILE.exists():
        raise FileNotFoundError(
            "WorldCover file not found:\n"
            f"{WORLDCOVER_FILE.resolve()}"
        )

    print(f"Reading input file: {INPUT_FILE}")
    data = read_input_file(INPUT_FILE)

    if data.empty:
        raise ValueError("No valid coordinate rows were found.")

    print(f"Number of valid points: {len(data):,}")
    print(f"Opening WorldCover tile: {WORLDCOVER_FILE}")

    with rasterio.open(WORLDCOVER_FILE) as raster:

        print(f"Raster CRS: {raster.crs}")
        print(f"Raster bounds: {raster.bounds}")
        print(f"Raster NoData value: {raster.nodata}")

        if raster.crs is None:
            raise ValueError(
                "The WorldCover raster does not have a defined CRS."
            )

        # Transform directly from UTM Zone 23S into the raster CRS.
        # For ESA WorldCover, the raster CRS will normally be EPSG:4326.
        coordinate_transformer = Transformer.from_crs(
            INPUT_CRS,
            raster.crs,
            always_xy=True
        )

        x_values = data["X[m]"].to_numpy()
        y_values = data["Y[m]"].to_numpy()

        raster_x, raster_y = coordinate_transformer.transform(
            x_values,
            y_values
        )

        # Check whether each point lies inside the downloaded tile.
        inside_raster = (
            (raster_x >= raster.bounds.left) &
            (raster_x <= raster.bounds.right) &
            (raster_y >= raster.bounds.bottom) &
            (raster_y <= raster.bounds.top)
        )

        outside_count = int((~inside_raster).sum())

        if outside_count > 0:
            print(
                f"Warning: {outside_count:,} point(s) fall outside "
                "the S24W045 tile."
            )
            print(
                "Those points will be classified as 'Outside raster'."
            )
        else:
            print("All points fall inside the WorldCover tile.")

        classifications = ["Outside raster"] * len(data)

        # Prepare only valid coordinates for raster sampling.
        valid_indices = [
            index
            for index, is_inside in enumerate(inside_raster)
            if is_inside
        ]

        valid_coordinates = [
            (raster_x[index], raster_y[index])
            for index in valid_indices
        ]

        print(
            f"Sampling land cover for "
            f"{len(valid_coordinates):,} point(s)..."
        )

        # raster.sample accepts an iterable of coordinates and is
        # considerably more efficient than reopening the raster for
        # every point.
        sampled_values = raster.sample(valid_coordinates)

        for index, sampled_pixel in zip(
            valid_indices,
            sampled_values
        ):
            land_cover_value = sampled_pixel[0]

            classifications[index] = classify_worldcover(
                land_cover_value,
                raster.nodata
            )

    # Create the requested three-column output.
    output = data.copy()
    output["Classification"] = classifications

    output.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        float_format="%.6e"
    )

    print()
    print(f"Output written to: {OUTPUT_FILE.resolve()}")
    print()
    print("Classification counts:")
    print(output["Classification"].value_counts(dropna=False))


if __name__ == "__main__":
    main()