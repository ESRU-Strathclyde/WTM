import sys
import pickle
import numpy as np
import pandas as pd

# This predictor uses one ANN for each nominal CFD wind direction.
# The requested station direction is used only to select one ANN or interpolate
# between the two closest nominal-direction ANNs.

if len(sys.argv) != 6:
    print("Usage: python predict_wind.py <station_wspd> <station_dir> <target_x> <target_y> <target_zground>")
    sys.exit(1)

try:
    station_wspd = float(sys.argv[1])
    station_dir = float(sys.argv[2]) % 360.0
    target_x = float(sys.argv[3])
    target_y = float(sys.argv[4])
    target_zground = float(sys.argv[5])
except ValueError:
    print("Error: All five input parameters must be numbers.")
    sys.exit(1)

model_filename = "Rio_Copacabana" + "_ann_model.pkl"
try:
    with open(model_filename, "rb") as file:
        bundle = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Please train the ANN first.")
    sys.exit(1)

models = bundle.get("models", {})
if not models:
    print("Error: The model bundle contains no directional ANNs.")
    sys.exit(1)

# Model keys are rounded integer directions derived from the vector-average
# wind direction measured at the reference station for each nominal CFD group.
models_by_direction = {}
for original_key, model in models.items():
    direction_key = int(round(float(original_key))) % 360
    if direction_key in models_by_direction:
        print(f"Error: Duplicate ANN model for direction index {direction_key} degrees.")
        sys.exit(1)
    models_by_direction[direction_key] = model

directions = sorted(models_by_direction)
if len(directions) == 1:
    lower_key = directions[0]
    upper_key = directions[0]
    interpolation_weight = 0.0
    circular_interval = False
else:
    exact_direction = next(
        (direction for direction in directions if abs(station_dir - direction) < 1.0e-9),
        None
    )

    if exact_direction is not None:
        lower_key = exact_direction
        upper_key = exact_direction
        interpolation_weight = 0.0
        circular_interval = False
    else:
        # Extend the sorted indexes by repeating the first index at +360 degrees.
        # If the query is below the minimum index, add 360 so it falls between
        # the maximum index and the wrapped minimum. Queries above the maximum
        # naturally fall in that same circular interval.
        extended_directions = directions + [directions[0] + 360]
        query_extended = station_dir
        if query_extended < directions[0]:
            query_extended += 360.0

        lower_extended = None
        upper_extended = None
        for candidate_lower, candidate_upper in zip(
            extended_directions[:-1], extended_directions[1:]
        ):
            if candidate_lower < query_extended < candidate_upper:
                lower_extended = float(candidate_lower)
                upper_extended = float(candidate_upper)
                break

        if lower_extended is None:
            # This covers numerical edge cases at the circular boundary.
            lower_extended = float(directions[-1])
            upper_extended = float(directions[0] + 360)
            if query_extended <= directions[-1]:
                query_extended += 360.0

        lower_key = int(lower_extended) % 360
        upper_key = int(upper_extended) % 360
        interpolation_weight = (
            (query_extended - lower_extended)
            / (upper_extended - lower_extended)
        )
        circular_interval = upper_extended >= 360.0

input_data = pd.DataFrame(
    [[station_wspd, target_x, target_y, target_zground]],
    columns=["station_wspd", "target_x", "target_y", "target_zground"]
)

uv_lower = models_by_direction[lower_key].predict(input_data)[0]
speed_lower = max(0.0, float(np.hypot(uv_lower[0], uv_lower[1])))

if lower_key == upper_key:
    uv_upper = uv_lower.copy()
    speed_upper = speed_lower
    uv_interpolated = uv_lower.copy()
else:
    uv_upper = models_by_direction[upper_key].predict(input_data)[0]
    speed_upper = max(0.0, float(np.hypot(uv_upper[0], uv_upper[1])))
    uv_interpolated = (
        (1.0 - interpolation_weight) * uv_lower
        + interpolation_weight * uv_upper
    )

predicted_speed = max(
    0.0,
    float(np.hypot(uv_interpolated[0], uv_interpolated[1]))
)

# Explicit prediction details. These are intentionally part of the standalone
# ANN output while the EPW integration is being reviewed separately.
print(f"Requested station direction: {station_dir:.6f} degrees")
print(f"Lower ANN direction: {lower_key} degrees")
print(f"Upper ANN direction: {upper_key} degrees")
print(f"Interpolation weight toward upper ANN: {interpolation_weight:.8f}")
print(f"Circular boundary interpolation: {circular_interval}")
print(
    f"Lower ANN result: U={uv_lower[0]:.6f} m/s, "
    f"V={uv_lower[1]:.6f} m/s, speed={speed_lower:.6f} m/s"
)
print(
    f"Upper ANN result: U={uv_upper[0]:.6f} m/s, "
    f"V={uv_upper[1]:.6f} m/s, speed={speed_upper:.6f} m/s"
)
print(
    f"Interpolated result: U={uv_interpolated[0]:.6f} m/s, "
    f"V={uv_interpolated[1]:.6f} m/s, speed={predicted_speed:.6f} m/s"
)
print(f"Final predicted wind speed: {predicted_speed:.4f} m/s")
