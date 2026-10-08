import sys
import os
import pandas as pd
import math
import re
import pickle
import time
import copy
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Check for a non-blocking input solution
try:
    import msvcrt
except ImportError:
    # On non-Windows systems, use a different approach
    import select
    
    def non_blocking_input():
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.readline().strip().lower()
        return None
else:
    def non_blocking_input():
        if msvcrt.kbhit():
            return msvcrt.getch().decode().strip().lower()
        return None

print("WTM - Directional Wind Transposition Modeller (U/V)")
# This application uses CFD results obtained with WindStation.
# It creates correlations between data at the weather station and data at other points in the domain.
# WindStation results should be named as:
#        <weatherstationname>_x<UTM-coordinate>_y<UTM-coordinate>_<DIRECTION>_<speedat10m>.txt
# Example: INMET-838400_x677974.2_y7184195.2_30_5.txt

# Invoke the program with the station identifier, reference weather-station coordinates,
# and the number of simulated wind directions and speeds.
# The station coordinates are command-line inputs and are NOT read from input filenames.
# Example, for 12 wind directions [30°] and 3 wind speeds:
# python WTM.py INMET-838400 677974.2 7184195.2 12 3

# --- Command line argument handling ---
if len(sys.argv) != 6:
    print("Usage: python WTM.py <station_name> <station_x> <station_y> <number_wind_directions> <number_wind_speeds>")
    sys.exit(1)

# Store input parameters
station_name = sys.argv[1].strip().rstrip('_').rstrip()
station_x = float(sys.argv[2]) # Convert to float for numerical operations
station_y = float(sys.argv[3]) # Convert to float for numerical operations
number_wind_direction = (sys.argv[4])
number_wind_speed = (sys.argv[5])

# Print them
print("station_name:", station_name)
print("station_x:", station_x)
print("station_y:", station_y)
print("number_wind_direction:", number_wind_direction)
print("number_wind_speed:", number_wind_speed)

num_wind_dir = int(number_wind_direction)
num_wind_speed = int(number_wind_speed)

# Calculate number of files
total_files = num_wind_dir * num_wind_speed
print("\nTotal number of files:", total_files)

# Calculate angular discretization
angular_discretization = 360 / num_wind_dir
print("Angular discretization (degrees):", angular_discretization)

CLASSIFIED_POINTS_FILE = "classified_points.txt"

# Find all matching CFD input files in the current folder.
# This deliberately avoids reconstructing filenames. It extracts the final
# direction and speed numbers from each candidate filename, tolerating spaces,
# decimal notation, and suffixes such as "-sample".
expected_pairs = set()
for d in range(num_wind_dir):
    direction = int(round((d + 1) * angular_discretization)) % 360
    for speed in range(1, num_wind_speed + 1):
        expected_pairs.add((direction, speed))


def parse_filename_direction_speed(filename):
    """Extract the final _direction_speed pair from a CFD filename."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(
        r'_\s*([+-]?\d+(?:\.\d+)?)\s*_\s*([+-]?\d+(?:\.\d+)?)',
        stem
    )
    matches = list(re.finditer(
        r'_\s*([+-]?\d+(?:\.\d+)?)\s*_\s*([+-]?\d+(?:\.\d+)?)',
        stem
    ))
    if not matches:
        raise ValueError(f"Cannot identify direction and speed in filename: {filename}")
    match = matches[-1]
    direction = int(round(float(match.group(1)))) % 360
    speed = int(round(float(match.group(2))))
    return direction, speed


def matches_expected_simulation(filename):
    if not filename.lower().endswith('.txt'):
        return False
    if filename.lower() == CLASSIFIED_POINTS_FILE.lower():
        return False

    # Retain the station-name check, but ignore accidental leading/trailing spaces
    # in the command-line station name and filename.
    clean_station_name = station_name.strip().casefold()
    if clean_station_name and not filename.strip().casefold().startswith(clean_station_name):
        return False

    try:
        pair = parse_filename_direction_speed(filename)
    except ValueError:
        return False
    return pair in expected_pairs


all_text_files = sorted(
    filename for filename in os.listdir('.')
    if os.path.isfile(filename) and filename.lower().endswith('.txt')
)
existing_files = [filename for filename in all_text_files if matches_expected_simulation(filename)]

print("\nCFD files recognised:", len(existing_files), "of", total_files, "expected")
print("Files found in current folder:")
if existing_files:
    for filename in existing_files:
        direction, speed = parse_filename_direction_speed(filename)
        print(f"  {filename}  [direction={direction:g}, speed={speed:g}]")
else:
    print("  None found")
    print("\nText files inspected but not recognised as CFD inputs:")
    for filename in all_text_files:
        if filename.lower() != CLASSIFIED_POINTS_FILE.lower():
            print("  ", filename)

# -------------------------------    
# function saves ann parameters
def export_ann_parameters(model, filename="ann_parameters.txt"):
    """Export ANN parameters to an ASCII file."""
    # If model is a pipeline, extract the MLPRegressor
    mlp = None
    if hasattr(model, "steps"):
        for step_name, step in model.steps:
            if isinstance(step, MLPRegressor):
                mlp = step
                break
    elif isinstance(model, MLPRegressor):
        mlp = model
    
    if mlp is None:
        print("No MLPRegressor found in model. Cannot export parameters.")
        return
    
    with open(filename, "w") as f:
        f.write("=== Artificial Neural Network Parameters ===\n\n")
        
        # Layers and nodes
        f.write(f"Number of layers (including input & output): {mlp.n_layers_}\n")
        f.write(f"Layer sizes: {mlp.hidden_layer_sizes} (hidden), output={mlp.n_outputs_}\n\n")
        
        # Activation function
        f.write(f"Activation function: {mlp.activation}\n\n")
        
        # Loop through layers
        for i, (weights, biases) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            f.write(f"--- Layer {i+1} ---\n")
            f.write(f"Number of nodes: {weights.shape[1]}\n")
            f.write("Weights:\n")
            for row in weights:
                f.write("  " + " ".join(f"{w:.6f}" for w in row) + "\n")
            f.write("Biases:\n")
            f.write("  " + " ".join(f"{b:.6f}" for b in biases) + "\n")
            
            # Upper and lower limits
            f.write(f"Weight range: {weights.min():.6f} to {weights.max():.6f}\n")
            f.write(f"Bias range: {biases.min():.6f} to {biases.max():.6f}\n\n")
    
    print(f"ANN parameters exported to '{filename}'")


def replace_text_in_file(input_filename, output_filename, tag, replacement_string):
    """
    Reads a file, replaces all occurrences of a specific tag with a string,
    and saves the content to a new file.

    Args:
        input_filename (str): The name of the file to be read.
        output_filename (str): The name of the new file to be created.
        tag (str): The string to be replaced (e.g., '<STATIONNAME>').
        replacement_string (str): The string to replace the tag with.
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_filename):
            print(f"Error: The input file '{input_filename}' was not found.")
            return

        # Read the content of the original file
        with open(input_filename, 'r') as infile:
            file_content = infile.read()

        # Replace the tag with the new string
        modified_content = file_content.replace(tag, replacement_string)

        # Write the modified content to the new file
        with open(output_filename, 'w') as outfile:
            outfile.write(modified_content)
        
        print(f"Script created: '{output_filename}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

# # --- Example Usage ---
# if __name__ == "__main__":
    # # Define the original and new file names
    # original_file = "template.txt"
    # new_file = "output.txt"
    
    # # Define the tag and the replacement string
    # target_tag = "<STATIONNAME>"
    # replacement_text = "TEST"

    # # --- Create a dummy file for demonstration ---
    # # In a real scenario, this file would already exist.
    # with open(original_file, 'w') as f:
        # f.write("This is a file for <STATIONNAME> data.\n")
        # f.write("Data for the <STATIONNAME> station is available.")

    # # Call the function to perform the replacement
    # replace_text_in_file(original_file, new_file, target_tag, replacement_text)







# --- Function to read and process files into a DataFrame ---
# --- Urban-point classification ---
COORDINATE_DECIMALS = 3


def coordinate_key(x, y):
    """Create a stable coordinate key for matching CFD and classification points."""
    return (round(float(x), COORDINATE_DECIMALS), round(float(y), COORDINATE_DECIMALS))


def load_urban_points(filename=CLASSIFIED_POINTS_FILE):
    """Load coordinates whose Classification field is exactly Urban (case-insensitive)."""
    if not os.path.isfile(filename):
        print(f"Error: Required classification file '{filename}' was not found.")
        sys.exit(1)

    try:
        classified = pd.read_csv(filename, sep=r'\s+', engine='python', encoding='latin-1')
    except Exception as error:
        print(f"Error reading classification file '{filename}': {error}")
        sys.exit(1)

    normalised_columns = {
        str(column).strip().lower().replace('\\', ''): column
        for column in classified.columns
    }

    x_column = next((original for normalised, original in normalised_columns.items()
                     if normalised in ('x[m]', 'x')), None)
    y_column = next((original for normalised, original in normalised_columns.items()
                     if normalised in ('y[m]', 'y')), None)
    classification_column = next((original for normalised, original in normalised_columns.items()
                                  if normalised == 'classification'), None)

    if x_column is None or y_column is None or classification_column is None:
        print(
            f"Error: '{filename}' must contain X[m], Y[m], and Classification columns. "
            f"Columns found: {list(classified.columns)}"
        )
        sys.exit(1)

    urban_rows = classified[
        classified[classification_column].astype(str).str.strip().str.casefold() == 'urban'
    ].copy()

    urban_points = {
        coordinate_key(x, y)
        for x, y in zip(urban_rows[x_column], urban_rows[y_column])
    }

    if not urban_points:
        print(f"Error: No points classified as Urban were found in '{filename}'.")
        sys.exit(1)

    print(f"Loaded {len(classified)} classified points from '{filename}'.")
    print(f"Urban points retained for ANN training and validation: {len(urban_points)}")
    return urban_points


# --- Direction and vector utilities ---
def parse_simulation_direction(filename):
    """Return the simulated inflow direction encoded in a CFD filename."""
    direction, _ = parse_filename_direction_speed(filename)
    return float(direction)


def speed_direction_to_uv(speed, direction_deg):
    """Convert meteorological direction-from and speed to eastward U/northward V."""
    angle = np.radians(direction_deg)
    u = -speed * np.sin(angle)
    v = -speed * np.cos(angle)
    return u, v


def circular_mean_degrees(directions_deg):
    """Return circular mean direction in degrees."""
    angles = np.radians(np.asarray(directions_deg, dtype=float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))) % 360.0)


# --- Function to read and process files into a DataFrame ---
def read_data_for_ann(filenames, station_x, station_y, urban_points):
    all_data = []
    print("\nReading files and collecting data for directional ANNs...")

    for fname in filenames:
        try:
            simulation_dir = parse_simulation_direction(fname)
            with open(fname, 'r', encoding='latin-1') as file:
                headers = file.readline().strip().split()
                try:
                    x_idx = headers.index('X[m]')
                    y_idx = headers.index('Y[m]')
                    zground_idx = headers.index('ZGround[m]')
                    wspd_idx = headers.index('Wspd_2D[m/s]')
                    dir_idx = headers.index('Direction[º]')
                except ValueError as e:
                    print(f"Required header not found in {fname}: {e}")
                    continue

                file.readline()
                file_points = []
                for line in file:
                    values = line.strip().split()
                    if len(values) > max(x_idx, y_idx, zground_idx, wspd_idx, dir_idx):
                        try:
                            point = {
                                'x': float(values[x_idx]),
                                'y': float(values[y_idx]),
                                'zground': float(values[zground_idx]),
                                'wspd': float(values[wspd_idx]),
                                'dir': float(values[dir_idx]) % 360.0
                            }
                            file_points.append(point)
                        except (IndexError, ValueError) as error:
                            print(f"Error parsing data in {fname}: {error}")

            if not file_points:
                print(f"No valid data points found in {fname}. Skipping file.")
                continue

            closest_point = min(
                file_points,
                key=lambda point: math.hypot(point['x'] - station_x, point['y'] - station_y)
            )
            station_wspd = closest_point['wspd']
            station_dir = closest_point['dir']

            for point in file_points:
                if coordinate_key(point['x'], point['y']) not in urban_points:
                    continue
                if math.hypot(point['x'] - closest_point['x'], point['y'] - closest_point['y']) > 0.001:
                    target_u, target_v = speed_direction_to_uv(point['wspd'], point['dir'])
                    all_data.append([
                        simulation_dir, fname, station_wspd, station_dir,
                        point['x'], point['y'], point['zground'],
                        target_u, target_v, point['wspd'], point['dir']
                    ])
        except (IOError, ValueError) as error:
            print(f"Error processing {fname}: {error}")

    return pd.DataFrame(all_data, columns=[
        'simulation_dir', 'source_file', 'station_wspd', 'station_dir', 'target_x', 'target_y', 'target_zground',
        'target_u', 'target_v', 'target_wspd', 'target_dir'
    ])


# --- Main script logic ---
if existing_files:
    model_filename = station_name + '_ann_model.pkl'
    urban_points = load_urban_points()
    df = read_data_for_ann(existing_files, station_x, station_y, urban_points)

    if df.empty:
        print("\nNo Urban CFD points matched classified_points.txt. Cannot train the models.")
        sys.exit(1)

    print("\nSuccessfully collected data for training.")
    print(f"DataFrame head:\n{df.head()}")

    feature_columns = ['station_wspd', 'target_x', 'target_y', 'target_zground']
    output_columns = ['target_u', 'target_v']
    trained_models = {}
    training_results_all = []
    validation_results_all = []
    ann_metrics = []

    unique_directions = sorted(df['simulation_dir'].unique())
    print(f"\nTraining {len(unique_directions)} directional ANNs: {unique_directions}")
    print("Maximum epochs per ANN: 500")
    print("Automatic stopping: 10 epochs without an improvement greater than 0.0001")
    print("Press 's' and then 'Enter' to stop the current ANN early.")

    for simulation_dir in unique_directions:
        direction_df = df[df['simulation_dir'] == simulation_dir].copy()
        if len(direction_df) < 5:
            print(f"Skipping {simulation_dir:g} degrees: only {len(direction_df)} samples.")
            continue

        # Each CFD file contributes one actual direction measured at the
        # reference weather-station point. Average these directions as vectors,
        # then round the circular mean to the nearest integer for the ANN index.
        station_direction_by_file = (
            direction_df[['source_file', 'station_dir']]
            .drop_duplicates(subset=['source_file'])
        )
        actual_station_dir = circular_mean_degrees(
            station_direction_by_file['station_dir']
        )
        actual_station_dir_index = int(round(actual_station_dir)) % 360
        station_dir_min = float(station_direction_by_file['station_dir'].min())
        station_dir_max = float(station_direction_by_file['station_dir'].max())

        X = direction_df[feature_columns]
        y = direction_df[output_columns]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n--- ANN for nominal {simulation_dir:g} degrees; mean station direction {actual_station_dir:.3f} degrees; ANN index {actual_station_dir_index} degrees ---")
        print(f"Training samples: {len(X_train)}; validation samples: {len(X_test)}")

        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(200, 100), activation='relu', solver='adam',
                max_iter=1, warm_start=True, random_state=42
            )
        )
        best_score = -float('inf')
        best_model = None
        best_epoch = 0
        max_epochs = 500
        patience = 10
        min_improvement = 0.0001
        epochs_without_improvement = 0
        epochs_completed = 0

        for epoch in range(max_epochs):
            model.fit(X_train, y_train)
            current_score = model.score(X_test, y_test)
            epochs_completed = epoch + 1

            if current_score > best_score + min_improvement:
                best_score = current_score
                best_model = copy.deepcopy(model)
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1}: current score {current_score:.4f}; "
                    f"best score {best_score:.4f}; "
                    f"epochs without improvement {epochs_without_improvement}"
                )

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: no improvement greater "
                    f"than {min_improvement:.4f} for {patience} consecutive epochs."
                )
                break

            if non_blocking_input() == 's':
                print("Training interrupted for this direction. Keeping its best epoch.")
                break

        if best_model is None:
            print(f"No valid model produced for {simulation_dir:g} degrees.")
            continue

        # Store the ANN under the rounded vector-average direction measured
        # at the reference station for this nominal-direction file group.
        key = actual_station_dir_index
        if key in trained_models:
            raise ValueError(
                f"Duplicate rounded station-direction index {key} degrees. "
                f"Nominal direction {simulation_dir:g} conflicts with another ANN. "
                "Inspect the reference-station directions or use finer indexing."
            )
        trained_models[key] = best_model

        train_uv = best_model.predict(X_train)
        test_uv = best_model.predict(X_test)
        train_speed = np.maximum(
            0.0,
            np.hypot(train_uv[:, 0], train_uv[:, 1])
        )

        test_speed = np.maximum(
            0.0,
            np.hypot(test_uv[:, 0], test_uv[:, 1])
)

        test_true_speed = y_test.join(direction_df[['target_wspd']], how='left')['target_wspd'].to_numpy()
        speed_errors = test_speed - test_true_speed
        absolute_errors = np.abs(speed_errors)
        bias_wspd = float(np.mean(speed_errors))
        rmse_wspd = float(np.sqrt(np.mean(speed_errors ** 2)))
        mae_wspd = float(np.mean(absolute_errors))
        median_ae_wspd = float(np.median(absolute_errors))
        p90_ae_wspd = float(np.percentile(absolute_errors, 90))
        p95_ae_wspd = float(np.percentile(absolute_errors, 95))
        max_ae_wspd = float(np.max(absolute_errors))
        pct_above_2 = float(np.mean(absolute_errors > 2.0) * 100.0)
        pct_above_3 = float(np.mean(absolute_errors > 3.0) * 100.0)
        print(f"Best combined U/V R2: {best_score:.4f}")
        print(f"Best epoch: {best_epoch}; epochs completed: {epochs_completed}")
        print(f"Wind-speed bias: {bias_wspd:.4f} m/s")
        print(f"Wind-speed RMSE: {rmse_wspd:.4f} m/s")
        print(f"MAE: {mae_wspd:.4f} m/s; P95 absolute error: {p95_ae_wspd:.4f} m/s")
        print(f"Absolute errors >2 m/s: {pct_above_2:.2f}%; >3 m/s: {pct_above_3:.2f}%")

        ann_metrics.append({
            'direction': float(simulation_dir),
            'actual_station_direction': float(actual_station_dir),
            'ann_direction_index': int(actual_station_dir_index),
            'station_direction_file_count': int(len(station_direction_by_file)),
            'station_direction_min': station_dir_min,
            'station_direction_max': station_dir_max,
            'samples': int(len(direction_df)),
            'training_samples': int(len(X_train)),
            'validation_samples': int(len(X_test)),
            'best_epoch': int(best_epoch),
            'epochs_completed': int(epochs_completed),
            'best_uv_r2': float(best_score),
            'wind_speed_bias': float(bias_wspd),
            'wind_speed_rmse': rmse_wspd,
            'wind_speed_mae': mae_wspd,
            'median_absolute_error': median_ae_wspd,
            'p90_absolute_error': p90_ae_wspd,
            'p95_absolute_error': p95_ae_wspd,
            'maximum_absolute_error': max_ae_wspd,
            'percent_absolute_error_above_2': pct_above_2,
            'percent_absolute_error_above_3': pct_above_3
        })

        train_results = direction_df.loc[X_train.index].copy()
        train_results['predicted_u'] = train_uv[:, 0]
        train_results['predicted_v'] = train_uv[:, 1]
        train_results['predicted_wspd'] = train_speed
        training_results_all.append(train_results)

        test_results = direction_df.loc[X_test.index].copy()
        test_results['predicted_u'] = test_uv[:, 0]
        test_results['predicted_v'] = test_uv[:, 1]
        test_results['predicted_wspd'] = test_speed
        validation_results_all.append(test_results)

        export_ann_parameters(
            best_model,
            station_name + f"_ann_parameters_{simulation_dir:g}deg.txt"
        )

    if not trained_models:
        print("\nNo directional ANN could be trained.")
        sys.exit(1)

    bundle = {
        'architecture': 'one_ann_per_direction_uv',
        'feature_columns': feature_columns,
        'models': trained_models,
        'directions': sorted(trained_models.keys()),
        'direction_keys': 'rounded circular-mean station directions for each nominal CFD file group',
        'maximum_epochs': 500,
        'early_stopping_patience': 10,
        'minimum_improvement': 0.0001,
        'training_point_filter': 'Classification == Urban',
        'classification_file': CLASSIFIED_POINTS_FILE
    }
    with open(model_filename, 'wb') as file:
        pickle.dump(bundle, file)
    print(f"\nAll directional ANNs saved together in '{model_filename}'.")

    # Final summary of validation performance for every successfully trained ANN.
    metrics_filename = station_name + "_directional_ann_metrics.txt"
    summary_lines = [
        "=== Directional ANN validation summary ===",
        "",
        "Architecture: one ANN per nominal CFD file group, indexed by rounded vector-average station direction, predicting U and V",
        "Maximum epochs per ANN: 500",
        "Early-stopping patience: 10 epochs",
        "Minimum score improvement: 0.0001",
        "Training/validation filter: Classification == Urban",
        f"Classification file: {CLASSIFIED_POINTS_FILE}",
        "",
        "Direction  Samples  Train  Validation  Best epoch  Epochs run  Best U/V R2  Bias [m/s]  RMSE [m/s]",
        "---------  -------  -----  ----------  ----------  ----------  -----------  ----------  ----------"
    ]
    for metric in sorted(ann_metrics, key=lambda item: item['direction']):
        summary_lines.append(
            f"{metric['direction']:9.1f}  {metric['samples']:7d}  "
            f"{metric['training_samples']:5d}  {metric['validation_samples']:10d}  "
            f"{metric['best_epoch']:10d}  {metric['epochs_completed']:10d}  "
            f"{metric['best_uv_r2']:11.4f}  {metric['wind_speed_bias']:10.4f}  "
            f"{metric['wind_speed_rmse']:10.4f}"
        )
        summary_lines.append(
            f"           Actual station direction={metric['actual_station_direction']:.3f} deg; "
            f"ANN index={metric['ann_direction_index']} deg; files={metric['station_direction_file_count']}; "
            f"MAE={metric['wind_speed_mae']:.4f}; MedianAE={metric['median_absolute_error']:.4f}; "
            f"P90AE={metric['p90_absolute_error']:.4f}; P95AE={metric['p95_absolute_error']:.4f}; "
            f"MaxAE={metric['maximum_absolute_error']:.4f}; "
            f">2m/s={metric['percent_absolute_error_above_2']:.2f}%; "
            f">3m/s={metric['percent_absolute_error_above_3']:.2f}%"
        )

    if validation_results_all:
        all_validation = pd.concat(validation_results_all, ignore_index=True)
        overall_errors = (all_validation['predicted_wspd'] - all_validation['target_wspd']).to_numpy()
        overall_abs = np.abs(overall_errors)
        overall_bias = float(np.mean(overall_errors))
        overall_rmse = float(np.sqrt(np.mean(overall_errors ** 2)))
        summary_lines.extend([
            "",
            "=== Overall validation performance across all directional ANNs ===",
            f"Validation samples: {len(all_validation)}",
            f"Overall wind-speed bias: {overall_bias:.4f} m/s",
            f"Overall wind-speed RMSE: {overall_rmse:.4f} m/s",
            f"Overall wind-speed MAE: {np.mean(overall_abs):.4f} m/s",
            f"Median absolute error: {np.median(overall_abs):.4f} m/s",
            f"90th percentile absolute error: {np.percentile(overall_abs, 90):.4f} m/s",
            f"95th percentile absolute error: {np.percentile(overall_abs, 95):.4f} m/s",
            f"Maximum absolute error: {np.max(overall_abs):.4f} m/s",
            f"Absolute errors above 2 m/s: {np.mean(overall_abs > 2.0) * 100.0:.2f}%",
            f"Absolute errors above 3 m/s: {np.mean(overall_abs > 3.0) * 100.0:.2f}%"
        ])

    final_summary = "\n".join(summary_lines)
    print("\n" + final_summary)
    with open(metrics_filename, 'w', encoding='utf-8') as metrics_file:
        metrics_file.write(final_summary + "\n")
    print(f"\nDirectional ANN metrics saved to '{metrics_filename}'.")

    if training_results_all:
        pd.concat(training_results_all, ignore_index=True).to_csv(
            station_name + "_training_predictions.csv", index=False
        )
    if validation_results_all:
        pd.concat(validation_results_all, ignore_index=True).to_csv(
            station_name + "_validation_predictions.csv", index=False
        )

    # Generate the same user-facing helper scripts as before. The multiple ANNs,
    # nearest-direction selection, and U/V interpolation remain behind the scenes.
    replace_text_in_file(
        "WTM_station_template.py",
        "WTM_" + station_name + "_ann.py",
        "<STATIONNAME>", station_name
    )
    replace_text_in_file(
        "WTM_epw_template.py",
        "WTM_epw_" + station_name + ".py",
        "<STATIONNAME>", station_name
    )
else:
    print("\nNo matching input files were found. Cannot train the models.")
