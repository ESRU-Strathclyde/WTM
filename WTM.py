import sys
import os
import pandas as pd
import math
import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

print("WTM - Wind Transposition Modeller")
# This application uses CFD results obtained with WindStation.
# It creates correlations between data at the weather station and data at other points in the domain.
# WindStation results should be named as:
#        <weatherstationname>_x<UTM-coordinate>_y<UTM-coordinate>_<DIRECTION>_<speedat10m>.txt
# Example: INMET-838400_x677974.2_y7184195.2_30_5.txt

# invoke the program using as parameters data matching the file name and number of simulations for wind direction and speed.
# Example, for 12 wind directions [30°] and 3 wind speeds:
# python WTM.py INMET-838400 677974.2 7184195.2 12 3

# --- Command line argument handling ---
if len(sys.argv) != 6:
    print("Usage: python WTM.py <param1> <param2> <param3> <param4> <param5>")
    sys.exit(1)

# Store input parameters
station_name = (sys.argv[1])
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

# Generate filenames
expected_files = []
for d in range(num_wind_dir - 1):
    direction = int((d + 1) * angular_discretization)  # e.g. 30, 60, ...
    for s in range(1, num_wind_speed + 1):
        fname = f"{station_name}_x{station_x}_y{station_y}_{direction}_{s}.txt"
        expected_files.append(fname)

# Check existing files in current folder
existing_files = [f for f in expected_files if os.path.isfile(f)]

# # Print results
# print("\nExpected filenames:")
# for f in expected_files:
    # print("  ", f)

print("\nFiles found in current folder:")
if existing_files:
    for f in existing_files:
        print("  ", f)
else:
    print("  None found")

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
def read_data_for_ann(filenames, station_x, station_y):
    all_data = []
    print("\nReading files and collecting data for ANN...")

    for fname in filenames:
        try:
            with open(fname, 'r') as file:
                # Read headers and find indices
                headers = file.readline().strip().split()
                try:
                    x_idx = headers.index('X[m]')
                    y_idx = headers.index('Y[m]')
                    wspd_idx = headers.index('Wspd_2D[m/s]')
                    dir_idx = headers.index('Direction[º]')
                except ValueError as e:
                    print(f"Required header not found in {fname}: {e}")
                    continue
                
                # Skip the second line (dimensions)
                file.readline()
                
                # Read all data points for the current file
                file_points = []
                for line in file:
                    values = line.strip().split()
                    if len(values) > max(x_idx, y_idx, wspd_idx, dir_idx):
                        try:
                            point_x = float(values[x_idx])
                            point_y = float(values[y_idx])
                            point_wspd = float(values[wspd_idx])
                            point_dir = float(values[dir_idx])
                            file_points.append({'x': point_x, 'y': point_y, 'wspd': point_wspd, 'dir': point_dir})
                        except (IndexError, ValueError) as ve:
                            print(f"Error parsing data in {fname}: {ve}")
                            
            if not file_points:
                print(f"No valid data points found in {fname}. Skipping file.")
                continue

            # Find the closest data point to the station coordinates
            closest_point = None
            min_distance = float('inf')
            
            for point in file_points:
                distance = math.sqrt((point['x'] - station_x)**2 + (point['y'] - station_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
                    
            station_wspd = closest_point['wspd']
            station_dir = closest_point['dir']

            # Append data for all other points using the closest station's data
            for point in file_points:
                # Use a small epsilon to avoid including the station point itself in the training data
                if math.sqrt((point['x'] - closest_point['x'])**2 + (point['y'] - closest_point['y'])**2) > 0.001:
                    all_data.append([station_wspd, station_dir, point['x'], point['y'], point['wspd'], point['dir']])
                            
        except (IOError, ValueError) as e:
            print(f"Error processing {fname}: {e}")
            
    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data, columns=['station_wspd', 'station_dir', 'target_x', 'target_y', 'target_wspd', 'target_dir'])
    return df

# --- Main script logic ---
if existing_files:
    model_filename = station_name + '_ann_model.pkl'
 
    # Check if a trained model already exists
    # if os.path.exists(model_filename):
        # print("\nLoading pre-trained ANN model from file...")
        # with open(model_filename, 'rb') as f:
            # model = pickle.load(f)
        # print("Model loaded successfully.")
    # else:
        # Read and process data from files
    df = read_data_for_ann(existing_files, station_x, station_y)
    
    if not df.empty:
        print("\nSuccessfully collected data for training.")
        print(f"DataFrame head:\n{df.head()}")
        
        # Define features (X) and labels (y)
        X = df[['station_wspd', 'station_dir', 'target_x', 'target_y']]
        y = df[['target_wspd', 'target_dir']]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Print data split percentages
        total_samples = len(df)
        train_percent = len(X_train) / total_samples * 100
        test_percent = len(X_test) / total_samples * 100
        print(f"\nData used for training: {train_percent:.2f}% ({len(X_train)} samples)")
        print(f"Data used for validation: {test_percent:.2f}% ({len(X_test)} samples)")
        
        # Define the ANN model with a pipeline
        model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1, warm_start=True, random_state=42))
        
        print("\nStarting ANN training...")
        print("Press 's' and then 'Enter' to stop training early.")
        
        best_score = -float('inf')
        
        for epoch in range(1000): # Train for a maximum of 1000 epochs
            start_time = time.time()
            model.fit(X_train, y_train)
            
            # Evaluate the model
            current_score = model.score(X_test, y_test)
            
            # Check if this is the best model so far
            if current_score > best_score:
                best_score = current_score
                # Save the best model
                with open(model_filename, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Epoch {epoch + 1}: Score improved to {current_score:.4f}. Model saved.")
            else:
                print(f"Epoch {epoch + 1}: Current score {current_score:.4f} (Best: {best_score:.4f}).")
            
            # Check for user interruption
            if non_blocking_input() == 's':
                print("\nTraining interrupted by user. Using the best saved model.")
                break
        
        print("\nTraining complete.")
        # Load the best saved model
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

        # Calculate and save final results for the training and validation sets
        print("\nCalculating and saving final metrics...")
        
        # Predictions for training data
        y_train_pred = model.predict(X_train)
        
        # Predictions for validation data
        y_test_pred = model.predict(X_test)
        
        # Calculate validation metrics
        bias_wspd = np.mean(y_test_pred[:, 0] - y_test['target_wspd'])
        rmse_wspd = np.sqrt(np.mean((y_test_pred[:, 0] - y_test['target_wspd'])**2))
        bias_dir = np.mean(y_test_pred[:, 1] - y_test['target_dir'])
        rmse_dir = np.sqrt(np.mean((y_test_pred[:, 1] - y_test['target_dir'])**2))
        
        print(f"\nValidation Metrics:")
        print(f"  Wind Speed Bias: {bias_wspd:.4f} m/s")
        print(f"  Wind Speed RMSE: {rmse_wspd:.4f} m/s")
        print(f"  Wind Direction Bias: {bias_dir:.4f} degrees")
        print(f"  Wind Direction RMSE: {rmse_dir:.4f} degrees")

        # Combine data and predictions into DataFrames
        train_results = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        train_results['predicted_wspd'] = y_train_pred[:, 0]
        train_results['predicted_dir'] = y_train_pred[:, 1]
        train_results.to_csv(station_name + "_training_predictions.csv", index=False)
        print("Training predictions saved to 'training_predictions.csv'")

        test_results = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        test_results['predicted_wspd'] = y_test_pred[:, 0]
        test_results['predicted_dir'] = y_test_pred[:, 1]
        test_results.to_csv(station_name + "_validation_predictions.csv", index=False)
        print("Validation predictions saved to 'validation_predictions.csv'")

    else:
        print("\nNo data was collected from the files. Cannot train the model.")
        sys.exit()

    # --- Example prediction ---
    print("\n--- Example Prediction ---")
    
    # We'll use the first entry from the test set as an example
    df = read_data_for_ann(existing_files, station_x, station_y)
    X = df[['station_wspd', 'station_dir', 'target_x', 'target_y']]
    y = df[['target_wspd', 'target_dir']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sample_input = X_test.iloc[0].values.reshape(1, -1)
    actual_output = y_test.iloc[0].values
    
    # Predict the wind at the new location
    predicted_output = model.predict(sample_input)
    
    print(f"Input: {sample_input}")
    print(f"Actual Wind Speed/Direction: {actual_output[0]:.2f} m/s, {actual_output[1]:.2f} degrees")
    print(f"Predicted Wind Speed/Direction: {predicted_output[0][0]:.2f} m/s, {predicted_output[0][1]:.2f} degrees")
    
    export_ann_parameters(model, station_name + "_ann_parameters.txt")

    # generate python code to transpose data using the trained ann
    # it uses a template file and generates an output 
    replace_text_in_file("WTM_station_template.py", "WTM_" + station_name + "_ann.py", "<STATIONNAME>", station_name)
    # create the py script that modifies epw files using the ann
    replace_text_in_file("WTM_epw_template.py", "WTM_epw_" + station_name + ".py", "<ANNFILE>", "WTM_" + station_name + "_ann.py")
    
