import sys
import pickle
import numpy as np
import pandas as pd

# --- Command line argument handling ---
if len(sys.argv) != 5:
    print("Usage: python predict_wind.py <station_wspd> <station_dir> <target_x> <target_y>")
    # python WTM_STATIONNAME.py 3.58e+00  8.26e+01 6.582773e+05  7.158411e+06
    sys.exit(1)

try:
    # Store input parameters
    station_wspd = float(sys.argv[1])
    station_dir = float(sys.argv[2])
    target_x = float(sys.argv[3])
    target_y = float(sys.argv[4])
except ValueError:
    print("Error: All four input parameters must be numbers.")
    sys.exit(1)

# --- Load the pre-trained model ---
model_filename = "<STATIONNAME>" + "_ann_model.pkl"
try:
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Please train the ANN first using WTM.py.")
    sys.exit(1)

# --- Make a prediction ---
# Create a DataFrame with the same feature names used during training.
input_data = pd.DataFrame([[station_wspd, station_dir, target_x, target_y]], 
                           columns=['station_wspd', 'station_dir', 'target_x', 'target_y'])

predicted_output = model.predict(input_data)

# --- Print the results ---
# Print only the two predicted values, without any text or units.
print(f"{predicted_output[0][0]:.4f} {predicted_output[0][1]:.4f}")