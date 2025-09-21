import sys
import os
import subprocess

# This program edits wind data on EPW files to account for changes in rouughness and topography.
# It relies on trained ANN for the location, developed using Windproject_site results and the programa WTM

def modify_epw_file(filename, project_site_x, project_site_y):
    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"File '{filename}' not found.")
        return

    # Create output filename by appending 'ANN' before the extension
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}_ANN{ext}"

    with open(filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(',')

            # EPW data lines typically have 35 fields; wind direction is at index 20, wind speed at index 21
            if len(parts) >= 35 and parts[0].isdigit():
                try:
                    print("------")
                    wind_dir = float(parts[20])
                    print("wind_dir", wind_dir)
                    wind_speed = float(parts[21])
                    print("wind_speed", wind_speed)
                    wind_dir = wind_dir % 360
                    print("wind_dir", wind_dir)
                    print("------")

                    # Call external script with parameters
                    result = subprocess.run(
                        ["python", "<ANNFILE>", str(wind_speed), str(wind_dir), str(project_site_x), str(project_site_y)],
                        capture_output=True,
                        text=True
                    )

                    # Check if execution was successful
                    if result.returncode != 0:
                        print("Error running <ANNFILE>:")
                        print(result.stderr)
                        sys.exit(1)

                    # Capture output (assuming the script prints two values separated by whitespace or newline)
                    outputs = result.stdout.strip().split()
                    print("outputs: ", outputs)

                    if len(outputs) < 2:
                        print("Error: Expected at least two outputs, got:", outputs)
                        sys.exit(1)

                    wind_speed_trans_s, wind_dir_trans_s = outputs[0], outputs[1]
                    print("parts output: ", wind_speed_trans_s, "   ", wind_dir_trans_s)
                    wind_speed_trans = float(wind_speed_trans_s)
                    wind_dir_trans = float(wind_dir_trans_s)
                    wind_dir_trans = wind_dir_trans % 360
                    print("^^^^^^")

                    parts[20] = f"{wind_dir_trans:.1f}"
                    parts[21] = f"{wind_speed_trans:.1f}"
                    print(parts[20])
                    print(parts[21])

                except ValueError:
                    pass

            outfile.write(','.join(parts) + '\n')

    print(f"Modified EPW file saved as '{output_filename}'.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python  <ANNFILE> ",filename, " ",project_site_x, " ", project_site_y)
    else:
        epw_filename = sys.argv[1]
        project_site_x = float(sys.argv[2])
        project_site_y = float(sys.argv[3])
        modify_epw_file(epw_filename, project_site_x, project_site_y)
