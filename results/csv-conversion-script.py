import csv
import os
import sys
import numpy as np
from math import isqrt

def preprocess_results_to_csv(filepath, output_prefix):
    """
    Simplified process for handling the results file:
    1. Import the file and clean it by removing all unnecessary whitespace.
    2. Parse the cleaned data to extract `cg_matrix`, `start_times_array`, and metadata.
    3. Save them into separate CSV files, ensuring arrays are square and converting numbers to integers when possible.
    """
    def maybe_convert_to_int(number):
        """Convert a float to int if it's effectively an integer."""
        return int(number) if number == int(number) else number

    def reshape_to_square(array):
        """Reshape a 1D or malformed 2D array into a square 2D array."""
        total_elements = array.size
        side_length = isqrt(total_elements)  # Integer square root
        if side_length * side_length != total_elements:
            raise ValueError("The total number of elements is not a perfect square.")
        return array.reshape((side_length, side_length))

    # Resolve absolute paths
    filepath = os.path.abspath(filepath)
    output_prefix = os.path.abspath(output_prefix)

    print(f"Processing file: {filepath}")
    print(f"Output folder: {output_prefix}")

    # Ensure the output folder exists
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)

    # Step 1: Read and clean the file
    with open(filepath, 'r') as file:
        raw_data = file.read()

    # Remove all unnecessary whitespace (tabs, carriage returns, etc.)
    cleaned_data = ' '.join(raw_data.split())

    # Step 2: Parse the cleaned data
    data = {}
    current_key = None
    matrix_lines = []

    for line in cleaned_data.split(' '):
        if line.endswith(":"):  # New key
            # Save previous matrix if applicable
            if current_key == "cg_matrix" and matrix_lines:
                data[current_key] = np.array(matrix_lines)  # Convert to 2D array
                matrix_lines = []
            
            current_key = line[:-1]  # Remove the colon
            if current_key not in data:
                data[current_key] = None  # Initialize the key

        elif current_key == "cg_matrix":
            if line.startswith("[") and line.endswith("]]"):  # Final row of the matrix
                row = [maybe_convert_to_int(float(x)) for x in line.strip('[]').split()]
                matrix_lines.append(row)
                data[current_key] = np.array(matrix_lines)  # Convert to NumPy array after all rows
                current_key = None  # End of matrix
            elif line.startswith("[") or line.endswith("]"):  # Part of the matrix
                row = [maybe_convert_to_int(float(x)) for x in line.strip('[]').split()]
                matrix_lines.append(row)

        elif current_key == "start_times_array":
            if line.startswith("[") and line.endswith("]]"):  # Final row of the array
                row = [maybe_convert_to_int(float(x)) for x in line.strip('[]').split()]
                if current_key not in data or data[current_key] is None:
                    data[current_key] = []
                data[current_key].append(row)
                data[current_key] = np.array(data[current_key])  # Convert to NumPy array after all rows
                current_key = None
            elif line.startswith("[") or line.endswith("]"):  # Part of the array
                row = [maybe_convert_to_int(float(x)) for x in line.strip('[]').split()]
                if current_key not in data or data[current_key] is None:
                    data[current_key] = []
                data[current_key].append(row)

        elif current_key:  # Handle metadata
            try:
                data[current_key] = maybe_convert_to_int(eval(line))
            except:
                data[current_key] = line

    # Step 3: Save the data
    # Save cg_matrix
    if "cg_matrix" in data and isinstance(data["cg_matrix"], np.ndarray):
        try:
            cg_matrix = reshape_to_square(data["cg_matrix"])
            cg_matrix_path = os.path.join(output_prefix, "cg_matrix.csv")
            np.savetxt(cg_matrix_path, cg_matrix, delimiter=",", fmt="%s")
            print(f"  - Saved: {cg_matrix_path}")
        except ValueError as e:
            print(f"Error reshaping cg_matrix: {e}")

    # Save start_times_array
    if "start_times_array" in data and isinstance(data["start_times_array"], np.ndarray):
        try:
            start_times_array = reshape_to_square(data["start_times_array"])
            start_times_path = os.path.join(output_prefix, "start_times_array.csv")
            np.savetxt(start_times_path, start_times_array, delimiter=",", fmt="%s")
            print(f"  - Saved: {start_times_path}")
        except ValueError as e:
            print(f"Error reshaping start_times_array: {e}")

    # Save metadata
    metadata = {k: v for k, v in data.items() if k not in {"cg_matrix", "start_times_array"}}
    if metadata:
        metadata_path = os.path.join(output_prefix, "metadata.csv")
        with open(metadata_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Key", "Value"])
            for key, value in metadata.items():
                writer.writerow([key, value])
        print(f"  - Saved: {metadata_path}")
    else:
        print("No metadata to save.")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv-conversion-script.py <input_file> <output_folder>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    preprocess_results_to_csv(input_file, output_folder)
