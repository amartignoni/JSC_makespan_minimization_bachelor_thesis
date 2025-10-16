import numpy as np

def parse_results_file_for_cg_matrix(filepath):
    
    with open(filepath, 'r') as file:
        lines = file.readlines()

    matrix_lines = []
    is_matrix = False

    for line in lines:
        line = line.strip()

        # Skip the lines before 'cg_matrix:'
        if line.startswith("cg_matrix:"):
            is_matrix = True  # Start reading the matrix
            continue  # Skip the 'cg_matrix:' line itself

        if is_matrix:
            # Check if the line ends with ']]', which is part of the last row
            if line.endswith("]]"):
                # Remove the closing brackets and split into numbers
                row = list(map(float, line.strip('[]').split()))
                matrix_lines.append(row)
                break  # Stop reading after this line
            elif line.startswith("[") and line.endswith("]"):  # A complete row
                row = list(map(float, line.strip('[]').split()))
                matrix_lines.append(row)
            elif line:  # Part of a row
                row = list(map(float, line.strip('[]').split()))
                matrix_lines[-1].extend(row)  # Append to the previous row

    print(matrix_lines)

    return np.array(matrix_lines)

def parse_file_to_arrays(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    times, operations = [], []
    section = None

    for line in lines:
        line = line.strip()
        if line == "Times":
            section = "Times"
        elif line == "Machines":
            section = "operations"
        elif line:
            numbers = list(map(int, line.split()))
            if section == "Times":
                times.append(numbers)
            elif section == "operations":
                operations.append(numbers)

    return np.array(times), np.array(operations)

