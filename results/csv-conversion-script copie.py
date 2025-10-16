import csv
import math

def parse_cg_matrix(filename):
    """
    Parse the cg_matrix from the file, collecting 0s and 1s from relevant lines
    """
    matrix_values = []
    in_matrix = False
    
    with open(filename, 'r') as f:
        for line in f:
            # Start collecting when we find the cg_matrix line
            if line.strip().startswith('cg_matrix:'):
                in_matrix = True
                continue
            
            # If in matrix section, remove all whitespace and collect 0s and 1s
            if in_matrix:
                # Remove all whitespace
                cleaned_line = ''.join(line.split())
                
                # Collect only 0s and 1s
                matrix_values.extend([int(char) for char in cleaned_line if char in '01'])
                
                # Stop collecting after processing the line with ']]'
                if ']]' in line:
                    break
    
    # Calculate matrix dimension
    matrix_size = int(math.sqrt(len(matrix_values)))
    
    # Sanity check
    if matrix_size * matrix_size != len(matrix_values):
        print(matrix_values)
        raise ValueError(f"Matrix size is not square. Total elements: {len(matrix_values)}, Expected perfect square")
    
    return matrix_values, matrix_size

def save_matrix_to_csv(matrix_values, matrix_size, output_filename='cg_matrix.csv'):
    """
    Save the matrix to a CSV file
    """
    with open(output_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Reshape the flat list into a 2D matrix
        for i in range(0, len(matrix_values), matrix_size):
            csv_writer.writerow(matrix_values[i:i+matrix_size])
    
    print(f"CG Matrix saved to {output_filename}")

def main(input_filename='results.txt'):
    """
    Main function to process the results file
    """
    matrix_values, matrix_size = parse_cg_matrix(input_filename)
    save_matrix_to_csv(matrix_values, matrix_size)

if __name__ == '__main__':
    main()
