import numpy as np
import argparse
import csv

def generate_undirected_graph(N, p=0.8):
    
    upper = np.triu(np.random.rand(N, N) <= p, k=1).astype(int)
    A = upper + upper.T
    return A

def save_to_csv(matrix, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate undirected graph adjacency matrix")
    parser.add_argument("N", type=int, help="Number of nodes")
    parser.add_argument("output_csv", type=str, help="Path to save CSV file")
    args = parser.parse_args()

    matrix = generate_undirected_graph(args.N)
    save_to_csv(matrix, args.output_csv)
    print(f"Adjacency matrix saved to {args.output_csv}")
