#!/bin/bash

# Base directories
results_dir="results"
instances_dir="instance-files"
evaluation_dir="evaluation"

# Loop through each subfolder in the results directory
for folder in "$results_dir"/*; do
    if [[ -d "$folder" ]]; then
        folder_name=$(basename "$folder")

        instance_subdir=$(echo "$folder_name" | cut -d'_' -f1)
        instance_file_name="${instance_subdir}_problem.txt"

        cg_matrix_file="$folder/cg_matrix.csv"
        instance_file="$instances_dir/$instance_subdir/$instance_file_name"
        lower_bounds_file="$evaluation_dir/$folder_name/lower_bounds.csv"

        mkdir -p "$(dirname "$lower_bounds_file")"

        if [[ -f "$cg_matrix_file" && -f "$instance_file" ]]; then
            uv run ./evaluation/compute_lower_bounds.py "$cg_matrix_file" "$instance_file" "$lower_bounds_file"
        else
            echo "Skipping $folder: missing cg_matrix.csv or corresponding instance file."
        fi
    fi
done
