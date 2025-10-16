#!/bin/bash

results_dir="results"
evaluation_dir="evaluation"

output_file="./instance_analysis.txt"

uv run ./evaluation/analyze_instances.py "$results_dir" "$evaluation_dir" "$output_file"

sed 's/%/\\%/g' "$output_file"
