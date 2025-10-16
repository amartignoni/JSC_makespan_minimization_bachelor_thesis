import os
import sys
import pandas as pd
import numpy as np

results_dir = sys.argv[1]
evaluation_dir = sys.argv[2]
output_file = sys.argv[3]

metrics = {}

for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    parts = folder.split('_')
    if len(parts) != 4:
        continue

    prob = parts[2]
    start_times_file = os.path.join(folder_path, "start_times_array.csv")
    metadata_file = os.path.join(folder_path, "metadata.csv")
    lower_bounds_file = os.path.join(evaluation_dir, folder, "lower_bounds.csv")

    if not (os.path.exists(start_times_file) and os.path.exists(metadata_file) and os.path.exists(lower_bounds_file)):
        continue

    start_times_array = np.loadtxt(start_times_file, delimiter=',')
    size = f"{start_times_array.shape[0]}x{start_times_array.shape[1]}"

    # Load metadata
    metadata_df = pd.read_csv(metadata_file, index_col=0, header=0, names=["key", "value"]).squeeze()
    optimal_status = int(metadata_df.get('optimal_status', 0))
    solution = float(metadata_df.get('solution', np.nan))
    cpu_time = float(metadata_df.get('cpu_time', np.nan))
    gap = float(metadata_df.get('gap', np.nan))

    # Load lower bounds
    lb_data = pd.read_csv(lower_bounds_file, index_col=0, header=0).squeeze().to_dict()
    best_bound = max(lb_data.values())

    if prob not in metrics:
        metrics[prob] = {}
    if size not in metrics[prob]:
        metrics[prob][size] = {
            'gap_sum': 0,
            'optimal_count': 0,
            'cpu_time_sum': 0,
            'best_bound_sum': 0,
            'deviation_sum': 0,
            'count': 0
        }


    metrics[prob][size]['gap_sum'] += gap
    metrics[prob][size]['optimal_count'] += optimal_status
    metrics[prob][size]['cpu_time_sum'] += cpu_time
    metrics[prob][size]['best_bound_sum'] += best_bound
    if best_bound > 0 and not np.isnan(solution):
        deviation = (solution - best_bound) / solution
    else:
        deviation = 0
    metrics[prob][size]['deviation_sum'] += deviation
    metrics[prob][size]['count'] += 1

# Generate LaTeX tables
latex_tables = {}

for prob in ['0.2', '0.5', '0.8']:
    table = "\\begin{tabular}{lccccc}\n"
    table += "Problem Size & Avg Gap & NB Optimal & Avg CPU Time & Avg Best Bound & Avg Deviation \\\\\n\\hline\n"

    if prob in metrics:
        sorted_sizes = sorted(metrics[prob].keys(), key=lambda s: tuple(map(int, s.split('x'))))
        for size in sorted_sizes:
            data = metrics[prob][size]
            count = data['count']
            avg_gap = data['gap_sum'] / count if count else 0
            optimal_count = data['optimal_count']
            avg_cpu_time = data['cpu_time_sum'] / count if count else 0
            avg_best_bound = data['best_bound_sum'] / count if count else 0
            avg_deviation = data['deviation_sum'] / count if count else 0

            table += f"{size} & {avg_gap:.2%} & {optimal_count} & {avg_cpu_time:.2f} & {avg_best_bound:.2f} & {avg_deviation:.2%} \\\\\n"

    table += "\\end{tabular}\n"
    latex_tables[prob] = table

with open(output_file, 'w') as f:
    for prob in ['0.2', '0.5', '0.8']:
        f.write(f"Probability {prob}:\n")
        f.write(latex_tables[prob])
        f.write("\n\n")

print(f"Analysis saved to {output_file}")
