import os
import numpy as np
import pandas as pd

results_dir = "results"
evaluation_dir = "evaluation"
probabilities = ["0.2", "0.5", "0.8"]
lower_bound_keys = ["jlb", "mlb", "gwmin_lb", "gwmin2_lb", "upper_bound"]

count_per_size_prob = {p: {} for p in probabilities}
frequency_counts = {p: {} for p in probabilities}

for folder in os.listdir(evaluation_dir):

    parts = folder.split('_')

    print(parts)

    if len(parts) != 4:
        continue
    prob = parts[2]
    if prob not in probabilities:
        continue

    eval_folder = os.path.join(evaluation_dir, folder)
    lower_bounds_file = os.path.join(eval_folder, "lower_bounds.csv")
    start_times_file = os.path.join(results_dir, folder, "start_times_array.csv")

    if not os.path.exists(lower_bounds_file) or not os.path.exists(start_times_file):
        continue

    # lb_data = pd.read_csv(lower_bounds_file, index_col=0).squeeze().to_dict()
    lb_data = pd.read_csv(lower_bounds_file, index_col=0, header=0, names=["key", "value"]).squeeze().to_dict()
    start_times_array = np.loadtxt(start_times_file, delimiter=',')
    size = f"{start_times_array.shape[0]}x{start_times_array.shape[1]}"


    count_per_size_prob[prob][size] = count_per_size_prob[prob].get(size, 0) + 1

    max_value = max(lb_data.values())
    best_bounds = [k for k, v in lb_data.items() if v == max_value]

    if size not in frequency_counts[prob]:
        frequency_counts[prob][size] = {k: 0 for k in lower_bound_keys}

    for bound in best_bounds:
        frequency_counts[prob][size][bound] += 1

latex_output = ""

for prob in probabilities:
    latex_output += f"\n--- LaTeX Table for Probability {prob} ---\n\n"
    latex_output += "\\begin{tabular}{c|ccccc}\n"
    latex_output += "NxM & JLB & MLB & GWMIN LB & GWMIN2 LB & Solver ULB \\\\\n"
    latex_output += "\\hline\n"

    sorted_sizes = sorted(frequency_counts[prob].keys(), key=lambda s: tuple(map(int, s.split('x'))))

    for size in sorted_sizes:
        total = count_per_size_prob[prob][size]
        percentages = [
            100 * frequency_counts[prob][size].get(lb, 0) / total
            for lb in lower_bound_keys
        ]
        row = f"{size} & " + " & ".join(f"{p:.1f}\\%" for p in percentages) + " \\\\"
        latex_output += row + "\n"

    latex_output += "\\end{tabular}\n"

with open("lower_bounds_tables.txt", "w") as f:
    f.write(latex_output)
