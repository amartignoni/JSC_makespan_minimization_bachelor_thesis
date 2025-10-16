import os
import sys
import pandas as pd
import numpy as np

results_dir = sys.argv[1]         # "results"
evaluation_dir = sys.argv[2]      # "evaluation"
genetic_dir = sys.argv[3]         # "genetic_results"
output_file = sys.argv[4]         # e.g. "evaluation/genetic_analysis.txt"

metrics = {}
instance_data = {}

# Step 1: collect all 5 runs per instance
for folder in os.listdir(genetic_dir):
    folder_path = os.path.join(genetic_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    parts = folder.split('_')
    if len(parts) != 5:
        continue

    prob = parts[2]
    instance_key = "_".join(parts[:4])  # TaiXX_problem_P_I
    run = parts[4]

    results_folder = os.path.join(results_dir, instance_key)
    evaluation_folder = os.path.join(evaluation_dir, instance_key)

    start_times_file = os.path.join(results_folder, "start_times_array.csv")
    metadata_file = os.path.join(results_folder, "metadata.csv")
    lower_bounds_file = os.path.join(evaluation_folder, "lower_bounds.csv")

    ga_results_folder = os.path.join(folder_path, "pop200_xoverLOX_mutMOVE_initSEMI")
    ga_results_file = os.path.join(ga_results_folder, "result.csv")

    if not (os.path.exists(start_times_file) and os.path.exists(metadata_file) and os.path.exists(lower_bounds_file) and os.path.exists(ga_results_file)):
        continue

    # Load solver metadata
    metadata_df = pd.read_csv(metadata_file, index_col=0, header=0, names=["key", "value"]).squeeze()
    solution = float(metadata_df.get('solution', np.nan))
    print("solution : ", solution)

    # Load lower bounds
    lb_data = pd.read_csv(lower_bounds_file, index_col=0, header=0).squeeze().to_dict()
    best_lower_bound = max(lb_data.values())
    print("lb_best : ", best_lower_bound)

    # Load GA result
    ga_df = pd.read_csv(ga_results_file)
    best_cmax = float(ga_df['best_cmax'].iloc[0])
    print("best_ga_cmax : ", best_cmax)
    elapsed_seconds = float(ga_df['elapsed_seconds'].iloc[0])
    stopping_condition = ga_df['stopping_condition'].iloc[0]

    print("GA gap : ", abs(best_cmax - solution) / abs(solution))

    # Store per instance
    if instance_key not in instance_data:
        instance_data[instance_key] = {
            'prob': prob,
            'start_times_file': start_times_file,
            'solution': solution,
            'best_lower_bound': best_lower_bound,
            'runs': []
        }

    instance_data[instance_key]['runs'].append({
        'best_cmax': best_cmax,
        'elapsed_seconds': elapsed_seconds,
        'stopping_condition': stopping_condition
    })

# Step 2: aggregate statistics (best + average of 5 runs)
for instance_key, inst in instance_data.items():
    prob = inst['prob']
    start_times_array = np.loadtxt(inst['start_times_file'], delimiter=',')
    size = f"{start_times_array.shape[0]}x{start_times_array.shape[1]}"

    solution = inst['solution']
    best_lower_bound = inst['best_lower_bound']
    runs = inst['runs']

    # Compute best run
    best_run = min(runs, key=lambda r: r['best_cmax'])
    best_cmax = best_run['best_cmax']
    elapsed_seconds = best_run['elapsed_seconds']
    stopping_condition = best_run['stopping_condition']

    # Compute averages over 5 runs
    rel_errors = []
    deviations = []
    avg_cmax_ga = 0

    # for r in runs:
    #     avg_cmax_ga += r['best_cmax']
      
    ##  5 is hardcoded, so not optimal --> quick fix for hand-in  
    avg_cmax_ga = sum([r['best_cmax'] for r in runs]) / 5 

    avg_rel_error_5runs = abs(solution - avg_cmax_ga) / abs(avg_cmax_ga) if avg_cmax_ga > 0 else 0
    avg_deviation_5runs = abs(avg_cmax_ga - best_lower_bound) / abs(avg_cmax_ga) if avg_cmax_ga > 0 else 0

    # Initialize stats
    if prob not in metrics:
        metrics[prob] = {}
    if size not in metrics[prob]:
        metrics[prob][size] = {
            # Best run metrics
            'rel_error_sum': 0,
            'optimal_count': 0,
            'cpu_time_sum': 0,
            'deviation_sum': 0,
            'no_improvement_count': 0,
            'count': 0,
            # Average-over-5 metrics
            'avg_rel_error_5runs_sum': 0,
            'avg_deviation_5runs_sum': 0
        }

    # Best run metrics
    rel_error_best = abs(solution - best_cmax) / abs(best_cmax) if best_cmax > 0 else 0
    metrics[prob][size]['rel_error_sum'] += rel_error_best

    if stopping_condition == "lower_bound":
        metrics[prob][size]['optimal_count'] += 1
    if stopping_condition == "no_improvement":
        metrics[prob][size]['no_improvement_count'] += 1

    metrics[prob][size]['cpu_time_sum'] += elapsed_seconds
    deviation_best = abs(best_cmax - best_lower_bound) / abs(best_cmax) if best_cmax > 0 else 0
    metrics[prob][size]['deviation_sum'] += deviation_best
    metrics[prob][size]['count'] += 1

    # Average-of-5 metrics
    metrics[prob][size]['avg_rel_error_5runs_sum'] += avg_rel_error_5runs
    metrics[prob][size]['avg_deviation_5runs_sum'] += avg_deviation_5runs

# Step 3: generate LaTeX tables
latex_tables = {}

for prob in ['0.2', '0.5', '0.8']:
    table = "\\begin{tabular}{lccccc|cc}\n"
    table += "\\multicolumn{1}{c}{} & \\multicolumn{5}{c}{\\textbf{Best}} & \\multicolumn{2}{c}{\\textbf{Average}} \\\\\n"
    table += "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}\n"
    table += "Problem Size & Rel Error & \\#Opt & \\#NoImp & CPU Time & Deviation & Rel Error$_{avg}$ & Deviation$_{avg}$ \\\\\n\\hline\n"

    if prob in metrics:
        sorted_sizes = sorted(metrics[prob].keys(), key=lambda s: tuple(map(int, s.split('x'))))
        for size in sorted_sizes:
            data = metrics[prob][size]
            count = data['count']
            #print(count)
            if count == 0:
                continue
            avg_rel_error_best = data['rel_error_sum'] / count
            avg_cpu_time = data['cpu_time_sum'] / count
            avg_deviation_best = data['deviation_sum'] / count
            avg_rel_error_5runs = data['avg_rel_error_5runs_sum'] / count
            avg_deviation_5runs = data['avg_deviation_5runs_sum'] / count
            optimal_count = data['optimal_count']
            no_improvement_count = data['no_improvement_count']

            # escape % for LaTeX
            fmt = lambda x: f"{x * 100:.2f}\\%"

            table += (
                f"{size} & {fmt(avg_rel_error_best)} & {optimal_count} & "
                f"{no_improvement_count} & {avg_cpu_time:.2f} & {fmt(avg_deviation_best)} & "
                f"{fmt(avg_rel_error_5runs)} & {fmt(avg_deviation_5runs)} \\\\\n"
            )

    table += "\\end{tabular}\n"
    latex_tables[prob] = table

with open(output_file, 'w') as f:
    for prob in ['0.2', '0.5', '0.8']:
        f.write(f"Probability {prob}:\n")
        f.write(latex_tables[prob])
        f.write("\n\n")

print(f"Analysis saved to {output_file}")
