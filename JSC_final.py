import numpy as np
import sys
import random
import gurobipy as gp
from gurobipy import GRB
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from itertools import product
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize) # prevent truncated output in result files

# Load problem data function
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

# Gantt chart plotting function
def plot_gantt_chart(start_times_array, times_array, operations_array, save_path):
    num_jobs, num_ops = start_times_array.shape
    end_times = np.array([start_times_array[job, op].X + times_array[job, op] for job in range(num_jobs) for op in range(num_ops)])
    max_end_time = end_times.max()
    colors = plt.get_cmap('tab20', num_jobs)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    for job in range(num_jobs):
        for op in range(num_ops):
            start_time = start_times_array[job, op].X
            duration = times_array[job, op]
            machine = operations_array[job, op]
            ax.barh(machine, duration, left=start_time, color=colors(job % num_jobs), edgecolor='black')
    
    ax.set_yticks(np.arange(1, num_ops + 1), labels=[f'Machine {i + 1}' for i in range(num_ops)])
    #ax.set_yticklabels([f'Machine {i + 1}' for i in range(num_ops)], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('Execution Time', fontsize=14)
    ax.set_xlim(0, max_end_time)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors(job), edgecolor='black') for job in range(num_jobs)]
    labels = [f'Task {job + 1}' for job in range(num_jobs)]
    ncols = max(1, num_jobs // 25)
    ax.legend(handles, labels, title="Tasks", loc='upper left', bbox_to_anchor=(1, 1), ncol=ncols)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# generate pairs of coordinates for all pairs of operations
# used for creating constraints 
def gen_coordinate_pairs(x, y):
    for x1 in range(x):
        for y1 in range(y):
            for x2 in range(x1, x):
                y2_start = y1 + 1 if x1 == x2 else 0
                for y2 in range(y2_start, y):
                    yield ((x1, y1), (x2, y2))


## main program

# save parameters and create save folder
filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]
probability = float(sys.argv[2])
run_number = int(sys.argv[4])
folder_name = f"{filename}_{probability}_{run_number}"

os.makedirs(folder_name, exist_ok=True)

# Parse input file and set up parameters
times_array, operations_array = parse_file_to_arrays(sys.argv[1])

# compute big M
big_M = times_array.sum() + 1

# create conflict graph adjacency matrix
cg_matrix = np.zeros((operations_array.shape[0], operations_array.shape[0]))
for i in range(operations_array.shape[0]):
    for j in range(operations_array.shape[1]):
        if i != j:
            cg_matrix[i, j] = 1 if random.random() < probability else 0
            cg_matrix[j, i] = cg_matrix[i, j]

# start model definition for the solver
try:
    m = gp.Model("JSC")
    # set time limit and maximum number of threads the solver will use in paralle
    m.setParam('TimeLimit', int(sys.argv[3]))
    m.setParam('Threads', 16)

    # create start times decision variables
    start_times_array = m.addMVar(operations_array.shape, vtype=GRB.CONTINUOUS)
    
    # create objective variable and set objective function
    t_end = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    m.setObjective(t_end, GRB.MINIMIZE)

    # binary variables for machine constraints and job conflict constraints,
    # used for big M notation
    machines_bin_vars, cg_bin_vars = [], []

    # create precendence, machine and job conflict contraints
    for (x_pos, y_pos) in gen_coordinate_pairs(*operations_array.shape):
        
        # precedence constraints
        if x_pos[0] == y_pos[0]:
            m.addConstr(start_times_array[y_pos] - start_times_array[x_pos] >= times_array[x_pos])
        
        else:

            # machine constraints
            if operations_array[x_pos] == operations_array[y_pos]:

                machines_bin_vars.append(m.addVar(vtype=GRB.BINARY))
                m.addConstr(start_times_array[y_pos] - start_times_array[x_pos] >=
                            times_array[x_pos] + (machines_bin_vars[-1] - 1) * big_M)
                m.addConstr(start_times_array[x_pos] - start_times_array[y_pos] >=
                            times_array[y_pos] - machines_bin_vars[-1] * big_M)
            
            # job conflict contraints
            elif cg_matrix[x_pos[0], y_pos[0]]:
                cg_bin_vars.append(m.addVar(vtype=GRB.BINARY))
                m.addConstr(start_times_array[y_pos] - start_times_array[x_pos] >=
                            times_array[x_pos] + (cg_bin_vars[-1] - 1) * big_M)
                m.addConstr(start_times_array[x_pos] - start_times_array[y_pos] >=
                            times_array[y_pos] - cg_bin_vars[-1] * big_M)

    # add non-negativity constraints for all operations,
    # and constraints that ensure the final dummy task is scheduled after the last
    # operation of each task
    for i in range(operations_array.shape[0]):
        m.addConstr(start_times_array[i, 0] >= 0)
        m.addConstr(t_end - start_times_array[i, -1] >= times_array[i, -1])

    # start optimization of the problem
    m.optimize()

    # Collect and save results
    results = {
        "optimal_status": 1 if m.status == GRB.OPTIMAL else 0,
        "solution": m.ObjVal if m.SolCount > 0 else "N/A",
        "cpu_time": m.Runtime,
        # gap : abs(bestBound - bestSolution) / abs(bestSolution)
        "gap": m.MIPGap if m.MIPGap is not None else "N/A",
        "upper_bound": m.ObjVal if m.ObjBound is not None else "N/A",
        "lower_bound": m.ObjBound if m.ObjVal is not None else "N/A",
        "cg_matrix": cg_matrix,
        "start_times_array": np.array([[start_times_array[j, k].X for k in range(start_times_array.shape[1])]
                                       for j in range(start_times_array.shape[0])])
    }

    # Save cg_matrix to CSV
    cg_matrix_file = os.path.join(folder_name, "cg_matrix.csv")
    np.savetxt(cg_matrix_file, results["cg_matrix"], delimiter=",", fmt="%s")

    # Save start_times_array to CSV
    start_times_file = os.path.join(folder_name, "start_times_array.csv")
    np.savetxt(start_times_file, results["start_times_array"], delimiter=",", fmt="%s")
    
    # Save metadata to a text file
    results_file = os.path.join(folder_name, "results.txt")
    with open(results_file, "w") as f:
        for key, value in results.items():
            if isinstance(value, (np.ndarray, list)):  # If the value is an array, skip the type conversion
                continue
            f.write(f"{key}: {value}\n")

    # Save the Gantt chart
    gantt_chart_path = os.path.join(folder_name, f"{folder_name}.png")
    plot_gantt_chart(start_times_array, times_array, operations_array, save_path=gantt_chart_path)

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")
