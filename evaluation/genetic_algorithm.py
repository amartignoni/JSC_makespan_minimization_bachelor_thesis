import numpy as np
import random
import sys
import csv
from read_result_values import read_operations_array, read_times_array, read_cg_matrix

def load_problem(operations_file, times_file, cg_matrix_file):
    """Load the problem data from text files."""
    operations_array = read_operations_array(operations_file)
    times_array = read_times_array(times_file)
    cg_matrix = read_cg_matrix(cg_matrix_file)
    return operations_array, times_array, cg_matrix

import numpy as np

def non_delay_schedule(machine_operations, times_array, cg_matrix):
    
    "Generate a non-delay schedule that respects the order of operations and the constraint graph. "

    num_jobs = cg_matrix.shape[0]

    # Initialize machine ready times and job completion times
    machine_ready_time = {m: 0 for m in machine_operations}
    job_ready_time = [0] * num_jobs

    # Schedule to store (job, operation, start_time, end_time) for each machine
    schedule = {m: [] for m in machine_operations}

    while any(any(op[3] for op in ops) for ops in machine_operations.values()):  # Check if any operation is active
        
        # Select the machine with the lowest end time of its last scheduled operation
        machine = min(
            (m for m in machine_operations if any(op[3] for op in machine_operations[m])),
            key=lambda m: machine_ready_time[m]
        )


        operations = machine_operations[machine]
        for idx, (job, op_idx, processing_time, is_active) in enumerate(operations):
            if not is_active:
                continue

            # Calculate earliest start time considering machine and job readiness
            earliest_start = max(machine_ready_time[machine], job_ready_time[job])

            # Check for conflicts in the constraint graph
            conflict_delay = 0
            for other_job in range(num_jobs):
                if cg_matrix[job, other_job] == 1 and other_job != job:
                    for other_op_idx in range(len(times_array[other_job])):
                        if (other_job, other_op_idx) in {t[:2] for t in sum(schedule.values(), [])}:
                            other_job_end_time = next(t[3] for t in sum(schedule.values(), []) if t[:2] == (other_job, other_op_idx))
                            if earliest_start < other_job_end_time:
                                conflict_delay = max(conflict_delay, other_job_end_time - earliest_start)

            # Adjust start time for conflicts
            start_time = earliest_start + conflict_delay
            end_time = start_time + processing_time

            # Update schedule and ready times
            schedule[machine].append((job, op_idx, start_time, end_time))
            machine_ready_time[machine] = end_time
            job_ready_time[job] = end_time

            # Mark this operation as scheduled and activate the next operation for the job
            operations[idx] = (job, op_idx, processing_time, False)
            if op_idx + 1 < len(times_array[job]):
                for m, ops in machine_operations.items():
                    for i, (j, next_op_idx, next_proc_time, _) in enumerate(ops):
                        if j == job and next_op_idx == op_idx + 1:
                            ops[i] = (j, next_op_idx, next_proc_time, True)
                            break
            break

    # Calculate makespan (Cmax)
    Cmax = max(max(end_time for _, _, _, end_time in tasks) for tasks in schedule.values())

    return schedule, Cmax

def generate_random_solution(operations_array, times_array):
    """Generate a random solution."""
    machine_operations = {}
    num_jobs, num_ops = operations_array.shape
    for m in np.unique(operations_array):
        machine_operations[m] = [
            (i, j + 1, times_array[i, j])
            for i in range(num_jobs)
            for j in range(num_ops)
            if operations_array[i, j] == m
        ]
        random.shuffle(machine_operations[m])
    return machine_operations

def genetic_algorithm(
    operations_array, times_array, cg_matrix, P=10, Prob=1.0, max_evols=10000, stag_iter=100, lower_bound=0
):
    """Genetic algorithm implementation."""
    def evaluate_solution(machine_operations):
        return non_delay_heuristic(machine_operations, cg_matrix, times_array)

    def linear_order_crossover(parent1, parent2):
        child = {m: parent1[m][:] for m in parent1}
        for m in parent1:
            machine_ops = parent1[m]
            n_ops = len(machine_ops)
            if n_ops <= 2:
                continue
            p, q = sorted(random.sample(range(1, n_ops), 2))
            parent2_ops = parent2[m]
            child[m] = (
                machine_ops[:p]
                + [op for op in parent2_ops if op not in machine_ops[p:q]]
                + machine_ops[q:]
            )
        return child

    def mutate_solution(solution, Prob): #swap
        for m in solution:
            if random.random() < Prob:
                machine_ops = solution[m]
                if len(machine_ops) > 1:
                    i, j = random.sample(range(len(machine_ops)), 2)
                    machine_ops[i], machine_ops[j] = machine_ops[j], machine_ops[i]
        return solution

    population = [generate_random_solution(operations_array, times_array) for _ in range(P)]
    fitness = [evaluate_solution(sol)[1] for sol in population]
    best_Cmax = min(fitness)
    best_solution = population[fitness.index(best_Cmax)]
    stagnant_iterations = 0

    for _ in range(max_evols):
        if best_Cmax <= lower_bound or stagnant_iterations >= stag_iter:
            break

        sorted_population = [x for _, x in sorted(zip(fitness, population))]
        sorted_fitness = sorted(fitness)

        prob_first = [(2 * (i + 1)) / (P * (P + 1)) for i in range(P)]
        prob_second = [1 / P for _ in range(P)]

        parent1 = np.random.choice(sorted_population, p=prob_first)
        parent2 = np.random.choice(sorted_population, p=prob_second)

        child = linear_order_crossover(parent1, parent2)
        child = mutate_solution(child, Prob)

        child_Cmax, _ = evaluate_solution(child)
        if child_Cmax < sorted_fitness[P // 2]:
            replace_index = random.randint(0, P // 2 - 1)
            sorted_population[replace_index] = child
            sorted_fitness[replace_index] = child_Cmax

        if child_Cmax < best_Cmax:
            best_Cmax = child_Cmax
            best_solution = child
            stagnant_iterations = 0
        else:
            stagnant_iterations += 1

    return best_solution, best_Cmax

def save_schedule_to_csv(schedule, output_file):
    """Save the schedule to a CSV file."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Machine", "Job", "Operation", "Start Time", "End Time"])
        for machine, tasks in schedule.items():
            for (job, op, start, end) in tasks:
                writer.writerow([machine, job, op, start, end])

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <operations_file> <times_file> <cg_matrix_file>")
        return

    operations_file = sys.argv[1]
    times_file = sys.argv[2]
    cg_matrix_file = sys.argv[3]

    operations_array, times_array, cg_matrix = load_problem(operations_file, times_file, cg_matrix_file)

    best_solution, best_Cmax = genetic_algorithm(operations_array, times_array, cg_matrix)

    print(f"Best Cmax: {best_Cmax}")
    print("Best solution schedule saved to 'schedule.csv'")

    schedule, _ = non_delay_heuristic(best_solution, cg_matrix, times_array)
    save_schedule_to_csv(schedule, "schedule.csv")

if __name__ == "__main__":
    main()
