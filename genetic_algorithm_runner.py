#!/usr/bin/env python3

import os
import argparse
import csv
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# parse text files for times and operations arrays
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

    return np.array(times, dtype=int), np.array(operations, dtype=int)

## create a non-delay schedule from a dictionary op
def non_delay_schedule(machine_operation_order, times_array, cg_matrix, machine_ops):
    
    # numbers of machines and jobs
    num_jobs = times_array.shape[0]
    num_ops = times_array.shape[1]

    # Ensure mapping keys are plain ints
    job_op_to_machine = {
        (int(j), int(o)): int(m)
        for (j, o), m in machine_ops.items()
    }

    # earliest start times and list of operations that have no unscheduled predecessors (active)
    earliest = {}   # (job, op) -> est
    active = []     

    # initialize: first op of each job
    for j in range(num_jobs):
        first = (int(j), 0)
        earliest[first] = 0
        active.append(first)

    schedule_by_machine = {int(m): [] for m in machine_ops.values()}
    scheduled_set = set()
    last_end_time = 0

    # helper: index in machine list
    def machine_index_of(data, m, op_tuple):

        for i, tup in enumerate(data[m-1]):

            if tup == op_tuple: 
                return i

        return None

    # while there are operations in the active set
    while active:

        # find minimum Earliest Starting Time (EST) among active ops
        est_values = [earliest[operation] for operation in active]
        min_est = min(est_values)

        # further shrink the list by only keeping operations with the minimum EST
        eligible = [(int(j), int(o)) for j, o in active if earliest[(int(j), int(o))] == min_est]

        # for each machine, keep only the eligible op that appears earliest in that machine's list
        per_machine_choice = {}
        
        for op in eligible:
            
            j, o = op
            m = job_op_to_machine.get(op, None)
            
            if m is None:
                continue
            
            idx = machine_index_of(machine_operation_order, (m+1), op)
            
            prev = per_machine_choice.get(m)
            
            if prev is None:
                
                per_machine_choice[m] = (op, idx)

            if prev is not None:  
                
                if idx < b:
                    per_machine_choice[m] = (op, idx)

        # final list of candidates for scheduling
        filtered = [val[0] for val in per_machine_choice.values()]
        
        if not filtered:
            
            filtered = eligible

        # randomly choose one among filtered
        chosen = random.choice(filtered)
        
        job_chosen_id, operation_chosen_id = chosen
        machine_chosen = job_op_to_machine[chosen]

        # schedule at its EST (not earlier)
        start = int(earliest[chosen])
        duration = int(times_array[job_chosen_id, operation_chosen_id])
        end = start + duration

        # record schedule
        schedule_by_machine[machine_chosen].append((job_chosen_id, operation_chosen_id, start, end))
        scheduled_set.add((job_chosen_id, operation_chosen_id))
        last_end_time = max(last_end_time, end)

        # remove chosen from active
        active.remove(chosen)
        earliest.pop(chosen, None)

        # add successor at end of active list if exists
        if operation_chosen_id + 1 < num_ops:
            
            succ = (job_chosen_id, operation_chosen_id + 1)
            
            if succ not in earliest and succ not in scheduled_set:
                earliest[succ] = 0
                active.append(succ)

        # update EST of remaining active ops based only on this scheduled op
        for op in active:
            
            job_active_id, operation_active_id = int(op[0]), int(op[1])
            
            # check if current op and scheduled op belong to the same job
            same_job = (job_active_id == job_chosen_id)
            
            # check if current op and scheduled op belong on the same machine
            machine_active = job_op_to_machine.get((job_active_id, operation_active_id), None)
            same_machine = (machine_active == machine_chosen)
            
            # check if current op and scheduled op belong to jobs that are in conflict
            conflict = False
            
            if 0 <= job_active_id < cg_matrix.shape[0] and 0 <= job_chosen_id < cg_matrix.shape[0]:
                conflict = (int(cg_matrix[job_active_id, job_chosen_id]) == 1)
            
            # update EST of current op if it conflicts in any way with the scheduled op
            if same_job or same_machine or conflict:
                earliest[(job_active_id, operation_active_id)] = max(earliest.get((job_active_id, operation_active_id), 0), last_end_time)

    # compute makespan
    cmax = 0
    f
    for ops in schedule_by_machine.values():
        for (_, _, s, e) in ops:
            if e > cmax:
                cmax = e

    return schedule_by_machine, int(cmax)

## main genetic algorithm class

class GeneticAlgorithm:
    def __init__(self, times_array, operations_array, cg_matrix,
                 lower_bounds_file=None, population_size=200, max_steps=1000000,
                 max_steps_no_improve=1000, crossover_method="LOX",
                 mutation_method="SWAP", mutation_probability=0.2,
                 output_base_dir=".", init_method="RANDOM", max_cpu_time=1200):

        # store progres of cmax for plotting
        self.cmax_progress = []

        # maximum cpu time for large instances
        self.max_cpu_time = max_cpu_time

        self.times_array = np.array(times_array, dtype=int)
        self.operations_array = np.array(operations_array, dtype=int)
        self.cg_matrix = np.array(cg_matrix, dtype=int)

        # mapping of (job_id, op_id) to machine_id for all operations
        self.job_op_to_machine = {
            (job_id, op_id): int(self.operations_array[job_id][op_id]-1)
            for job_id in range(len(self.operations_array))
            for op_id in range(len(self.operations_array[job_id]))
        }

        self.num_jobs, self.num_operations = self.times_array.shape
        self.num_machines = self.operations_array.shape[0]

        self.population_size = int(population_size)
        self.max_steps = int(max_steps)
        self.max_steps_no_improve = int(max_steps_no_improve)

        self.crossover_method = str(crossover_method).upper()
        self.mutation_method = str(mutation_method).upper()
        self.mutation_probability = float(mutation_probability)
        self.init_method = str(init_method).upper()

        self.output_base_dir = output_base_dir
        self.lower_bound = self._load_lower_bound(lower_bounds_file)

        # population: list of chromosomes where chromosome is dict {machine_id: [(job,op), ...]}
        self.population = []
        self.population_scores = []  # parallel list of Cmax
        self.best_schedule = None
        self.best_cmax = float('inf') # best Cmax to infinite as placeholder

    # load lower bounds file
    def _load_lower_bound(self, filepath):

        if filepath and os.path.exists(filepath):

            try:
                
                df = pd.read_csv(filepath)
                if "Value" in df.columns:
                    return float(df["Value"].max())

            except Exception:
                pass

        # if not found, don't use a lower bound
        return 0.0

    # make deterministric chromosomes for 
    def _build_heuristic_individuals(self, base):
        
        heuristic_individuals = []
        
        # compute the degree of each job vertex in the conflict graph
        job_degrees = {j: int(np.sum(self.cg_matrix[j])) for j in range(self.num_jobs)}

        for key, reverse in [("p", False), ("p", True), ("c", False), ("c", True),
                             ("c/p", False), ("c/p", True)]:
            chromo = {}
            
            # treat each subchromosome
            for m in base.keys():

                ops = base[m]

                # operations sorted by processing time in ascending and descending order
                if key == "p":
                    ops_sorted = sorted(ops, key=lambda x: self.times_array[x[0], x[1]], reverse=reverse)

                # operations sorted by job vertex conflict degree in ascending and descending order
                elif key == "c":
                    ops_sorted = sorted(ops, key=lambda x: job_degrees[x[0]], reverse=reverse)

                # operations sorted by the ratio of the two previous factors, in ascending and descending order
                elif key == "c/p":
                    ops_sorted = sorted(ops, key=lambda x: job_degrees[x[0]] / max(1, self.times_array[x[0], x[1]]),
                                        reverse=reverse)
                chromo[m] = ops_sorted

            heuristic_individuals.append(chromo)

        return heuristic_individuals

    # create base chromosome
    # we go from the operations sorted by job to the operations sorted by machines
    def _build_operation_dict(self): # complexity : O(mn), so somewhat linear
        
        # create dict
        base_machine_list = {}
        
        # populate dict with machine and operations list KV-couples
        for m in range(self.operations_array.shape[1]): 
            
            base_machine_list[m] = []

        # add each operation ((job,op) tuple) to the list of each machine
        for job in range(self.operations_array.shape[0]): 
            
            for operation in range(self.operations_array.shape[1]):
            
                machine = self.operations_array[job,operation] - 1
                base_machine_list[machine].append((job, operation))

        return base_machine_list

    ### Crossover operators

    # PMX
    # applied on a single 
    def _pmx(self, p1, p2, number_of_children=2):
        
        size = len(p1)
    
        # Select random crossover segment
        start, end = sorted(random.sample(range(size), 2))
    
        # Initialize children 
        child1 = [None] * size
        child2 = [None] * size
    
        # Copy the crossover segment from the opposite parent
        child1[start:end] = p2[start:end]
        child2[start:end] = p1[start:end]
    
        # Build allele mapping for each child
        mapping1 = {p2[i]: p1[i] for i in range(start, end)}
        mapping2 = {p1[i]: p2[i] for i in range(start, end)}
    
        # Complete first child from p1
        # only check elements outside 
        for i in list(range(0, start)) + list(range(end, size)):
            
            val = p1[i]
            
            # if allele is already in chromosome, take as the next allele
            # the allele that is in at the same place is p1 as the current allele
            # occupies in p2
            while val in child1[start:end]:
            
                val = mapping1[val]
            
            child1[i] = val
    
        # Complete second child from p2
        # identical as previous for loop with p1/p2 and child1/child2 swapped
        for i in list(range(0, start)) + list(range(end, size)):
            
            val = p2[i]
            
            
            while val in child2[start:end]:
            
                val = mapping2[val]
            
            child2[i] = val
    
        # Return one or both children
        if number_of_children == 1:

            return child1 if random.random() < 0.5 else child2
        
        else:
        
            return child1, child2


    # Order crossover
    def _ox1(self, p1, p2):
        
        size = len(a)
        
        start, end = sorted(random.sample(range(size), 2))
        
        # swapped segments
        hole_p1 = p1[start:end+1]
        hole_p2 = p1[start:end+1]

        # take the remaining elements in order from other parent chromosome
        fill_p1 = [allele for allele in p2 if allele not in hole_p1]
        fill_p2 = [allele for allele in p1 if allele not in hole_p2]

        # fill-in the rest of the children
        child1 = fill_p1[start:] + hole_p1 + fill_p1[:start]
        child2 = fill_p2[start:] + hole_p2 + fill_p2[:start]
        
        return child1, child2

    # Linear order crossover
    # same as order crossover, except the creation of the children
    def _lox(self, a, b):
        
        size = len(a)
        
        start, end = sorted(random.sample(range(size), 2))
        
        # swapped segments
        swap_p1 = p1[start:end+1]
        swap_p2 = p2[start:end+1]

        # take the remaining elements in order from other parent sub-chromosome
        fill_p1 = [allele for allele in p2 if allele not in swap_p1]
        fill_p2 = [allele for allele in p1 if allele not in swap_p2]

        # fill-in the rest of the children
        child1 = fill_p1[:start] + swap_p1 + fill_p1[start:]
        child2 = fill_p2[:start] + swap_p2 + fill_p2[start:]
        
        return child1, child2

    # higher-order function to apply the correct crossover to the whole chromosome
    def apply_crossover(self, parent1, parent2):
        
        # each child chrosome is a mapping "machine -> ordered list of operations"
        child1 = {}
        child2 = {}
        
        # each pair machine subchromosome must undergo crossover
        for m in parent1.keys():

            subchromo1 = parent1[m]
            subchromo2 = parent2[m]
            
            # apply chosen crossover to the subchromosome pair
            if self.crossover_method == "PMX":
                a, b = self._pmx(subchromo1, subchromo2)

            elif self.crossover_method == "OX1":
                a, b = self._ox1(subchromo1, subchromo2)

            elif self.crossover_method == "LOX":
                a, b = self._lox(subchromo1, subchromo2)

            child1[m] = a
            child2[m] = b

        return child1, child2

    # function to mutate a chromosome
    def apply_mutation(self, chromo):
        
        for m in chromo.keys():
        
            subchromo = chromo[m]
        
            if len(subchromo) < 2:
                continue
        
            # mutate only if probability is reached
            if random.random() < self.mutation_probability:

                # SWAP mutation, just exchange two alleles
                if self.mutation_method == "SWAP":

                    i, j = random.sample(range(len(subchromo)), 2)
                    subchromo[i], subchromo[j] = subchromo[j], subchromo[i]

                # MOVE mutation, remove an allele and insert it in another place
                elif self.mutation_method == "MOVE":

                    i, j = random.sample(range(len(subchromo)), 2)
                    elem = subchromo.pop(i)
                    subchromo.insert(j, elem)

                chromo[m] = subchromo

        return chromo

    # higher-order wrapper for chromosome evaluation, to keep code cleaner-ish
    def evaluate(self, chromosome):
        schedule_by_machine, cmax = non_delay_schedule(chromosome, self.times_array, self.cg_matrix, self.job_op_to_machine)
        return schedule_by_machine, cmax

    # create the population
    def initialize_population(self):

        # unpermutated base individual, operations sorted by machine
        base = self._build_operation_dict()
        
        # list of individuals and corresponding makespans
        self.population = []
        self.population_scores = []
        
        # remember the best schedule and its makespan
        self.best_schedule = None
        self.best_cmax = float('inf')
        
        # keep track of all makespans in the population, 
        # we want a population with pairwise unique makespans
        cmax_set = set()

        # if population initiation method is semi-heuristic
        if self.init_method == "SEMI":

            for chromo in self._build_heuristic_individuals(base):
                sched, cmax = self.evaluate(chromo)
                
                self.population.append(chromo)
                self.population_scores.append(cmax)
                
                if cmax not in cmax_set:
                    cmax_set.add(cmax)
                    
                if cmax < self.best_cmax:
                    self.best_cmax = cmax
                    self.best_schedule = sched

        # fill in rest of population randomly
        while len(self.population) < self.population_size:
            
            chromo = {m: random.sample(base[m], len(base[m])) for m in base.keys()}
            
            sched, cmax = self.evaluate(chromo)
            
            # ensure unique cmax
            if cmax in cmax_set:
                continue

            self.population.append(chromo)
            self.population_scores.append(cmax)
            cmax_set.add(cmax)
            
            if cmax < self.best_cmax:
                
                self.best_cmax = cmax
                self.best_schedule = sched

    # -----------------------
    # run main loop
    # -----------------------
    def run(self, output_base_dir=None, instance_name=None, run_id=1, verbose=True):
        
        # create folders to save the data
        if output_base_dir is not None:
            self.output_base_dir = output_base_dir

        folder_suffix = f"pop{self.population_size}_xover{self.crossover_method}_mut{self.mutation_method}_init{self.init_method}"
        
        if instance_name:
            self.output_dir = os.path.join(self.output_base_dir, f"{instance_name}_run{run_id}", folder_suffix)
        
        else:
            self.output_dir = os.path.join(self.output_base_dir, f"run{run_id}", folder_suffix)
        
        os.makedirs(self.output_dir, exist_ok=True)

        # start clock
        start_time = time.time()

        # create population
        self.initialize_population()

        # append best cmax from population, after population is generated 
        curr_time = time.time()
        self.cmax_progress.append((0,self.best_cmax))
        self.cmax_progress.append((curr_time-start_time, self.best_cmax))

        # counter for the number of steps without any improvement
        no_improve = 0

        # counter for the total number of steps
        step = 0
        
        # stopping condition
        stopping_reason = "max_steps"

        # start Genetic Algorithm loop
        while step < self.max_steps and no_improve < self.max_steps_no_improve and self.best_cmax > self.lower_bound:
            
            ## select chromosomes to undergo crossover

            #Sort population and compute weights for weighted probabilities
            N = len(self.population_scores)
            ranked = sorted(range(N), key=lambda i: self.population_scores[i])  # best first
            weights = [2.0 * (k+1) / (N * (N + 1)) for k in range(N)]  # k from 0..N-1 => (k+1)
            
            # randomly choose first parent chromosome
            parent1_index = random.choices(ranked, weights=weights, k=1)[0]
            parent1 = self.population[parent1_index]

            # choose second parent randomly among the remaining population, with uniform probability
            remaining = [i for i in range(N) if i != parent1_index]
            parent2_index = random.choice(remaining)
            parent2 = self.population[parent2_index]

            ## Crossover
            
            # produce children via crossover, choose one randomly
            child_a, child_b = self.apply_crossover(parent1, parent2)
            chosen_child = random.choice([child_a, child_b])

            ## Mutation


            # randomly decide whether to apply mutation
            if random.random() < self.mutation_probability:
                chosen_child = self.apply_mutation(chosen_child)
            
            # evaluate child
            sched, cmax = self.evaluate(chosen_child)
            
            child = chosen_child
            child_cmax = cmax
            child_schedule = sched
        

            # Replacement: replace uniformly one chromosome in lower half if child's makespan is new
            if child_cmax not in self.population_scores:

                sorted_indices = sorted(range(len(self.population_scores)), key=lambda i: self.population_scores[i])
                lower_half = sorted_indices[len(sorted_indices)//2:]
                victim_idx = random.choice(lower_half)
                self.population[victim_idx] = child
                self.population_scores[victim_idx] = child_cmax

            # Update best individual and makespan
            if child_cmax < self.best_cmax:

                curr_time = time.time()
                self.best_cmax = child_cmax
                self.cmax_progress.append((curr_time-start_time, self.best_cmax))
                self.best_schedule = child_schedule
                # we improved, so reset improvement counter
                no_improve = 0
            
            else:

                no_improve += 1

            # check stop reasons
            if self.best_cmax <= self.lower_bound:
                stopping_reason = "lower_bound"
                break

            if no_improve >= self.max_steps_no_improve:
                stopping_reason = "no_improvement"
                break

            # increase step counter
            step += 1

            if step >= self.max_steps:
                stopping_reason = "max_steps"
                break

            # additional stopping condition for large job-count   
            if self.num_jobs * >= 50:
                
                time_now = time.time()

                if time_now - start_time >= self.max_cpu_time:
                    stopping_reason = "max_cpu_time"
                    break

        # compute total running time
        elapsed_total = time.time() - start_time

        # save data csv
        result_csv = os.path.join(self.output_dir, "result.csv")
        with open(result_csv, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["best_cmax", "stopping_condition", "elapsed_seconds", "iterations"])
            writer.writerow([self.best_cmax, stopping_reason, round(elapsed_total, 3), step])

        # plot makespan evolution graph
        xs, ys = zip(*self.cmax_progress)

        plt.figure(figsize=(8, 5))
        plt.step(xs, ys, where="post", linewidth=2, label="Best makespan")
        plt.scatter(xs, ys, color="red")
    
        # vertical line at x = 1200
        plt.axvline(x=1200, color="black", linestyle="--", linewidth=1.5, label="Limit at 1200")
    
        plt.xlabel("cpu_time (in seconds)")
        plt.ylabel("best_cmax")
        plt.title("Evolution of makespan over cpu time")
        plt.legend()
    
        output_path = os.path.join(self.output_dir, "cmax_evolution.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        # save best schedule as a csv
        best_savepath = os.path.join(self.output_dir, "best_schedule.csv")
        with open(best_savepath, "w") as f:
            for key, tuple_list in self.best_schedule.items():
                row = [str(key)] + [f"{','.join(map(str, t))}" for t in tuple_list]
                f.write(",".join(row) + "\n")

        # Save gantt chart (machines numbered 1..M)
        if self.best_schedule is not None:
            gantt_path = os.path.join(self.output_dir, "gantt_chart.png")
            self.plot_gantt_chart_machines_on_y(save_path=gantt_path)

        return self.best_cmax, self.best_schedule, stopping_reason, elapsed_total

    
    # function to plot a gantt chart
    def plot_gantt_chart_machines_on_y(self, save_path):
    
        schedule = self.best_schedule
        num_machines = len(schedule)
        
        # Collect all jobs for coloring
        all_jobs = set()
        for ops in schedule.values():
            for job, _, _, _ in ops:
                all_jobs.add(job)
        num_jobs = len(all_jobs)
    
        colors = plt.get_cmap('tab20', num_jobs)
        fig, ax = plt.subplots(figsize=(12, 8))
    
        max_end_time = 0
        for m, ops in schedule.items():
            for job, op, start, end in ops:
                duration = end - start
                max_end_time = max(max_end_time, end)
                ax.barh(
                    m, duration, left=start,
                    color=colors(job % num_jobs), edgecolor='black'
                )
    
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f"Machine {m+1}" for m in range(num_machines)], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Execution Time", fontsize=12)
        ax.set_xlim(0, max_end_time)
    
        # legend (adaptive columns)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors(job), edgecolor='black')
                   for job in range(num_jobs)]
        labels = [f"Job {job+1}" for job in range(num_jobs)]
        ncols = max(1, num_jobs // 25)
        ax.legend(handles, labels, title="Jobs",
                  loc='upper left', bbox_to_anchor=(1, 1), ncol=ncols)
    
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

# -----------------------
# CLI wrapper
# -----------------------
def run_ga_on_instance(problem_path, cg_path, lower_bounds_path, pop_size, crossover, mutation, pmut,
                       max_steps, max_no_improve, output_dir, generate_gantt=False, run_id=1, verbose=True, init_method="SEMI", max_cpu_time=1200):
    
    # get processing times and machines for all operations
    times_array, operations_array = parse_file_to_arrays(problem_path)
    
    # load conflict graph adjacency matrix
    cg_matrix = np.genfromtxt(cg_path, delimiter=",", dtype=int)

    # Build machine_operation_order base from operations_array; machines are 0..M-1
    base_ops = {}
    for m in range(operations_array.shape[0]):
        row = list(operations_array[m])
        base_ops[m] = [(int(row[o]), int(o)) for o in range(len(row))]

    # initiate class
    ga = GeneticAlgorithm(times_array, operations_array, cg_matrix,
                          lower_bounds_file=lower_bounds_path,
                          population_size=pop_size,
                          max_steps=max_steps,
                          max_steps_no_improve=max_no_improve,
                          crossover_method=crossover,
                          mutation_method=mutation,
                          mutation_probability=pmut,
                          init_method=init_method,
                          output_base_dir=output_dir,
                          max_cpu_time=max_cpu_time)

    # create instance name
    instance_name = os.path.basename(os.path.dirname(cg_path))

    # launch computation and retrieve results
    best_cmax, best_schedule, stop_reason, elapsed = ga.run(output_base_dir=output_dir,
                                                           instance_name=instance_name,
                                                           run_id=run_id,
                                                           verbose=verbose)

    return best_cmax, best_schedule, stop_reason, elapsed


# -----------------------
# Command line entry
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_path", required=True)
    parser.add_argument("--cg_path", required=True)
    parser.add_argument("--lower_bounds_path", required=False, default=None)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--crossover", type=str, default="PMX")
    parser.add_argument("--mutation", type=str, default="SWAP")
    parser.add_argument("--pmut", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--max_no_improve", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results_ga")
    parser.add_argument("--generate_gantt", action="store_true")
    parser.add_argument("--run_id", type=int, default=1, help="Run id (1..5) for repeated runs")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--init_method", type=str, default="SEMI", choices=["RANDOM", "SEMI"])
    parser.add_argument("--max_cpu_time", type=int, default=1200)
    args = parser.parse_args()

    run_ga_on_instance(args.problem_path, args.cg_path, args.lower_bounds_path,
                       args.pop_size, args.crossover, args.mutation, args.pmut,
                       args.max_steps, args.max_no_improve, args.output_dir,
                       generate_gantt=args.generate_gantt, run_id=args.run_id, verbose=args.verbose, init_method=args.init_method, max_cpu_time=args.max_cpu_time)
