import numpy as np
import sys
import random
import gurobipy as gp
from gurobipy import GRB
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import networkx as nx
from itertools import product
from tqdm import tqdm
import copy
import csv
np.set_printoptions(threshold=sys.maxsize)
sys.path.append('./utils')
import read_result_values
import pandas as pd


def load_upper_bound(metadata_file):
    upper_bound = None
    with open(metadata_file, 'r') as f:
        next(f)  # skip header line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                key, value = parts
                if key.strip() == 'upper_bound':
                    upper_bound = float(value.strip())
                    break

    if upper_bound is None:
        raise KeyError("Key 'upper_bound' not found in metadata.csv")

    return upper_bound

def job_lower_bound(times_array):
    return times_array.sum(axis=1).max()


def machine_lower_bound(operations_array, times_array):
    num_machines = np.max(operations_array)
    machine_times = {machine: 0 for machine in range(1, num_machines + 1)}
    for i in range(operations_array.shape[0]):
        for j in range(operations_array.shape[1]):
            machine = operations_array[i][j]
            time = times_array[i][j]
            machine_times[machine] += time
    return max(machine_times.values())


def create_job_graph_complement(cg_matrix, times_array):
    cg_graph = nx.Graph()
    job_weights = times_array.sum(axis=1)

    for job in range(len(job_weights)):
        cg_graph.add_node(job, weight=job_weights[job])

    num_jobs = cg_matrix.shape[0]
    for i in range(num_jobs):
        for j in range(i + 1, num_jobs):
            if cg_matrix[i, j] == 0 and i != j:
                cg_graph.add_edge(i, j)

    return cg_graph


def gwmin(graph):
    total_weight = 0
    graph = copy.deepcopy(graph)

    while graph.number_of_nodes() > 0:
        scores = {
            node: int(graph.nodes[node]["weight"]) / (int(graph.degree[node]) + 1)
            for node in graph.nodes
        }
        selected_node = max(scores, key=scores.get)
        total_weight += graph.nodes[selected_node]["weight"]
        neighbors = list(graph.neighbors(selected_node))
        graph.remove_node(selected_node)
        graph.remove_nodes_from(neighbors)

    return total_weight


def gwmin2(graph):
    total_weight = 0
    graph = copy.deepcopy(graph)

    while graph.number_of_nodes() > 0:
        scores = {}
        for node in graph.nodes():
            neighborhood = list(graph.neighbors(node)) + [node]
            neighborhood_weight_sum = sum(int(graph.nodes[neighbor]["weight"]) for neighbor in neighborhood)
            scores[node] = int(graph.nodes[node]["weight"]) / neighborhood_weight_sum

        selected_node = max(scores, key=scores.get)
        total_weight += graph.nodes[selected_node]["weight"]
        neighbors = list(graph.neighbors(selected_node))
        graph.remove_node(selected_node)
        graph.remove_nodes_from(neighbors)

    return total_weight


def load_cg_matrix(file_path):
    return np.loadtxt(file_path, delimiter=',')


### MAIN

cg_matrix = load_cg_matrix(sys.argv[1])
operations_array, times_array = read_result_values.parse_file_to_arrays(sys.argv[2])

metadata_file = os.path.join(os.path.dirname(sys.argv[1]), 'metadata.csv')
print(metadata_file)
upper_bound = load_upper_bound(metadata_file)
print(upper_bound)

complement_graph = create_job_graph_complement(cg_matrix, times_array)

jlb = job_lower_bound(times_array)
mlb = machine_lower_bound(operations_array, times_array)
gwmin_lb = gwmin(complement_graph)
gwmin2_lb = gwmin2(complement_graph)

results = {
    "jlb": jlb,
    "mlb": mlb,
    "gwmin_lb": gwmin_lb,
    "gwmin2_lb": gwmin2_lb,
    "upper_bound": upper_bound
}

results_file = sys.argv[3]
with open(results_file, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "Value"])
    for key, value in results.items():
        writer.writerow([key, value])

print(f"Results saved in CSV file: {results_file}")
