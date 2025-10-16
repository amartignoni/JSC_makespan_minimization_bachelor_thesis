import os
import pandas as pd

OUTPUT_DIR = "genetic_parameters_test"
INSTANCES = [
    "Tai02_problem_0.5_3", "Tai01_problem_0.5_1", "Tai05_problem_0.2_3",
    "Tai10_problem_0.5_3", "Tai07_problem_0.2_3", "Tai06_problem_0.8_3",
    "Tai09_problem_0.8_2", "Tai03_problem_0.2_1", "Tai04_problem_0.8_2",
    "Tai20_problem_0.5_3", "Tai15_problem_0.2_2", "Tai11_problem_0.2_1",
    "Tai17_problem_0.8_3", "Tai12_problem_0.8_2", "Tai14_problem_0.8_3",
    "Tai13_problem_0.2_1", "Tai16_problem_0.5_2", "Tai18_problem_0.5_1",
    "Tai29_problem_0.8_3", "Tai25_problem_0.2_2", "Tai26_problem_0.5_1",
    "Tai30_problem_0.5_2", "Tai22_problem_0.2_2", "Tai21_problem_0.8_1",
    "Tai27_problem_0.8_3", "Tai23_problem_0.2_2", "Tai24_problem_0.5_3",
    "Tai32_problem_0.5_1", "Tai33_problem_0.8_3", "Tai34_problem_0.8_3",
    "Tai40_problem_0.5_2", "Tai38_problem_0.2_1", "Tai36_problem_0.8_3",
    "Tai39_problem_0.2_2", "Tai35_problem_0.2_2", "Tai31_problem_0.5_3",
    "Tai48_problem_0.8_1", "Tai43_problem_0.2_1", "Tai45_problem_0.2_3",
    "Tai42_problem_0.5_1", "Tai47_problem_0.5_3", "Tai50_problem_0.8_2",
    "Tai49_problem_0.2_3", "Tai44_problem_0.5_1", "Tai46_problem_0.8_2",
    "Tai57_problem_0.8_3", "Tai58_problem_0.5_1", "Tai55_problem_0.2_3",
    "Tai59_problem_0.2_1", "Tai51_problem_0.8_3", "Tai56_problem_0.8_2",
    "Tai53_problem_0.5_3", "Tai54_problem_0.5_3", "Tai52_problem_0.2_3",
    "Tai64_problem_0.5_3", "Tai63_problem_0.5_1", "Tai66_problem_0.8_1",
    "Tai67_problem_0.2_3", "Tai62_problem_0.5_2", "Tai65_problem_0.2_1",
    "Tai61_problem_0.8_3", "Tai69_problem_0.2_1", "Tai68_problem_0.8_1",
    "Tai78_problem_0.2_1", "Tai79_problem_0.2_1", "Tai73_problem_0.8_1",
    "Tai80_problem_0.5_1", "Tai75_problem_0.2_2", "Tai76_problem_0.8_1",
    "Tai74_problem_0.5_3", "Tai71_problem_0.8_3", "Tai77_problem_0.5_1"
]

# Must match your bash TAGUCHI_ARRAY order
# L18 Taguchi array: Initialization, Population, Crossover, Mutation
TAGUCHI_ARRAY = [
    ["RANDOM", "100", "LOX", "SWAP"],
    ["RANDOM", "100", "OX1", "MOVE"],
    ["RANDOM", "100", "PMX", "SWAP"],
    ["RANDOM", "200", "LOX", "SWAP"],
    ["RANDOM", "200", "OX1", "MOVE"],
    ["RANDOM", "200", "PMX", "SWAP"],
    ["RANDOM", "300", "LOX", "MOVE"],
    ["RANDOM", "300", "OX1", "SWAP"],
    ["RANDOM", "300", "PMX", "SWAP"],
    ["SEMI", "100", "LOX", "MOVE"],
    ["SEMI", "100", "OX1", "SWAP"],
    ["SEMI", "100", "PMX", "SWAP"],
    ["SEMI", "200", "LOX", "SWAP"],
    ["SEMI", "200", "OX1", "MOVE"],
    ["SEMI", "200", "PMX", "SWAP"],
    ["SEMI", "300", "LOX", "SWAP"],
    ["SEMI", "300", "OX1", "MOVE"],
    ["SEMI", "300", "PMX", "SWAP"]
]


RUNS = range(1, 6)
SETS = range(1, 19)

all_results = []

# Initialize ranking counters (one dict per rank)
rank_counts = {set_id: {"1st": 0, "2nd": 0, "3rd": 0} for set_id in SETS}

for instance in INSTANCES:
    instance_results = []

    for set_id in SETS:
        best_cmax = float("inf")
        for run in RUNS:
            result_path = os.path.join(
                OUTPUT_DIR,
                f"set{set_id}",
                f"{instance}_run{run}",
                f"pop{TAGUCHI_ARRAY[set_id - 1][1]}_xover{TAGUCHI_ARRAY[set_id - 1][2]}_mut{TAGUCHI_ARRAY[set_id - 1][3]}_init{TAGUCHI_ARRAY[set_id - 1][0]}",
                "result.csv"
            )

            if not os.path.exists(result_path):
                continue

            df = pd.read_csv(result_path)
            cmax = df["best_cmax"].iloc[0]
            if cmax < best_cmax:
                best_cmax = cmax

        if best_cmax != float("inf"):
            instance_results.append({
                "Set": set_id,
                "best_cmax": best_cmax
            })

    if not instance_results:
        print(f"No results for {instance}")
        continue

    # Create DataFrame for this instance's results
    df_instance = pd.DataFrame(instance_results)
    df_sorted = df_instance.sort_values("best_cmax")

    # Extract top 3 (or fewer if less data)
    top_sets = df_sorted["Set"].tolist()

    if len(top_sets) >= 1:
        rank_counts[top_sets[0]]["1st"] += 1
    if len(top_sets) >= 2:
        rank_counts[top_sets[1]]["2nd"] += 1
    if len(top_sets) >= 3:
        rank_counts[top_sets[2]]["3rd"] += 1

# Convert rank counts + parameters into final DataFrame
rank_df = pd.DataFrame([
    {
        "Set": set_id,
        "Init": TAGUCHI_ARRAY[set_id - 1][0],
        "Population": TAGUCHI_ARRAY[set_id - 1][1],
        "Crossover": TAGUCHI_ARRAY[set_id - 1][2],
        "Mutation": TAGUCHI_ARRAY[set_id - 1][3],
        "1st": rank_counts[set_id]["1st"],
        "2nd": rank_counts[set_id]["2nd"],
        "3rd": rank_counts[set_id]["3rd"]
    }
    for set_id in SETS
])

# Optional: total top-3 appearances and sorting
rank_df["TotalTop3"] = rank_df["1st"] + rank_df["2nd"] + rank_df["3rd"]
rank_df = rank_df.sort_values(["1st", "2nd", "3rd"], ascending=[False, False, False])

# Table A — sort by 1st > 2nd > 3rd
rank_df_sorted_by_ranks = rank_df.sort_values(["1st", "2nd", "3rd"], ascending=[False, False, False])
print("=== Sorted by 1st > 2nd > 3rd ===")
print(rank_df_sorted_by_ranks)
rank_df_sorted_by_ranks.to_csv("parameter_set_rankings_by_ranks.csv", index=False)

# Table B — sort by TotalTop3 > 1st > 2nd > 3rd
rank_df_sorted_by_total = rank_df.sort_values(["TotalTop3", "1st", "2nd", "3rd"], ascending=[False, False, False, False])
print("\n=== Sorted by TotalTop3 > 1st > 2nd > 3rd ===")
print(rank_df_sorted_by_total)
rank_df_sorted_by_total.to_csv("parameter_set_rankings_by_total.csv", index=False)

# Compute average row index for each Crossover, Init, Population, and Mutation
crossover_ranks = {}
init_ranks = {}
pop_ranks = {}
mutation_ranks = {}  # ✅ New for Mutation

# Reset index to get row numbers
df_ranked = rank_df_sorted_by_ranks.reset_index(drop=True)

for idx, row in df_ranked.iterrows():
    rank = idx + 1  # 1-based row number
    crossover = row["Crossover"]
    init = row["Init"]
    pop = row["Population"]
    mutation = row["Mutation"]  # ✅ New

    # Crossover
    if crossover not in crossover_ranks:
        crossover_ranks[crossover] = []
    crossover_ranks[crossover].append(rank)

    # Init
    if init not in init_ranks:
        init_ranks[init] = []
    init_ranks[init].append(rank)

    # Population
    if pop not in pop_ranks:
        pop_ranks[pop] = []
    pop_ranks[pop].append(rank)

    # Mutation
    if mutation not in mutation_ranks:
        mutation_ranks[mutation] = []
    mutation_ranks[mutation].append(rank)

# Compute averages
def compute_average_rank(rank_dict):
    return {key: sum(indices) / len(indices) for key, indices in rank_dict.items()}

avg_crossover_ranks = compute_average_rank(crossover_ranks)
avg_init_ranks = compute_average_rank(init_ranks)
avg_pop_ranks = compute_average_rank(pop_ranks)
avg_mutation_ranks = compute_average_rank(mutation_ranks)  # ✅

# Print results
print("\n=== Average Rank by Crossover ===")
for crossover, avg_rank in avg_crossover_ranks.items():
    print(f"{crossover}: {avg_rank:.2f}")

print("\n=== Average Rank by Init ===")
for init, avg_rank in avg_init_ranks.items():
    print(f"{init}: {avg_rank:.2f}")

print("\n=== Average Rank by Population ===")
for pop, avg_rank in avg_pop_ranks.items():
    print(f"{pop}: {avg_rank:.2f}")

print("\n=== Average Rank by Mutation ===")  # ✅
for mutation, avg_rank in avg_mutation_ranks.items():
    print(f"{mutation}: {avg_rank:.2f}")

