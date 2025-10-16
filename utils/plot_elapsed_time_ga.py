import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------
# Configuration
# -------------------------
class_order = ["15x15", "20x15", "20x20", "30x15", "30x20", "50x15", "50x20", "100x20"]
class_ops = {
    "15x15": 15 * 15,
    "20x15": 20 * 15,
    "20x20": 20 * 20,
    "30x15": 30 * 15,
    "30x20": 30 * 20,
    "50x15": 50 * 15,
    "50x20": 50 * 20,
    "100x20": 100 * 20,
}

def get_problem_class(xx):
    if 1 <= xx <= 10: return "15x15"
    if 11 <= xx <= 20: return "20x15"
    if 21 <= xx <= 30: return "20x20"
    if 31 <= xx <= 40: return "30x15"
    if 41 <= xx <= 50: return "30x20"
    if 51 <= xx <= 60: return "50x15"
    if 61 <= xx <= 70: return "50x20"
    if 71 <= xx <= 80: return "100x20"
    return None

poly_degree = 2  # degré polynôme

# -------------------------
# Collecte des données
# -------------------------
records = []
root = "genetic_results"
for xx in range(1, 81):             
    for P in [0.2, 0.5, 0.8]:       
        for S in range(1, 4):       
            for R in range(1, 6):   
                folder = f"{root}/Tai{xx:02d}_problem_{P}_{S}_run{R}/pop200_xoverLOX_mutMOVE_initSEMI"
                filepath = os.path.join(folder, "result.csv")
                if not os.path.isfile(filepath):
                    continue
                try:
                    df = pd.read_csv(filepath)
                    elapsed = float(df["elapsed_seconds"].iloc[0])
                    cls = get_problem_class(xx)
                    ops = class_ops[cls]
                    records.append({
                        "problem_id": xx,
                        "class": cls,
                        "operations": ops,
                        "probability": P,
                        "scenario": S,
                        "run": R,
                        "elapsed_seconds": elapsed,
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {filepath}: {e}")

print(f"Collected {len(records)} records")
if not records:
    raise RuntimeError("No data collected! Check paths and files.")

data = pd.DataFrame(records)
data["class"] = pd.Categorical(data["class"], categories=class_order, ordered=True)

# -------------------------
# Helper: format polynomial
# -------------------------
def poly_to_string(coeffs):
    degree = len(coeffs) - 1
    s = ""
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        mag = abs(c)
        mag_str = f"{mag:.6g}"
        if s == "":
            if c < 0:
                term = f"-{mag_str}"
            else:
                term = f"{mag_str}"
        else:
            sign = "+" if c > 0 else "-"
            term = f" {sign} {mag_str}"
        if power == 0:
            s += term
        elif power == 1:
            s += term + "*x"
        else:
            s += term + f"*x^{power}"
    if s == "":
        s = "0"
    return s

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

colors = {0.2: plt.get_cmap("tab10")(0), 0.5: plt.get_cmap("tab10")(2), 0.8: plt.get_cmap("tab10")(4)}
ops_list = [class_ops[c] for c in class_order]
min_ops, max_ops = min(ops_list) - 50, max(ops_list) + 50
x_line = np.linspace(min_ops, max_ops, 400)
equations = {}

# décalage horizontal par probabilité
jitter_offsets = {0.2: -0.2, 0.5: 0.0, 0.8: 0.2}

for P in [0.2, 0.5, 0.8]:
    grp = data[data["probability"] == P]
    x = grp["operations"].values.astype(float)
    y = grp["elapsed_seconds"].values.astype(float)

    # jitter horizontal dépend de la probabilité
    jitter_strength = 0.3 * np.mean(np.diff(sorted(set(ops_list))))
    x_jittered = x + jitter_strength * jitter_offsets[P]

    ax.scatter(x_jittered, y, color=colors[P], alpha=0.75, s=40)

    # regression polynomial
    if len(grp) >= poly_degree + 1:
        coeffs = np.polyfit(x, y, poly_degree)
        poly = np.poly1d(coeffs)
        y_line = poly(x_line)
        ax.plot(x_line, y_line, linestyle="--", linewidth=2, color=colors[P])
        eq_str = poly_to_string(coeffs)
        equations[P] = eq_str

# -------------------------
# Mise en forme
# -------------------------
ax.set_xticks(ops_list)
ax.set_xticklabels(class_order, rotation=30, ha="right")
ax.set_xlim(min_ops, max_ops)
ax.set_xlabel("Problem class")
ax.set_ylabel("CPU time")
ax.set_title(f"CPU time for each class, for each conflict density P")
ax.grid(axis="y", linestyle=":", alpha=0.6)

# -------------------------
# Légende avec équations
# -------------------------
legend_elements = []
for P in [0.2, 0.5, 0.8]:
    label = f"P={P}\n{equations.get(P,'')}"
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=colors[P], markersize=8))
ax.legend(handles=legend_elements, loc="upper left", fontsize=9, frameon=True)

plt.tight_layout()

# Sauvegarde
out_png = os.path.join(root, "cpu_time_evolution.png")
os.makedirs(root, exist_ok=True)
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print("Saved figure to:", out_png)

# console
print("\nRegression equations:")
for P, eq in equations.items():
    print(f"P={P}: {eq}")

plt.show()
