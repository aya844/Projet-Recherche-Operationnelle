# solveur_PLM_test.py
import csv
from gurobipy import Model, GRB, quicksum
from collections import defaultdict, deque

# -------------------------
# 1) Lire CSV
# -------------------------
edges = []
with open("map2.csv", newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        edges.append((row['start'], row['end'], float(row['capacity']), float(row['cost'])))

demands = []
with open("demand2.csv", newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        demands.append((row['start'], row['end'], float(row['traffic'])))

# -------------------------
# 2) Construire graphe pour BFS
# -------------------------
graph = defaultdict(list)
for i, j, cap, cost in edges:
    graph[i].append(j)

def all_paths(graph, start, end, maxlen=5):
    paths = []
    queue = deque()
    queue.append((start, [start]))
    while queue:
        node, path = queue.popleft()
        if node == end:
            paths.append(path)
        elif len(path) < maxlen:
            for neigh in graph[node]:
                if neigh not in path:
                    queue.append((neigh, path + [neigh]))
    return paths

# -------------------------
# 3) Modèle PLM / MILP
# -------------------------
def solve_network():
    model = Model("NetworkCapacity_PLM")

    u_exist = {(i, j): cap for i, j, cap, cost in edges}
    c = {(i, j): cost for i, j, cap, cost in edges}

    # variables capacité à ajouter
    x = model.addVars(u_exist.keys(), lb=0, name="x")

    # variables flux continues et binaires par chemin
    f_path = {}
    y_path = {}
    path_edges = {}

    for s, t, d in demands:
        paths = all_paths(graph, s, t, maxlen=5)
        f_path[(s, t)] = {}
        y_path[(s, t)] = {}
        for idx, path in enumerate(paths):
            edges_in_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
            f_var = model.addVar(lb=0, name=f"f_{s}_{t}_{idx}")
            y_var = model.addVar(vtype=GRB.BINARY, name=f"y_{s}_{t}_{idx}")
            f_path[(s, t)][idx] = f_var
            y_path[(s, t)][idx] = y_var
            path_edges[(s, t, idx)] = edges_in_path

    model.update()

    # -------------------------
    # 4) Contraintes
    # -------------------------

    # a) demande totale
    for s, t, d in demands:
        model.addConstr(quicksum(f_path[(s, t)][idx] for idx in f_path[(s, t)]) == d,
                        name=f"demand_{s}_{t}")

    # b) capacité des liens
    for i, j in u_exist.keys():
        model.addConstr(
            quicksum(f_path[(s, t)][idx]
                     for (s, t), paths in f_path.items()
                     for idx, var in paths.items()
                     if (i, j) in path_edges[(s, t, idx)]
                     ) <= u_exist[i, j] + x[i, j],
            name=f"cap_{i}_{j}"
        )

    # c) relier flux et binaire : flux >0 seulement si chemin utilisé
    M = max(d[2] for d in demands)  # borne sur flux
    for (s, t), paths in f_path.items():
        for idx, f_var in paths.items():
            model.addConstr(f_var <= M * y_path[(s, t)][idx])

    # -------------------------
    # 5) Objectif
    # -------------------------
    model.setObjective(quicksum(c[i, j] * x[i, j] for i, j in u_exist.keys()), GRB.MINIMIZE)

    # -------------------------
    # 6) Résolution
    # -------------------------
    model.optimize()

    # -------------------------
    # 7) Résultats
    # -------------------------
    if model.status == GRB.OPTIMAL:
        print("\n=== Capacités à ajouter ===")
        for i, j in u_exist.keys():
            if x[i, j].X > 1e-6:
                print(f"Lien {i}->{j}: ajouter {x[i, j].X:.2f} unités")

        print("\n=== Flux par chemin ===")
        for (s, t), paths in f_path.items():
            for idx, var in paths.items():
                if var.X > 1e-6:
                    print(f"Flux {s}->{t} sur chemin {idx} (edges {path_edges[(s,t,idx)]}): {var.X:.2f}")

        print(f"\nCoût total = {model.ObjVal:.2f}")
    else:
        print("Aucune solution optimale trouvée.")

# -------------------------
# Exécution
# -------------------------
if __name__ == "__main__":
    solve_network()
