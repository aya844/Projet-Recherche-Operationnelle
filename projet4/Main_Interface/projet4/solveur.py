# solveur.py
from gurobipy import Model, GRB, quicksum
from collections import defaultdict, deque

def all_paths(graph, start, end, maxlen=5):
    paths = []
    queue = deque()
    queue.append((start, [start]))
    while queue:
        node, path_ = queue.popleft()
        if node == end:
            paths.append(path_)
        elif len(path_) < maxlen:
            for neigh in graph[node]:
                if neigh not in path_:
                    queue.append((neigh, path_ + [neigh]))
    return paths

def solve_network(edges, demands):
    """
    edges: list of tuples (u, v, capacity, cost)
    demands: list of tuples (s, t, traffic)
    Returns: (added_capacities_dict, total_cost)
             added_capacities_dict: {(u, v): added_amount}
    """
    # Build graph
    graph = defaultdict(list)
    for i, j, cap, cost in edges:
        graph[i].append(j)

    # Model
    model = Model("PL")
    model.setParam('OutputFlag', 0) # Mute output
    
    u_exist = {(i,j): cap for i,j,cap,cost in edges}
    c = {(i,j): cost for i,j,cap,cost in edges}
    
    x = model.addVars(u_exist.keys(), lb=0, name="x")

    f_path = {}
    path_edges = {}
    
    for s, t, d in demands:
        paths = all_paths(graph, s, t, maxlen=5)
        f_path[(s,t)] = {}
        for idx, path_ in enumerate(paths):
            edges_in_path = [(path_[i], path_[i+1]) for i in range(len(path_)-1)]
            var = model.addVar(lb=0, name=f"f_{s}_{t}_{idx}")
            f_path[(s,t)][idx] = var
            path_edges[(s,t,idx)] = edges_in_path
            
    model.update()

    # Demand Constraints
    for s, t, d in demands:
        if (s,t) in f_path:
            model.addConstr(quicksum(f_path[(s,t)][idx] for idx in f_path[(s,t)]) == d)

    # Capacity Constraints
    for i, j in u_exist.keys():
        flow_sum = quicksum(f_path[(s,t)][idx] 
                            for (s,t), paths in f_path.items() 
                            for idx, var in paths.items() 
                            if (i,j) in path_edges.get((s,t,idx), []))
        model.addConstr(flow_sum <= u_exist[i,j] + x[i,j])

    # Objective
    model.setObjective(quicksum(c[i,j]*x[i,j] for i,j in u_exist.keys()), GRB.MINIMIZE)
    model.optimize()

    added_caps = {}
    if model.Status == GRB.OPTIMAL:
        for i, j in u_exist.keys():
            if x[i,j].X > 1e-6:
                added_caps[(i,j)] = x[i,j].X
        return added_caps, model.ObjVal
    else:
        return {}, float('inf')

if __name__ == "__main__":
    # Test with files logic if needed, or just dummy test
    pass
