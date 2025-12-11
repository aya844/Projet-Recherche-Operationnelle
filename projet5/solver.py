from __future__ import annotations

import time
from typing import Dict, Iterable, List, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    GUROBI_AVAILABLE = False


def solve_mwds(
    nodes: Iterable[str],
    neighbors: Dict[str, Iterable[str]],
    weights: Dict[str, float],
    time_limit: float | None = 60.0,
) -> Tuple[List[str], float, float]:
    start = time.time()
    node_list = list(nodes)

    for n in node_list:
        if n not in weights:
            raise ValueError(f"Missing weight for node {n}")

    if GUROBI_AVAILABLE:
        try:
            model = gp.Model("mwds")
            model.setParam('OutputFlag', 0)
            if time_limit is not None:
                model.setParam('TimeLimit', float(time_limit))

            x = {n: model.addVar(vtype=GRB.BINARY, name=f"x_{n}") for n in node_list}
            model.update()

            for i in node_list:
                nbrs = list(neighbors.get(i, []))
                expr = x[i]
                for j in nbrs:
                    if j in x:
                        expr = expr + x[j]
                model.addConstr(expr >= 1, name=f"dom_{i}")

            model.setObjective(gp.quicksum(weights[i] * x[i] for i in node_list), GRB.MINIMIZE)

            model.optimize()

            selected = [i for i in node_list if x[i].X > 0.5]
            obj = model.ObjVal if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else float('nan')
            runtime = time.time() - start
            return selected, float(obj), runtime

        except Exception as e:
            print(f"Gurobi error: {e}. Falling back to greedy approximation.")

    uncovered = set(node_list)
    selected = []
    while uncovered:
        best = None
        best_score = float('inf')
        best_covers = set()
        for v in node_list:
            covers = {v} | set(neighbors.get(v, []))
            new = uncovered & covers
            if not new:
                continue
            score = weights[v] / len(new)
            if score < best_score:
                best_score = score
                best = v
                best_covers = new
        if best is None:
            break
        selected.append(best)
        uncovered -= best_covers

    obj = sum(weights[s] for s in selected)
    runtime = time.time() - start
    return selected, float(obj), runtime
