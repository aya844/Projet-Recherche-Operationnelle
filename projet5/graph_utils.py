from __future__ import annotations

import csv
from typing import Dict, Iterable, Tuple

import networkx as nx


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_graph_from_edge_list(path: str) -> Tuple[nx.Graph, Dict[str, float]]:
    G = nx.Graph()
    node_weights: Dict[str, float] = {}
    path_lower = path.lower()

    if path_lower.endswith('.csv'):
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                row = [c.strip() for c in row]
                if len(row) == 1:
                    continue
                if len(row) == 2:
                    a, b = row[0], row[1]
                    if _is_number(b):
                        node_weights[a] = float(b)
                        continue
                    else:
                        G.add_edge(a, b)
                        continue
                u, v = row[0], row[1]
                G.add_edge(u, v)
                try:
                    w = float(row[2])
                    G[u][v]["weight"] = w
                except Exception:
                    pass
    else:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')] if ',' in line else line.split()
                if len(parts) == 1:
                    continue
                if len(parts) == 2:
                    a, b = parts[0], parts[1]
                    if _is_number(b):
                        node_weights[a] = float(b)
                        continue
                    else:
                        G.add_edge(a, b)
                        continue
                u, v = parts[0], parts[1]
                G.add_edge(u, v)
                try:
                    w = float(parts[2])
                    G[u][v]["weight"] = w
                except Exception:
                    pass

    for n in G.nodes():
        if str(n) not in node_weights:
            node_weights[str(n)] = 1.0

    return G, node_weights


def save_weights_to_file(path: str, G: nx.Graph, weights: Dict[str, float]) -> str:
    if path.lower().endswith('.csv'):
        out_path = path + '.node_weights.csv'
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for n, w in weights.items():
                writer.writerow([n, float(w)])
    else:
        out_path = path + '.node_weights.txt'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('# Node weights (format: node weight)\n')
            for n, w in weights.items():
                f.write(f"{n} {float(w)}\n")
    return out_path


def neighbors_map(G: nx.Graph) -> Dict[str, Iterable[str]]:
    return {str(n): [str(v) for v in G.neighbors(n)] for n in G.nodes()}


def draw_graph(ax, G: nx.Graph, pos=None, selected: Iterable[str] | None = None) -> Dict[str, Tuple[float, float]]:
    if pos is None:
        pos = nx.spring_layout(G)

    sel_set = set(selected) if selected is not None else set()

    node_colors = ['#ADD8E6' if str(n) not in sel_set else '#FF5555' for n in G.nodes()]

    ax.clear()
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=300)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    ax.set_axis_off()
    return pos
