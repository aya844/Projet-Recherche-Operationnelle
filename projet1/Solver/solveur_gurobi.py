"""
solver/run_solver.py

Script complet pour :
 - lire les CSV (data/nodes.csv, data/arcs.csv, data/params.csv)
 - construire et résoudre le MILP (Gurobi), support multi-type d'organes et véhicules entiers
 - exporter les résultats (data/results.csv)
 - générer une visualisation réseau (PNG)
 - générer un rapport PDF (data/report.pdf)
 - (optionnel) lancer une fenêtre PyQt6 montrant un tableau interactif et le graphique

Usage:
    python solver/run_solver.py            # exécute la résolution et génère fichiers
    python solver/run_solver.py --show     # ouvre une fenêtre PyQt6 affichant résultats

Dépendances:
    - gurobipy (Gurobi)
    - pandas
    - networkx
    - matplotlib
    - PyQt6 (pour l'affichage interactif optionnel)

Assure-toi d'avoir les fichiers :
    data/nodes.csv  (colonnes: id,type,name,supply)
    data/arcs.csv   (colonnes: origin,dest,cost,time,capacity,cap_cost or cost_<org>/time_<org>)
    data/params.csv (optionnel: alpha,beta)

Le script est conçu pour être autonome et bien commenté pour que tu puisses l'adapter
facilement à ton IHM PyQt principal.
"""

import os
import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB, quicksum
except Exception:
    gp = None
    GRB = None
    quicksum = None
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # backend non-GUI par défaut
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QHBoxLayout, QMessageBox
    )
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



# -----------------------------
# Utilities: file handling
# -----------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

NODES_CSV = DATA_DIR / "nodes.csv"
ARCS_CSV  = DATA_DIR / "arcs.csv"
PARAMS_CSV = DATA_DIR / "params.csv"
RESULTS_CSV = DATA_DIR / "results.csv"
PLOT_PNG = DATA_DIR / "network_plot.png"
REPORT_PDF = DATA_DIR / "report.pdf"


# -----------------------------
# Read input data
# -----------------------------
def read_inputs():
    # nodes
    if not NODES_CSV.exists():
        raise FileNotFoundError(f"Fichier manquant : {NODES_CSV}")
    df_nodes = pd.read_csv(NODES_CSV)
    # sanitize column names and id values
    df_nodes.columns = df_nodes.columns.str.strip()
    if 'id' in df_nodes.columns:
        df_nodes['id'] = df_nodes['id'].astype(str).str.strip()
    # expected columns: id,type,name and at least one supply column ('supply' or 'supply_<org>')
    required_nodes_cols = {"id", "type", "name"}
    if not required_nodes_cols.issubset(set(df_nodes.columns)):
        raise ValueError(f"nodes.csv doit contenir les colonnes {required_nodes_cols}")
    # ensure we have supply or supply_<org> columns
    supply_cols = [c for c in df_nodes.columns if c == 'supply' or c.startswith('supply_')]
    if not supply_cols:
        raise ValueError("nodes.csv doit contenir la colonne 'supply' ou au moins une colonne 'supply_<organ>'")

    # arcs
    if not ARCS_CSV.exists():
        raise FileNotFoundError(f"Fichier manquant : {ARCS_CSV}")
    df_arcs = pd.read_csv(ARCS_CSV)
    # sanitize columns and origin/dest
    df_arcs.columns = df_arcs.columns.str.strip()
    if 'origin' in df_arcs.columns:
        df_arcs['origin'] = df_arcs['origin'].astype(str).str.strip()
    if 'dest' in df_arcs.columns:
        df_arcs['dest'] = df_arcs['dest'].astype(str).str.strip()
    # expected columns: origin,dest,capacity,cap_cost and either cost/time or cost_<org>/time_<org>
    required_arcs_cols = {"origin", "dest", "capacity", "cap_cost"}
    if not required_arcs_cols.issubset(set(df_arcs.columns)):
        raise ValueError(f"arcs.csv doit contenir au minimum les colonnes {required_arcs_cols}")
    # ensure at least one cost/time column exists
    cost_cols = [c for c in df_arcs.columns if c == 'cost' or c.startswith('cost_')]
    time_cols = [c for c in df_arcs.columns if c == 'time' or c.startswith('time_')]
    if not cost_cols:
        raise ValueError("arcs.csv doit contenir 'cost' ou au moins une colonne 'cost_<organ>'")
    if not time_cols:
        raise ValueError("arcs.csv doit contenir 'time' ou au moins une colonne 'time_<organ>'")

    # params (optional)
    params = {"alpha": 1.0, "beta": 10.0, "vehicle_capacity": 1.0, "organs": ""}
    if PARAMS_CSV.exists():
        # Read as two-column CSV with key,value. Keep values as strings unless convertible to float.
        try:
            dfp = pd.read_csv(PARAMS_CSV, header=None)
            for _, row in dfp.iterrows():
                key = str(row[0]).strip()
                val = str(row[1]).strip()
                # try to convert numeric
                try:
                    params[key] = float(val)
                except Exception:
                    params[key] = val
        except Exception:
            # fallback: try first row as header + value
            try:
                dpf = pd.read_csv(PARAMS_CSV)
                for k in ["alpha","beta","vehicle_capacity","organs"]:
                    if k in dpf.columns:
                        v = dpf[k].iloc[0]
                        try:
                            params[k] = float(v)
                        except Exception:
                            params[k] = str(v)
            except Exception:
                pass

    return df_nodes, df_arcs, params


# -----------------------------
# Build and solve model
# -----------------------------

def solve_model(df_nodes, df_arcs, params, write_lp=False):
    # Preprocess
    nodes = df_nodes['id'].astype(str).tolist()
    # Determine organ types K
    organs = []
    # 1) Look for columns 'supply_<org>' in nodes.csv
    for col in df_nodes.columns:
        if col.startswith('supply_'):
            organs.append(col.replace('supply_', ''))
    # 2) Fallback: see params 'organs' comma-separated
    if not organs:
        organs_str = str(params.get('organs', '')).strip()
        if organs_str:
            organs = [o.strip() for o in organs_str.split(',') if o.strip()]
    # 3) Default: single type 'generic'
    if not organs:
        organs = ['generic']

    # Build b_n^k supply/demand dictionary
    b = {}
    for n in nodes:
        b[n] = {}
        for k_ in organs:
            col = f'supply_{k_}'
            if col in df_nodes.columns:
                b[n][k_] = float(df_nodes.loc[df_nodes['id'].astype(str)==n, col].iloc[0])
            else:
                # fallback: if legacy 'supply' exists, use it for generic or distribute proportionally (here use legacy for generic)
                if 'supply' in df_nodes.columns and k_ == 'generic':
                    b[n][k_] = float(df_nodes.loc[df_nodes['id'].astype(str)==n, 'supply'].iloc[0])
                else:
                    b[n][k_] = 0.0

    arcs = [(str(row.origin), str(row.dest)) for _, row in df_arcs.iterrows()]

    u = { (str(row.origin), str(row.dest)): float(row.capacity) for _, row in df_arcs.iterrows() }
    # cost/time may be generic or per organ (cost_<org>, time_<org>)
    c = {}
    tt = {}
    for _, row in df_arcs.iterrows():
        i = str(row.origin); j = str(row.dest)
        for k_ in organs:
            cost_col = f'cost_{k_}'
            time_col = f'time_{k_}'
            if cost_col in df_arcs.columns:
                c[(i,j,k_)] = float(row[cost_col])
            else:
                c[(i,j,k_)] = float(row.cost)
            if time_col in df_arcs.columns:
                tt[(i,j,k_)] = float(row[time_col])
            else:
                tt[(i,j,k_)] = float(row.time)
    # vehicle capacity and vehicle cost (per vehicle) per arc
    V = float(params.get('vehicle_capacity', 1.0))
    cost_vehicle = { (str(row.origin), str(row.dest)): float(row.get('cap_cost', 0.0)) for _, row in df_arcs.iterrows() }

    alpha = float(params.get('alpha', 1.0))
    beta  = float(params.get('beta', 10.0))

    if gp is None:
        raise ImportError("gurobipy (Gurobi) n'est pas installé. Installez Gurobi ou définissez un solveur MILP alternatif.")
    model = gp.Model("Organs_Transport_MILP")
    model.setParam('OutputFlag', 1)  # 0 to silence

    # helper: sanitize names
    def safe_name(s):
        return ''.join(c if c.isalnum() else '_' for c in str(s))
    # Variables: x_{ij}^k and integer n_{ij}
    x = { (i,j,k_): model.addVar(lb=0.0, name=f"x_{safe_name(i)}_{safe_name(j)}_{safe_name(k_)}") for (i,j) in arcs for k_ in organs }
    nVeh = { (i,j): model.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"n_{safe_name(i)}_{safe_name(j)}") for (i,j) in arcs }
    # derived y = V * n (not created as var to keep LP small)
    model.update()

    # --- Diagnostics before solving ---
    # 1) Check supply/demand balance per organ
    for k_ in organs:
        total_supply = sum(b[n].get(k_, 0.0) for n in nodes)
        if abs(total_supply) > 1e-9:
            print(f"Warning: total net supply for organ '{k_}' is {total_supply} (not zero) - model likely infeasible")
    # 2) Check node connectivity for non-zero supplies
    for n in nodes:
        outgoing = [(i,j) for (i,j) in arcs if i == n]
        incoming = [(i,j) for (i,j) in arcs if j == n]
        # check supply/demand per organ
        for k_ in organs:
            supply_val = b.get(n, {}).get(k_, 0.0)
            if supply_val > 0 and len(outgoing) == 0:
                print(f"Warning: node {n} has positive supply {supply_val} for organ {k_} but no outgoing arcs")
            if supply_val < 0 and len(incoming) == 0:
                print(f"Warning: node {n} has demand {supply_val} for organ {k_} but no incoming arcs")

    # write a simple pre-solve summary to data for debugging
    try:
        with open(DATA_DIR / 'pre_solve_checks.txt', 'w', encoding='utf-8') as pf:
            pf.write('Pre-solve diagnostics:\n')
            for k_ in organs:
                total_supply = sum(b[n].get(k_, 0.0) for n in nodes)
                pf.write(f"Total supply for {k_}: {total_supply}\n")
            pf.write('\nNode connectivity:\n')
            for n in nodes:
                outs = [a for a in arcs if a[0] == n]
                ins = [a for a in arcs if a[1] == n]
                pf.write(f"Node {n}: outgoing {len(outs)} arcs, incoming {len(ins)} arcs\n")
    except Exception:
        pass

    # Constraints : conservation des flux par type
    for n in nodes:
        for k_ in organs:
            out_vars = [x[(i,j,k_)] for (i,j) in arcs if i == n]
            in_vars  = [x[(i,j,k_)] for (i,j) in arcs if j == n]
            model.addConstr(gp.quicksum(out_vars) - gp.quicksum(in_vars) == float(b.get(n, {}).get(k_, 0.0)), name=f"flow_{n}_{k_}")

    # Capacity constraints
    for (i,j) in arcs:
        model.addConstr(gp.quicksum(x[(i,j,k_)] for k_ in organs) <= u[(i,j)] + V * nVeh[(i,j)], name=f"cap_{i}_{j}")

    # Objective
    total_transport_cost = gp.quicksum(c[(i,j,k_)] * x[(i,j,k_)] for (i,j) in arcs for k_ in organs)
    total_time_component  = gp.quicksum(tt[(i,j,k_)]  * x[(i,j,k_)] for (i,j) in arcs for k_ in organs)
    total_invest_cost = gp.quicksum(cost_vehicle[(i,j)] * nVeh[(i,j)] for (i,j) in arcs)

    obj = alpha * total_transport_cost + beta * total_time_component + total_invest_cost
    model.setObjective(obj, GRB.MINIMIZE)

    if write_lp:
        model.write(str(DATA_DIR / "model.lp"))

    model.optimize()

    # Check status
    if model.status != GRB.OPTIMAL:
        print("Attention : solution optimale non trouvée. Statut=", model.status)
    if model.status == GRB.INFEASIBLE:
        print('Model is INFEASIBLE. Attempting to compute IIS (Irreducible Infeasible Subsystem) ...')
        try:
            model.computeIIS()
            iis_file = DATA_DIR / 'infeasible_iis.ilp'
            model.write(str(iis_file))
            print('IIS written to', iis_file)
            # Also write a human readable summary of IIS constraints & vars
            try:
                with open(DATA_DIR / 'infeasible_iis.txt', 'w', encoding='utf-8') as f:
                    f.write('Variables in IIS (nonzero indicating membership):\n')
                    for v in model.getVars():
                        # var.IISLB or var.IISUB indicates membership in IIS (lower bound / upper bound)
                        iislb = getattr(v, 'IISLB', 0)
                        iisub = getattr(v, 'IISUB', 0)
                        if iislb or iisub:
                            f.write(f"VAR {v.VarName} IISLB={iislb} IISUB={iisub}\n")
                    f.write('\nConstraints in IIS:\n')
                    for c in model.getConstrs():
                        if c.IISConstr:
                            f.write(f"CONSTR {c.ConstrName}\n")
                print('IIS summary written to data/infeasible_iis.txt')
            except Exception as e:
                print('Error writing IIS summary:', e)
        except Exception as e:
            print('Error computing IIS or writing ILP:', e)

    # Write a status file reflecting Gurobi model status
    try:
        status_text = 'NO_SOLUTION'
        if model.SolCount > 0:
            # we have a solution; distinguish optimal vs feasible
            if model.status == GRB.OPTIMAL:
                status_text = 'OPTIMAL'
            else:
                status_text = 'FEASIBLE'
        else:
            if model.status == GRB.INFEASIBLE:
                status_text = 'INFEASIBLE'
            elif model.status == GRB.UNBOUNDED:
                status_text = 'UNBOUNDED'
            else:
                # other codes
                status_text = f'STATUS_{int(model.status)}'
        try:
            with open(DATA_DIR / 'solver_status.txt', 'w', encoding='utf-8') as sf:
                sf.write(status_text)
        except Exception:
            pass
    except Exception:
        pass

    # Collect results only if variables are available (optimal or feasible solution exists)
    rows = []
    if model.SolCount > 0:
        # Write one row per arc+organ, and also collect n_ij once per arc
        for (i,j) in arcs:
            nval = nVeh[(i,j)].X if hasattr(nVeh[(i,j)], 'X') and nVeh[(i,j)].X is not None else 0.0
            yv = V * float(nval)
            for k_ in organs:
                xv = x[(i,j,k_)].X if hasattr(x[(i,j,k_)], 'X') and x[(i,j,k_)].X is not None else 0.0
                rows.append({
                    'origin': i, 'dest': j,
                    'organ': k_, 'x': float(xv), 'u': float(u[(i,j)]), 'y': float(yv), 'n': float(nval),
                    'cost': float(c[(i,j,k_)]), 'cap_cost': float(cost_vehicle[(i,j)]), 'time': float(tt[(i,j,k_)])
                })
            # also add an aggregated row (organ=ALL) with sum of flows
            sumx = sum([float(x[(i,j,k_)].X) if hasattr(x[(i,j,k_)], 'X') and x[(i,j,k_)].X is not None else 0.0 for k_ in organs])
            rows.append({
                'origin': i, 'dest': j,
                'organ': 'ALL', 'x': float(sumx), 'u': float(u[(i,j)]), 'y': float(yv), 'n': float(nval),
                'cost': float(cost_vehicle[(i,j)]), 'cap_cost': float(cost_vehicle[(i,j)]), 'time': None
            })
    else:
        # No solution available; rows remains empty
        rows = []

    # Ensure df_res has the expected columns even if empty, so downstream tools don't crash
    cols = ['origin','dest','organ','x','u','y','n','cost','cap_cost','time']
    if len(rows) == 0:
        df_res = pd.DataFrame(columns=cols)
    else:
        df_res = pd.DataFrame(rows)
        # if columns are not present ensure they exist in the correct order
        for c in cols:
            if c not in df_res.columns:
                df_res[c] = None
        df_res = df_res[cols]

    # Summary
    total_cost = None
    transport_cost_val = None
    invest_cost_val = None
    time_val = None
    if model.SolCount > 0:
        total_cost = float(model.getObjective().getValue())
        transport_cost_val = float(total_transport_cost.getValue()) if total_transport_cost.getValue() is not None else None
        invest_cost_val = float(total_invest_cost.getValue()) if total_invest_cost.getValue() is not None else None
        time_val = float(total_time_component.getValue()) if total_time_component.getValue() is not None else None

    summary = {
        'objective': total_cost,
        'transport_cost': transport_cost_val,
        'investment_cost': invest_cost_val,
        'time_component': time_val
    }

    return df_res, summary


# -----------------------------
# Visualization utilities
# -----------------------------

def make_network_plot(df_arcs, df_res, path=PLOT_PNG):
    if nx is None or plt is None:
        print("networkx ou matplotlib non installé - impossible de tracer le graphe")
        return None

    G = nx.DiGraph()
    # Add nodes from arcs
    nodes = set(df_arcs['origin'].astype(str).tolist() + df_arcs['dest'].astype(str).tolist())
    for n in nodes:
        G.add_node(n)

    # Add edges with attributes
    for _, row in df_arcs.iterrows():
        i = str(row.origin); j = str(row.dest)
        # default attributes
        # if 'cost' or 'time' not present, try to aggregate per-organ columns
        def get_numeric_attr(r, base_name, default=0.0):
            if base_name in r.index and pd.notna(r.get(base_name)):
                try:
                    return float(r[base_name])
                except Exception:
                    pass
            # collect cost_<org> or time_<org>
            vals = []
            for c in r.index:
                if c.startswith(base_name + '_'):
                    try:
                        vals.append(float(r[c]))
                    except Exception:
                        pass
            if vals:
                # use average for display
                return float(sum(vals) / len(vals))
            return default
        attrs = {
            'cap': float(row.capacity),
            'cost': get_numeric_attr(row, 'cost', default=0.0),
            'time': get_numeric_attr(row, 'time', default=0.0)
        }
        G.add_edge(i, j, **attrs)

    # Overlay flow from df_res
    flow_dict = {}
    if not df_res.empty:
        # prefer aggregated 'ALL' rows when present
        all_rows = df_res[df_res['organ'] == 'ALL'] if 'organ' in df_res.columns else pd.DataFrame()
        if not all_rows.empty:
            flow_dict = {(r['origin'], r['dest']): r['x'] for _, r in all_rows.iterrows()}
        else:
            # fallback: sum by arc
            for _, r in df_res.iterrows():
                key = (r['origin'], r['dest'])
                flow_dict[key] = flow_dict.get(key, 0.0) + float(r['x'])

    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)

    # node drawing
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_labels(G, pos)

    # edges: width proportional to flow (x) or thin if zero
    edge_widths = []
    edge_colors = []
    for (u_node, v_node) in G.edges():
        f = flow_dict.get((u_node, v_node), 0.0)
        edge_widths.append(1.0 + 4.0 * (f))  # base + scaled
        edge_colors.append('red' if f >= (G[u_node][v_node].get('cap', 0)) - 1e-6 and f>0 else 'black')

    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrowsize=20)

    # edge labels: x/u
    edge_labels = {}
    for (i,j) in G.edges():
        f = flow_dict.get((i,j), 0.0)
        cap = G[i][j].get('cap', 0)
        edge_labels[(i,j)] = f"{f:.2f}/{cap:.2f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title('Réseau - flux (x) / capacité (u)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def make_pdf_report(df_arcs, df_res, summary, path=REPORT_PDF):
    if PdfPages is None or plt is None:
        print("matplotlib/PdfPages non disponible - impossible de générer PDF")
        return None

    with PdfPages(path) as pdf:
        # Page 1 : résumé texte
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.clf()
        ax = fig.add_subplot(111)
        ax.axis('off')
        lines = []
        lines.append('Rapport d\'optimisation - Transport urgent d\'organes')
        lines.append('')
        def fmt(v):
            try:
                return f"{v:.2f}" if v is not None else "N/A"
            except Exception:
                return str(v)

        lines.append(f"Objectif (valeur) : {fmt(summary.get('objective'))}")
        lines.append(f"Transport cost component : {fmt(summary.get('transport_cost'))}")
        lines.append(f"Investment cost component : {fmt(summary.get('investment_cost'))}")
        lines.append(f"Time component (weighted) : {fmt(summary.get('time_component'))}")
        lines.append('')
        lines.append('Flux par arc (x) et capacité achetée (y) :')

        txt = '\n'.join(lines)
        ax.text(0.01, 0.98, txt, va='top', fontsize=10, family='monospace')
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2 : network plot (if generated)
        if PLOT_PNG.exists():
            fig2 = plt.figure(figsize=(8.27, 11.69))
            img = plt.imread(str(PLOT_PNG))
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(fig2)
            plt.close(fig2)

        # Page 3 : table of flows (if any)
        fig3 = plt.figure(figsize=(8.27, 11.69))
        fig3.clf()
        ax3 = fig3.add_subplot(111)
        ax3.axis('off')
        # render df_res as table if we have rows, otherwise print a message
        if df_res is not None and not df_res.empty:
            table = ax3.table(cellText=df_res.round(3).values, colLabels=df_res.columns, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
        else:
            ax3.text(0.1, 0.6, 'Aucun flux trouvé: aucun résultat à afficher.', fontsize=12)
        pdf.savefig(fig3)
        plt.close(fig3)

    return path


# -----------------------------
# PyQt viewer (optionnel)
# -----------------------------
class ResultsWindow(QMainWindow):
    def __init__(self, df_res, summary, plot_path=None):
        super().__init__()
        self.setWindowTitle("Résultats - Transport d'organes")
        self.resize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        # Summary labels
        def fmt(v):
            try:
                return f"{v:.2f}" if v is not None else "N/A"
            except Exception:
                return str(v)

        lbl = QLabel(f"Objectif = {fmt(summary.get('objective'))} | Transport cost = {fmt(summary.get('transport_cost'))} | Invest = {fmt(summary.get('investment_cost'))}")
        vbox.addWidget(lbl)

        # Table of results
        table = QTableWidget()
        table.setColumnCount(len(df_res.columns))
        table.setHorizontalHeaderLabels(list(df_res.columns))
        table.setRowCount(len(df_res))
        for i, row in df_res.iterrows():
            for j, col in enumerate(df_res.columns):
                item = QTableWidgetItem(str(row[col]))
                table.setItem(i, j, item)
        vbox.addWidget(table)

        # Plot (if available) as Matplotlib canvas
        if plot_path and os.path.exists(plot_path):
            fig = Figure(figsize=(6,4))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            img = plt.imread(plot_path)
            ax.imshow(img)
            ax.axis('off')
            vbox.addWidget(canvas)

        # Buttons
        hbox = QHBoxLayout()
        btn_export = QPushButton('Exporter results.csv')
        btn_export.clicked.connect(lambda: df_res.to_csv(str(RESULTS_CSV), index=False))
        hbox.addWidget(btn_export)
        vbox.addLayout(hbox)


# -----------------------------
# Main command-line interface
# -----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='Solve PL and generate visual outputs')
    parser.add_argument('--show', action='store_true', help='Open PyQt window to show results')
    parser.add_argument('--write-lp', action='store_true', help='Export model to LP file')
    args = parser.parse_args(argv)

    try:
        df_nodes, df_arcs, params = read_inputs()
    except Exception as e:
        print('Erreur lecture données :', e)
        traceback.print_exc()
        sys.exit(1)

    try:
        df_res, summary = solve_model(df_nodes, df_arcs, params, write_lp=args.write_lp)
    except Exception as e:
        # If Gurobi not installed, write a solver status file for the IHM
        if isinstance(e, ImportError):
            try:
                with open(DATA_DIR / 'solver_status.txt', 'w', encoding='utf-8') as sf:
                    sf.write('NO_SOLVER')
            except Exception:
                pass
        print('Erreur résolution :', e)
        traceback.print_exc()
        sys.exit(1)

    # Save results
    df_res.to_csv(RESULTS_CSV, index=False)
    print(f'Results saved to {RESULTS_CSV}')

    # Make network plot
    try:
        plot_path = make_network_plot(df_arcs, df_res, path=PLOT_PNG)
        if plot_path:
            print(f'Network plot saved to {plot_path}')
    except Exception as e:
        print('Erreur génération plot :', e)

    # Make PDF report
    try:
        report_path = make_pdf_report(df_arcs, df_res, summary, path=REPORT_PDF)
        if report_path:
            print(f'Report PDF saved to {report_path}')
    except Exception as e:
        print('Erreur génération PDF :', e)

    # Show GUI if requested and PyQt available
    if args.show:
        app = QApplication(sys.argv)
        win = ResultsWindow(df_res, summary, plot_path if 'plot_path' in locals() else None)
        win.show()
        app.exec()


if __name__ == '__main__':
    main()
