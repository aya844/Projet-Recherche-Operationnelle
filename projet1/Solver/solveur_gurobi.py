"""
solver/run_solver.py

Script complet pour :
 - lire les CSV (data/nodes.csv, data/arcs.csv, data/params.csv)
 - construire et résoudre le PL (Gurobi)
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
    data/arcs.csv   (colonnes: origin,dest,cost,time,capacity,cap_cost)
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

import gurobipy as gp
from gurobipy import GRB, quicksum
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
    # expected columns: id,type,name,supply
    required_nodes_cols = {"id", "type", "name", "supply"}
    if not required_nodes_cols.issubset(df_nodes.columns):
        raise ValueError(f"nodes.csv doit contenir les colonnes {required_nodes_cols}")

    # arcs
    if not ARCS_CSV.exists():
        raise FileNotFoundError(f"Fichier manquant : {ARCS_CSV}")
    df_arcs = pd.read_csv(ARCS_CSV)
    required_arcs_cols = {"origin", "dest", "cost", "time", "capacity", "cap_cost"}
    if not required_arcs_cols.issubset(df_arcs.columns):
        raise ValueError(f"arcs.csv doit contenir les colonnes {required_arcs_cols}")

    # params (optional)
    params = {"alpha": 1.0, "beta": 10.0}
    if PARAMS_CSV.exists():
        dfp = pd.read_csv(PARAMS_CSV, header=None)
        # If params file is two-column CSV key,value
        try:
            params = dict(zip(dfp[0].astype(str), dfp[1].astype(float)))
        except Exception:
            # fallback: try first row as header
            dpf = pd.read_csv(PARAMS_CSV)
            for k in ["alpha","beta"]:
                if k in dpf.columns:
                    params[k] = float(dpf[k].iloc[0])

    return df_nodes, df_arcs, params


# -----------------------------
# Build and solve model
# -----------------------------

def solve_model(df_nodes, df_arcs, params, write_lp=False):
    # Preprocess
    nodes = df_nodes['id'].astype(str).tolist()
    b = dict(zip(df_nodes['id'].astype(str), df_nodes['supply'].astype(float)))

    arcs = [(str(row.origin), str(row.dest)) for _, row in df_arcs.iterrows()]

    u = { (str(row.origin), str(row.dest)): float(row.capacity) for _, row in df_arcs.iterrows() }
    c = { (str(row.origin), str(row.dest)): float(row.cost)     for _, row in df_arcs.iterrows() }
    k = { (str(row.origin), str(row.dest)): float(row.cap_cost) for _, row in df_arcs.iterrows() }
    tt = { (str(row.origin), str(row.dest)): float(row.time)     for _, row in df_arcs.iterrows() }

    alpha = float(params.get('alpha', 1.0))
    beta  = float(params.get('beta', 10.0))

    model = gp.Model("Organs_Transport_PL")
    model.setParam('OutputFlag', 1)  # 0 to silence

    # Variables
    x = { (i,j): model.addVar(lb=0.0, name=f"x_{i}_{j}") for (i,j) in arcs }
    y = { (i,j): model.addVar(lb=0.0, name=f"y_{i}_{j}") for (i,j) in arcs }
    model.update()

    # Constraints : conservation des flux
    for n in nodes:
        out_vars = [x[(i,j)] for (i,j) in arcs if i == n]
        in_vars  = [x[(i,j)] for (i,j) in arcs if j == n]
        # If node has supply/demand but no incident arcs, the equality may be infeasible
        model.addConstr(gp.quicksum(out_vars) - gp.quicksum(in_vars) == float(b.get(n, 0.0)), name=f"flow_{n}")

    # Capacity constraints
    for (i,j) in arcs:
        model.addConstr(x[(i,j)] <= u[(i,j)] + y[(i,j)], name=f"cap_{i}_{j}")

    # Objective
    total_transport_cost = gp.quicksum(c[(i,j)] * x[(i,j)] for (i,j) in arcs)
    total_time_component  = gp.quicksum(tt[(i,j)]  * x[(i,j)] for (i,j) in arcs)
    total_invest_cost = gp.quicksum(k[(i,j)] * y[(i,j)] for (i,j) in arcs)

    obj = alpha * total_transport_cost + beta * total_time_component + total_invest_cost
    model.setObjective(obj, GRB.MINIMIZE)

    if write_lp:
        model.write(str(DATA_DIR / "model.lp"))

    model.optimize()

    # Check status
    if model.status != GRB.OPTIMAL:
        print("Attention : solution optimale non trouvée. Statut=", model.status)

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
        for (i,j) in arcs:
            xv = x[(i,j)].X if hasattr(x[(i,j)], 'X') and x[(i,j)].X is not None else 0.0
            yv = y[(i,j)].X if hasattr(y[(i,j)], 'X') and y[(i,j)].X is not None else 0.0
            rows.append({
                'origin': i, 'dest': j,
                'x': float(xv), 'u': float(u[(i,j)]), 'y': float(yv),
                'cost': float(c[(i,j)]), 'cap_cost': float(k[(i,j)]), 'time': float(tt[(i,j)])
            })
    else:
        # No solution available; rows remains empty
        rows = []

    # Ensure df_res has the expected columns even if empty, so downstream tools don't crash
    cols = ['origin','dest','x','u','y','cost','cap_cost','time']
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
        attrs = {
            'cap': float(row.capacity),
            'cost': float(row.cost),
            'time': float(row.time)
        }
        G.add_edge(i, j, **attrs)

    # Overlay flow from df_res
    flow_dict = {(r['origin'], r['dest']): r['x'] for _, r in df_res.iterrows()} if not df_res.empty else {}

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
