# MWDS - Minimum Weighted Dominating Set

**Author:** Ferdawes-Benali

This project demonstrates solving the Minimum Weighted Dominating Set problem using Gurobi for exact optimization and PyQt5 for a graphical interface.

## Project Structure

- `main.py` — entry point
- `gui.py` — PyQt5 GUI implementation
- `solver.py` — Gurobi-based solver with greedy fallback
- `graph_utils.py` — graph loading, weight parsing, and visualization
- `example_graph.txt` — example graph with explicit node weights
- `example_graph.csv` — example graph in CSV format
- `requirements.txt` — Python dependencies
- `WEIGHTS_FORMAT.md` — detailed weight parsing and handling guide

## Setup

### 1. Create and activate a Python 3.10+ virtual environment

Windows (PowerShell):
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r project/requirements.txt
```

### 3. Install Gurobi (optional, but recommended for exact solving)

- Download and install Gurobi from https://www.gurobi.com
- Follow Gurobi's instructions to install the Python API (gurobipy)
- Without Gurobi, the app falls back to a greedy approximation

## Run

```powershell
python -m project.main
```

Or:
```powershell
python project/main.py
```

## Quick Example

1. Click "Load Graph" → select `project/example_graph.txt`
2. Click "Assign / Edit Weights" to view/modify node weights (or keep defaults)
3. Click "Run Optimization" to solve
4. Click "Visualize Graph" to see selected influencers (red = selected, light blue = not selected)
5. Click "Show Solution Text" to see the list of selected nodes

## Weight Format

The loader recognizes explicit node-weight entries:
- TXT format: `A 5` means node A has weight 5
- CSV format: `A,5` means node A has weight 5

Edges are specified with two non-numeric tokens:
- TXT: `A B` is an edge
- CSV: `A,B` is an edge

See `WEIGHTS_FORMAT.md` for detailed parsing rules.
