# Quick Start Guide

## 1. Installation (first time only)

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\\.venv\\Scripts\\Activate.ps1

# Install Python dependencies
pip install -r project/requirements.txt

# [Optional] Install Gurobi for exact solving
# Download from https://www.gurobi.com and follow their installer
```

## 2. Run the App

```powershell
python -m project.main
```

The PyQt5 GUI window should open.

## 3. Solve a Simple Example

1. **Load Graph**
   - Click "Load Graph"
   - Select `project/example_graph.txt`
   - You should see: "Loaded graph with 8 nodes and 9 edges"

2. **Review Weights**
   - Click "Assign / Edit Weights"
   - The table shows all nodes and their weights (already loaded from the file)
   - You can change any weight if needed
   - Click "Save" to update

3. **Run the Solver**
   - Click "Run Optimization"
   - Wait a moment (or a few seconds for larger graphs)
   - You should see results like: "Selected nodes: A, D, F" and "objective=10.0000"

4. **Visualize**
   - Click "Visualize Graph"
   - Red nodes = selected influencers (dominating set)
   - Light blue nodes = not selected

5. **View Solution**
   - Click "Show Solution Text"
   - See the list of selected nodes and count

6. **Save Weights (optional)**
   - Click "Save Weights to File"
   - Weights are written to a companion file (e.g. `example_graph.txt.node_weights.txt`)

## 4. Load Your Own Graph

### Format for TXT files

```
# Comments start with #
# Format: either "node weight" or "node1 node2" for edges

A 5
B 3
C 4
A B
B C
C A
```

### Format for CSV files

```
# Two columns where second is numeric: node weight
# Two columns where second is non-numeric: edge
A,5
B,3
A,B
B,C
```

Then just click "Load Graph" and select your file.

## Troubleshooting

- **GUI doesn't open:** Check that PyQt5 is installed: `pip install PyQt5`
- **Import error:** Make sure you're running with `python -m project.main` (not `python project/main.py`)
- **Solver error:** Check the log in the GUI. If Gurobi is not installed, a greedy approximation is used automatically.
- **Weight parsing:** Review `WEIGHTS_FORMAT.md` for detailed rules on how weights are parsed.

## More Info

- See `README.md` for project overview and architecture
- See `WEIGHTS_FORMAT.md` for detailed weight parsing rules
