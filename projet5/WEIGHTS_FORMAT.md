# Weight Format Guide

## How to Add Weights to Your Graph 

The loader now uses deterministic rules so there is no ambiguity between edges and node weights.

Parsing rules
- Lines/rows with exactly two tokens where the second token is numeric are treated as `node weight` entries (e.g. `A 2` or CSV `A,2`).
- Lines/rows with exactly two tokens where the second token is non-numeric are treated as edges `u v` (e.g. `A B`).
- Lines/rows with three or more tokens are treated as edges `u v [edge_weight]` and the third token is recorded as an edge attribute (not a node weight).

Default weights
- Any node without an explicit node-weight entry receives the default weight `1.0`.

Saving weights
- To avoid overwriting edge lists and keep formats clear, node weights are saved to a companion file when you click "Save Weights to File". The companion file is named `<original>.node_weights.csv` for CSV inputs or `<original>.node_weights.txt` for text inputs and contains lines `node,weight` (or `node weight`).

Examples
- TXT with explicit node weights and edges:
  ```
  # node weights and edges mixed
  A 5         # node A weight = 5
  B 3         # node B weight = 3
  A B         # edge A-B
  C D 2       # edge C-D with edge weight 2 (not a node weight)
  ```

- CSV examples:
  - Node-weight file rows: `A,5` means node A weight = 5
  - Edge rows with optional edge-weight: `A,B,2` means edge A-B with edge weight 2

Workflow
1. Load your graph file (`Load Graph`). The app reads explicit node weights (if present) and assigns defaults to the rest.
2. Edit weights in the UI (`Assign / Edit Weights`). Changes are kept in memory.
3. Save weights to a companion file (`Save Weights to File`). The app writes a companion `*.node_weights.*` file and reports its path.
4. Run the solver using the loaded/edited weights.
