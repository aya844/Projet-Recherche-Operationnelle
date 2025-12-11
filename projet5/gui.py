from __future__ import annotations

import sys
import traceback
from typing import Dict, Iterable, List

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QSplitter,
    QAction,
    QGroupBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx

from .graph_utils import load_graph_from_edge_list, neighbors_map, draw_graph, save_weights_to_file
from .solver import solve_mwds


class SolverThread(QThread):
    finished = pyqtSignal(list, float, float) 
    error = pyqtSignal(str)

    def __init__(self, G: nx.Graph, weights: Dict[str, float], time_limit: float | None = 60.0):
        super().__init__()
        self.G = G
        self.weights = weights
        self.time_limit = time_limit

    def run(self) -> None: 
        try:
            nodes = [str(n) for n in self.G.nodes()]
            neigh = neighbors_map(self.G)
            selected, obj, runtime = solve_mwds(nodes, neigh, self.weights, time_limit=self.time_limit)
            self.finished.emit(selected, obj, runtime)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"Solver error: {e}\n{tb}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MWDS - Gurobi + PyQt5")
        self.setGeometry(50, 50, 1200, 800)

        self.G: nx.Graph | None = None
        self.pos = None
        self.weights: Dict[str, float] = {}
        self.selected: List[str] = []
        self.graph_path: str | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        load_action = QAction("Load Graph", self)
        load_action.triggered.connect(self.load_graph)
        file_menu.addAction(load_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left = QVBoxLayout()

        self.load_btn = QPushButton("Load Graph")
        self.load_btn.clicked.connect(self.load_graph)
        left.addWidget(self.load_btn)

        self.weights_btn = QPushButton("Assign / Edit Weights")
        self.weights_btn.clicked.connect(self.edit_weights)
        left.addWidget(self.weights_btn)

        self.save_weights_btn = QPushButton("Save Weights to File")
        self.save_weights_btn.clicked.connect(self.save_weights)
        left.addWidget(self.save_weights_btn)

        self.run_btn = QPushButton("Run Optimization")
        self.run_btn.clicked.connect(self.run_optimization)
        left.addWidget(self.run_btn)

        self.show_btn = QPushButton("Show Solution Text")
        self.show_btn.clicked.connect(self.show_solution_text)
        left.addWidget(self.show_btn)

        self.visualize_btn = QPushButton("Visualize Graph")
        self.visualize_btn.clicked.connect(self.visualize_graph)
        left.addWidget(self.visualize_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        left.addWidget(self.clear_btn)

        left.addStretch()

        result_box = QGroupBox("Solution / Logs")
        r_layout = QVBoxLayout(result_box)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        r_layout.addWidget(self.result_text)
        left.addWidget(result_box)

        left_widget = QWidget()
        left_widget.setLayout(left)

        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def load_graph(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open edge list", "", "Edge list (*.txt *.csv);;All files (*)")
        if not path:
            return
        try:
            G, file_weights = load_graph_from_edge_list(path)
            if G.number_of_nodes() == 0:
                QMessageBox.warning(self, "Empty graph", "The loaded graph contains no nodes.")
                return
            self.G = G
            self.pos = None
            self.graph_path = path
            self.weights = file_weights
            self.result_text.append(f"Loaded graph '{path}' with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            self.result_text.append(f"Weights loaded from file (or defaults assigned).")

            self.run_btn.setEnabled(True)
            self.weights_btn.setEnabled(True)
            self.save_weights_btn.setEnabled(True)
            self.visualize_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def edit_weights(self) -> None:
        if self.G is None:
            QMessageBox.warning(self, "No graph", "Load a graph first.")
            return
        
        dlg = QWidget()
        dlg.setWindowTitle("Edit Node Weights")
        dlg.setGeometry(200, 200, 400, 600)
        layout = QVBoxLayout(dlg)
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Node", "Weight"])
        nodes = list(self.G.nodes())
        table.setRowCount(len(nodes))
        for i, n in enumerate(nodes):
            item_n = QTableWidgetItem(str(n))
            item_n.setFlags(item_n.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 0, item_n)
            w = self.weights.get(str(n), 1.0)
            item_w = QTableWidgetItem(str(w))
            table.setItem(i, 1, item_w)
        layout.addWidget(table)

        def save_and_close():
            try:
                for i, n in enumerate(nodes):
                    w_item = table.item(i, 1)
                    if w_item is None:
                        continue
                    w = float(w_item.text())
                    self.weights[str(n)] = float(w)
                dlg.close()
                self.result_text.append("Weights updated in memory.")
            except Exception as e:
                QMessageBox.critical(self, "Invalid weight", str(e))

        btn_save = QPushButton("Save")
        btn_save.clicked.connect(save_and_close)
        layout.addWidget(btn_save)
        dlg.show()

    def run_optimization(self) -> None:
        if self.G is None:
            QMessageBox.warning(self, "No graph", "Load a graph first.")
            return
        if not self.weights:
            QMessageBox.warning(self, "No weights", "Assign weights first.")
            return
        self.run_btn.setEnabled(False)
        self.result_text.append("Starting optimization (in background)...")
        self.thread = SolverThread(self.G, self.weights)
        self.thread.finished.connect(self._on_solver_finished)
        self.thread.error.connect(self._on_solver_error)
        self.thread.start()

    def _on_solver_finished(self, selected: List[str], obj: float, runtime: float) -> None:
        self.selected = selected
        self.result_text.append(f"Solver finished: objective={obj:.4f}, runtime={runtime:.3f}s")
        self.result_text.append(f"Selected nodes: {', '.join(map(str, selected))}")
        self.run_btn.setEnabled(True)
        self.show_btn.setEnabled(True)
        
        self.visualize_graph()

    def _on_solver_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Solver error", msg)
        self.result_text.append(msg)
        self.run_btn.setEnabled(True)

    def show_solution_text(self) -> None:
        if not self.selected:
            QMessageBox.information(self, "No solution", "No solution available. Run the solver first.")
            return
        text = f"Selected nodes (count {len(self.selected)}):\n" + '\n'.join(map(str, self.selected))
        QMessageBox.information(self, "Solution", text)

    def visualize_graph(self) -> None:
        if self.G is None:
            QMessageBox.warning(self, "No graph", "Load a graph first.")
            return
        ax = self.figure.subplots()
        sel = [str(s) for s in self.selected] if self.selected else []
        self.pos = draw_graph(ax, self.G, pos=self.pos, selected=sel)
        self.canvas.draw()

    def save_weights(self) -> None:
        if self.G is None or self.graph_path is None:
            QMessageBox.warning(self, "No graph", "Load a graph first.")
            return
        try:
            out_path = save_weights_to_file(self.graph_path, self.G, self.weights)
            self.result_text.append(f"Weights saved to '{out_path}'.")
            QMessageBox.information(self, "Success", f"Weights saved to {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def clear_all(self) -> None:
        self.G = None
        self.pos = None
        self.weights = {}
        self.selected = []
        self.graph_path = None
        self.result_text.clear()
        self.result_text.append("Cleared graph, weights, and solution.")
        try:
            self.figure.clear()
            self.canvas.draw()
        except Exception:
            pass
        try:
            self.run_btn.setEnabled(False)
            self.weights_btn.setEnabled(False)
            self.save_weights_btn.setEnabled(False)
            self.show_btn.setEnabled(False)
            self.visualize_btn.setEnabled(False)
        except Exception:
            pass


def main(argv=None) -> int:
    app = QApplication(argv or sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
