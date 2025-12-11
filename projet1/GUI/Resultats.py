from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QHBoxLayout, QDialog, QDialogButtonBox
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap
import subprocess
import sys
import pandas as pd
import os

class ResultsWindow(QMainWindow):
    def __init__(self, results_csv="data/results.csv"):
        super().__init__()
        self.setWindowTitle("Résultats - Transport d'Organes")
        self.resize(900, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Show solver status
        self.status_label = QLabel('Statut: inconnu')
        self.result_msg_label = QLabel('')
        layout.addWidget(self.status_label)
        layout.addWidget(self.result_msg_label)
        # start a timer to refresh status
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start()

        if not os.path.exists(results_csv):
            layout.addWidget(QLabel(f"Fichier {results_csv} non trouvé."))
            return
        try:
            df = pd.read_csv(results_csv)
        except pd.errors.EmptyDataError:
            layout.addWidget(QLabel("Fichier de résultats vide ou sans colonnes."))
            return
        except Exception as e:
            layout.addWidget(QLabel(f"Erreur lecture résultats: {e}"))
            return

        lbl = QLabel(f"Nombre d'arcs optimisés: {len(df)}")
        layout.addWidget(lbl)

        table = QTableWidget(len(df), len(df.columns))
        table.setHorizontalHeaderLabels(list(df.columns))

        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                table.setItem(i, j, QTableWidgetItem(str(row[col])))

        layout.addWidget(table)

        btn_export = QPushButton("Exporter CSV")
        btn_export.clicked.connect(lambda: df.to_csv(results_csv, index=False))
        layout.addWidget(btn_export)
        if df.empty:
            btn_export.setEnabled(False)
        # viewers (plot + PDF)
        view_h = QHBoxLayout()
        self.btn_view_plot = QPushButton("Voir graphe")
        self.btn_view_plot.setEnabled(False)
        self.btn_view_plot.clicked.connect(self.show_plot_dialog)
        view_h.addWidget(self.btn_view_plot)
        self.btn_view_pdf = QPushButton("Voir rapport PDF")
        self.btn_view_pdf.setEnabled(False)
        self.btn_view_pdf.clicked.connect(self.open_pdf)
        view_h.addWidget(self.btn_view_pdf)
        layout.addLayout(view_h)
        # initial status refresh
        self._refresh_status()

    def _refresh_status(self):
        status_file = os.path.join(os.getcwd(), 'data', 'solver_status.txt')
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    s = f.read().strip()
                self.status_label.setText(f'Statut: {s}')
                # set a friendly message about the solution
                if s.upper() == 'OPTIMAL':
                    self.result_msg_label.setText('Solution optimale trouvée.')
                    self.result_msg_label.setStyleSheet('color: green')
                elif s.upper() == 'FEASIBLE' or s.upper().startswith('DONE'):
                    self.result_msg_label.setText('Une solution (peut-être suboptimale) a été trouvée.')
                    self.result_msg_label.setStyleSheet('color: green')
                elif s.upper() == 'INFEASIBLE':
                    self.result_msg_label.setText('Aucune solution: modèle INFEASIBLE.')
                    self.result_msg_label.setStyleSheet('color: red')
                elif s.upper() == 'UNBOUNDED':
                    self.result_msg_label.setText('Aucune solution valide: modèle UNBOUNDED.')
                    self.result_msg_label.setStyleSheet('color: red')
                else:
                    self.result_msg_label.setText('Statut inconnu')
                    self.result_msg_label.setStyleSheet('color: black')
                # enable plot/pdf buttons if solver found a solution or if files exist
                plot_path = os.path.join(os.getcwd(), 'data', 'network_plot.png')
                report_path = os.path.join(os.getcwd(), 'data', 'report.pdf')
                if s.upper() in ('OPTIMAL','FEASIBLE','DONE','DONE:OK'):
                    self.btn_view_plot.setEnabled(os.path.exists(plot_path))
                    self.btn_view_pdf.setEnabled(os.path.exists(report_path))
                else:
                    # allow viewing if files exist, but keep disabled when infeasible/unbounded
                    if s.upper() in ('INFEASIBLE','UNBOUNDED'):
                        self.btn_view_plot.setEnabled(False)
                        self.btn_view_pdf.setEnabled(False)
                    else:
                        self.btn_view_plot.setEnabled(os.path.exists(plot_path))
                        self.btn_view_pdf.setEnabled(os.path.exists(report_path))
            except Exception:
                self.status_label.setText('Statut: erreur lecture')
        else:
            self.status_label.setText('Statut: non exécuté')
            self.result_msg_label.setText('Aucune exécution du solveur')
            self.result_msg_label.setStyleSheet('color: black')
            # no run yet
            self.btn_view_plot.setEnabled(False)
            self.btn_view_pdf.setEnabled(False)

    def show_plot_dialog(self):
        plot_path = os.path.join(os.getcwd(), 'data', 'network_plot.png')
        if not os.path.exists(plot_path):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Graphe non trouvé', f'Fichier {plot_path} introuvable.')
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('Graphe - flux par arc')
        v = QVBoxLayout(dlg)
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = QPixmap(plot_path)
        if pix.isNull():
            label.setText('Impossible de charger le graphe.')
        else:
            screen_w = dlg.screen().size().width() if dlg.screen() is not None else 800
            scaled = pix.scaledToWidth(int(screen_w * 0.6), Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled)
        v.addWidget(label)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bb.accepted.connect(dlg.accept)
        v.addWidget(bb)
        dlg.exec()

    def open_pdf(self):
        report_path = os.path.join(os.getcwd(), 'data', 'report.pdf')
        if not os.path.exists(report_path):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Rapport non trouvé', f'Fichier {report_path} introuvable.')
            return
        try:
            if sys.platform.startswith('win'):
                os.startfile(report_path)
            else:
                subprocess.Popen(['xdg-open', report_path])
        except Exception as e:
            print('Unable to open PDF:', e)


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = ResultsWindow()
    win.show()
    sys.exit(app.exec())
