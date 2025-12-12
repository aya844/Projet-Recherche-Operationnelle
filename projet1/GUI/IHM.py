from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHBoxLayout, QGridLayout, QDoubleSpinBox, QApplication, QStatusBar, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QBrush
import sys
import csv
import subprocess
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimisation Transport d'Organes")
        self.resize(900, 600)

        # compute project root relative to this file (stable across CWDs)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Nœuds ---
        self.nodes_table = QTableWidget(0, 4)
        self.nodes_table.setHorizontalHeaderLabels(["id", "type", "name", "supply"])
        layout.addWidget(QLabel("Nœuds"))
        layout.addWidget(self.nodes_table)
        # connect immediate validation handler
        self.nodes_table.itemChanged.connect(self.on_nodes_item_changed)

        btn_add_node = QPushButton("Ajouter nœud")
        btn_add_node.clicked.connect(lambda: self.nodes_table.insertRow(self.nodes_table.rowCount()))
        # Add / remove node buttons with shortcuts
        btn_remove_node = QPushButton('Supprimer nœud')
        btn_remove_node.setToolTip('Supprimer la ligne sélectionnée pour les nœuds (Del)')
        btn_remove_node.clicked.connect(lambda: self.remove_selected_row(self.nodes_table))
        node_h = QHBoxLayout()
        node_h.addWidget(btn_add_node)
        node_h.addWidget(btn_remove_node)
        layout.addLayout(node_h)

        # --- Arcs ---
        self.arcs_table = QTableWidget(0, 6)
        self.arcs_table.setHorizontalHeaderLabels(["origin", "dest", "cost", "time", "capacity", "cap_cost"])
        layout.addWidget(QLabel("Arcs"))
        layout.addWidget(self.arcs_table)
        # connect immediate validation handler
        self.arcs_table.itemChanged.connect(self.on_arcs_item_changed)

        btn_add_arc = QPushButton("Ajouter arc")
        btn_add_arc.clicked.connect(lambda: self.arcs_table.insertRow(self.arcs_table.rowCount()))
        btn_remove_arc = QPushButton('Supprimer arc')
        btn_remove_arc.setToolTip('Supprimer la ligne sélectionnée pour les arcs (Del)')
        btn_remove_arc.clicked.connect(lambda: self.remove_selected_row(self.arcs_table))
        arc_h = QHBoxLayout()
        arc_h.addWidget(btn_add_arc)
        arc_h.addWidget(btn_remove_arc)
        layout.addLayout(arc_h)

        # --- Paramètres ---
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("Alpha (coût)"), 0, 0)
        self.alpha = QDoubleSpinBox()
        self.alpha.setValue(0.7)
        param_layout.addWidget(self.alpha, 0, 1)

        param_layout.addWidget(QLabel("Beta (temps)"), 1, 0)
        self.beta = QDoubleSpinBox()
        self.beta.setValue(0.3)
        # place Beta spinbox in the grid at (1, 1)
        param_layout.addWidget(self.beta, 1, 1)

        layout.addLayout(param_layout)

        # --- Organ types and vehicle capacity ---
        organ_h = QHBoxLayout()
        organ_h.addWidget(QLabel('Organ types (comma-separated)'))
        self.organs_input = QLineEdit()
        organ_h.addWidget(self.organs_input)
        btn_set_organs = QPushButton('Set organ types')
        btn_set_organs.clicked.connect(self.set_organ_types)
        organ_h.addWidget(btn_set_organs)
        layout.addLayout(organ_h)

        vc_h = QHBoxLayout()
        vc_h.addWidget(QLabel('Vehicle capacity (V)'))
        self.vehicle_capacity = QDoubleSpinBox()
        self.vehicle_capacity.setDecimals(2)
        self.vehicle_capacity.setValue(1.0)
        vc_h.addWidget(self.vehicle_capacity)
        layout.addLayout(vc_h)

        # default organ list
        self.organs_list = []

        # --- Boutons ---
        btn_save = QPushButton("Enregistrer CSV")
        btn_save.clicked.connect(self.save_csv_files)
        layout.addWidget(btn_save)

        btn_load = QPushButton('Charger CSV')
        btn_load.setToolTip('Charger les CSV depuis le dossier data et remplir les tableaux')
        btn_load.clicked.connect(self.load_csvs)
        layout.addWidget(btn_load)

        btn_run = QPushButton("Résoudre avec Gurobi")
        btn_run.clicked.connect(self.run_gurobi)
        layout.addWidget(btn_run)

        # Status bar for messages
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)
        self.status.showMessage('Prêt')

    def save_csv_files(self):
        # Validate before saving
        valid, errors = self.validate_tables()
        if not valid:
            # Show errors in a message box and abort save
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Les fichiers CSV contiennent des erreurs :")
            msg.setInformativeText("\n".join(errors[:20]))
            msg.setWindowTitle("Validation CSV")
            msg.exec()
            return

        # Make sure the 'data' folder exists (lowercase is used by the solver)
        data_dir = os.path.join(self.project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        # nodes.csv
        with open(os.path.join(data_dir, 'nodes.csv'), "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            # dynamic header
            if hasattr(self, 'organs_list') and self.organs_list:
                header = ["id", "type", "name"] + [f"supply_{o}" for o in self.organs_list]
            else:
                header = ["id", "type", "name", "supply"]
            writer.writerow(header)
            for r in range(self.nodes_table.rowCount()):
                row = []
                for c in range(self.nodes_table.columnCount()):
                    item = self.nodes_table.item(r, c)
                    cell_text = item.text() if item else ""
                    # For supply columns, attempt to coerce labels to numeric
                    if header[c].startswith('supply'):
                        val = cell_text.strip()
                        try:
                            num = float(val)
                        except Exception:
                            low = val.lower()
                            if low in ("offre", "o", "supply"):
                                num = 1.0
                            elif low in ("demande", "d", "demand"):
                                num = -1.0
                            else:
                                try:
                                    num = float(val.replace(',',''))
                                except Exception:
                                    print(f"Warning: unable to parse supply value '{val}' on row {r}; defaulting to 0")
                                    num = 0.0
                        row.append(num)
                    else:
                        row.append(cell_text)
                writer.writerow(row)

        # arcs.csv
        with open(os.path.join(data_dir, 'arcs.csv'), "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            base_cols = ["origin", "dest", "cost", "time", "capacity", "cap_cost"]
            if hasattr(self, 'organs_list') and self.organs_list:
                header = base_cols + [f"cost_{o}" for o in self.organs_list] + [f"time_{o}" for o in self.organs_list]
            else:
                header = base_cols
            writer.writerow(header)
            for r in range(self.arcs_table.rowCount()):
                row = []
                for c in range(self.arcs_table.columnCount()):
                    item = self.arcs_table.item(r, c)
                    cell_text = item.text() if item else ""
                    # try to convert numeric fields to numbers when possible
                    if header[c] in ("cost", "time", "capacity", "cap_cost") or header[c].startswith('cost_') or header[c].startswith('time_'):
                        try:
                            val = float(cell_text)
                        except Exception:
                            try:
                                val = float(cell_text.replace(',',''))
                            except Exception:
                                val = 0.0
                        row.append(val)
                    else:
                        row.append(cell_text)
                writer.writerow(row)

        # params.csv
        with open(os.path.join(data_dir, 'params.csv'), "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["alpha", self.alpha.value()])
            writer.writerow(["beta", self.beta.value()])
            writer.writerow(["vehicle_capacity", self.vehicle_capacity.value()])
            organs_val = self.organs_input.text().strip()
            writer.writerow(["organs", organs_val])

        print("CSV sauvegardés.")

    def run_gurobi(self):
        # Validate and save CSVs then use current Python to run the solver wrapper
        # This ensures the solver receives valid CSVs.
        valid, errors = self.validate_tables()
        if not valid:
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Les fichiers CSV contiennent des erreurs :")
            msg.setInformativeText("\n".join(errors[:20]))
            msg.setWindowTitle("Validation CSV")
            msg.exec()
            return

        # Save them to disk before running
        self.save_csv_files()

        # Use the same Python executable to run the solver wrapper
        import sys
        script_path = os.path.join(os.path.dirname(__file__), '..', 'Solver', 'run_solver.py')
        script_path = os.path.normpath(script_path)
        # disable editing while solver runs
        self.nodes_table.setEnabled(False)
        self.arcs_table.setEnabled(False)
        subprocess.Popen([sys.executable, script_path])
        # Start polling the solver status file so we can update status bar
        self.start_polling_solver_status()


    def validate_tables(self):
        """Validate the contents of nodes and arcs tables.
        Returns (valid: bool, errors: list[str])"""
        errors = []
        # Nodes table checks
        for r in range(self.nodes_table.rowCount()):
            # ID
            id_item = self.nodes_table.item(r, 0)
            id_val = id_item.text().strip() if id_item else ""
            if not id_val:
                errors.append(f"Nœuds ligne {r+1} : ID vide")
            # Type
            type_item = self.nodes_table.item(r, 1)
            type_val = type_item.text().strip() if type_item else ""
            if not type_val:
                errors.append(f"Nœuds ligne {r+1} : Type vide")
            # Name
            name_item = self.nodes_table.item(r, 2)
            name_val = name_item.text().strip() if name_item else ""
            if not name_val:
                errors.append(f"Nœuds ligne {r+1} : Nom vide")
            # Supply: numeric (float) or 'Offre'/'Demande' - may be multiple supply columns
            supply_cols_idx = [i for i in range(self.nodes_table.columnCount()) if self.nodes_table.horizontalHeaderItem(i).text().startswith('supply') or self.nodes_table.horizontalHeaderItem(i).text() == 'Offre/Demande']
            for idxc in supply_cols_idx:
                supply_item = self.nodes_table.item(r, idxc)
                supply_text = supply_item.text().strip() if supply_item else ""
                if supply_text == "":
                    errors.append(f"Nœuds ligne {r+1} : Offre/Demande vide (col {idxc})")
                else:
                    try:
                        _ = float(supply_text)
                    except Exception:
                        if supply_text.lower() not in ("offre", "demande", "o", "d", "supply", "demand"):
                            errors.append(f"Nœuds ligne {r+1} : Offre/Demande non numérique ou non reconnue: '{supply_text}'")

        # Arcs table checks
        for r in range(self.arcs_table.rowCount()):
            # origin,dest must be present
            origin_item = self.arcs_table.item(r, 0)
            dest_item = self.arcs_table.item(r, 1)
            origin = origin_item.text().strip() if origin_item else ""
            dest = dest_item.text().strip() if dest_item else ""
            if not origin:
                errors.append(f"Arcs ligne {r+1} : Origine vide")
            if not dest:
                errors.append(f"Arcs ligne {r+1} : Destination vide")
            # numeric columns: cost, time, capacity, cap_cost and per-organ cost/time
            for c in range(self.arcs_table.columnCount()):
                colname = self.arcs_table.horizontalHeaderItem(c).text()
                it = self.arcs_table.item(r, c)
                txt = it.text().strip() if it else ""
                if txt == "":
                    errors.append(f"Arcs ligne {r+1} : {colname} vide")
                else:
                    # allow string columns for origin/dest only
                    colname_l = colname.lower()
                    if colname_l in ('origin','origine','dest','destination'):
                        continue
                    try:
                        _ = float(txt)
                    except Exception:
                        errors.append(f"Arcs ligne {r+1} : {colname} non numérique: '{txt}'")

        return (len(errors) == 0, errors)

    def remove_selected_row(self, table):
        """Remove the currently selected row(s) from given QTableWidget."""
        sel = table.selectionModel().selectedRows()
        if not sel:
            # try current row
            r = table.currentRow()
            if r >= 0:
                table.removeRow(r)
            return
        # remove rows in reverse order to keep indices valid
        rows = sorted([s.row() for s in sel], reverse=True)
        for r in rows:
            table.removeRow(r)

    def set_organ_types(self):
        """Parse organs_input and rebuild table columns accordingly."""
        txt = self.organs_input.text().strip()
        if txt:
            self.organs_list = [o.strip() for o in txt.split(',') if o.strip()]
        else:
            self.organs_list = []
        self.rebuild_tables_columns()

    def rebuild_tables_columns(self):
        """Rebuild nodes and arcs tables headers based on self.organs_list."""
        # Nodes: id,type,name + supplies
        if self.organs_list:
            headers = ['id','type','name'] + [f'supply_{o}' for o in self.organs_list]
        else:
            headers = ['id','type','name','supply']
        self.nodes_table.setColumnCount(len(headers))
        self.nodes_table.setHorizontalHeaderLabels(headers)
        # Arcs: origin,dest,cost,time,capacity,cap_cost + per organ cost/time
        base = ['origin','dest','cost','time','capacity','cap_cost']
        if self.organs_list:
            extras = []
            extras += [f'cost_{o}' for o in self.organs_list]
            extras += [f'time_{o}' for o in self.organs_list]
            headers_arcs = base + extras
        else:
            headers_arcs = base
        self.arcs_table.setColumnCount(len(headers_arcs))
        self.arcs_table.setHorizontalHeaderLabels(headers_arcs)

    def load_csvs(self):
        """Load CSV files from data/ and populate tables and params."""
        # nodes
        nodes_path = os.path.join(self.project_root, 'data', 'nodes.csv')
        arcs_path = os.path.join(self.project_root, 'data', 'arcs.csv')
        params_path = os.path.join(self.project_root, 'data', 'params.csv')
        if os.path.exists(nodes_path):
            try:
                # avoid triggering validation while populating
                self.nodes_table.blockSignals(True)
                self.arcs_table.blockSignals(True)
                rows = []
                try:
                    with open(nodes_path, newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                except UnicodeDecodeError:
                    with open(nodes_path, newline='', encoding='latin-1') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)

                # detect supply columns
                header = reader.fieldnames if reader.fieldnames else []
                supply_cols = [c for c in header if c.startswith('supply_')]
                if supply_cols:
                    self.organs_list = [c.replace('supply_','') for c in supply_cols]
                else:
                    self.organs_list = []
                self.rebuild_tables_columns()
                self.nodes_table.setRowCount(0)
                for r, row in enumerate(rows):
                    self.nodes_table.insertRow(r)
                    for c in range(self.nodes_table.columnCount()):
                        hdr_item = self.nodes_table.horizontalHeaderItem(c)
                        col = hdr_item.text() if hdr_item else ''
                        self.nodes_table.setItem(r, c, QTableWidgetItem(str(row.get(col, ''))))
            except Exception as e:
                print('Erreur chargement nodes.csv:', e)
            finally:
                self.nodes_table.blockSignals(False)
        if os.path.exists(arcs_path):
            try:
                rows = []
                try:
                    with open(arcs_path, newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                except UnicodeDecodeError:
                    with open(arcs_path, newline='', encoding='latin-1') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                header = reader.fieldnames if reader.fieldnames else []
                # infer organs from 'cost_<org>' and 'time_<org>' if not already set
                if not self.organs_list:
                    cost_orgs = [c.replace('cost_','') for c in header if c.startswith('cost_')]
                    time_orgs = [c.replace('time_','') for c in header if c.startswith('time_')]
                    if cost_orgs or time_orgs:
                        self.organs_list = sorted(set(cost_orgs + time_orgs))
                self.rebuild_tables_columns()
                self.arcs_table.setRowCount(0)
                for r, row in enumerate(rows):
                    self.arcs_table.insertRow(r)
                    for c in range(self.arcs_table.columnCount()):
                        hdr_item = self.arcs_table.horizontalHeaderItem(c)
                        col = hdr_item.text() if hdr_item else ''
                        self.arcs_table.setItem(r, c, QTableWidgetItem(str(row.get(col, ''))))
            except Exception as e:
                print('Erreur chargement arcs.csv:', e)
            finally:
                self.arcs_table.blockSignals(False)
        if os.path.exists(params_path):
            try:
                try:
                    with open(params_path, newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 2:
                                key = row[0].strip().lower()
                                val = row[1].strip()
                                try:
                                    v = float(val)
                                except Exception:
                                    continue
                                if key == 'alpha':
                                    self.alpha.setValue(v)
                                elif key == 'beta':
                                    self.beta.setValue(v)
                                elif key == 'vehicle_capacity':
                                    self.vehicle_capacity.setValue(v)
                except UnicodeDecodeError:
                    with open(params_path, newline='', encoding='latin-1') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 2:
                                key = row[0].strip().lower()
                                val = row[1].strip()
                                try:
                                    v = float(val)
                                except Exception:
                                    continue
                                if key == 'alpha':
                                    self.alpha.setValue(v)
                                elif key == 'beta':
                                    self.beta.setValue(v)
                                elif key == 'vehicle_capacity':
                                    self.vehicle_capacity.setValue(v)
            except Exception as e:
                print('Erreur chargement params.csv:', e)
        # re-read for organs name (string value)
        if os.path.exists(params_path):
            try:
                try:
                    with open(params_path, newline='', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 2 and row[0].strip().lower() == 'organs':
                                organs_val = row[1].strip()
                                self.organs_input.setText(organs_val)
                                if organs_val:
                                    self.organs_list = [o.strip() for o in organs_val.split(',') if o.strip()]
                                    self.rebuild_tables_columns()
                                break
                except UnicodeDecodeError:
                    with open(params_path, newline='', encoding='latin-1') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 2 and row[0].strip().lower() == 'organs':
                                organs_val = row[1].strip()
                                self.organs_input.setText(organs_val)
                                if organs_val:
                                    self.organs_list = [o.strip() for o in organs_val.split(',') if o.strip()]
                                    self.rebuild_tables_columns()
                                break
            except Exception as e:
                print('Erreur lecture organs dans params.csv:', e)
        # set status message
        self.status.showMessage('CSV chargés')

    def start_polling_solver_status(self):
        if not hasattr(self, '_solver_timer'):
            self._solver_timer = QTimer(self)
            self._solver_timer.setInterval(1000)
            self._solver_timer.timeout.connect(self._check_solver_status)
        self._solver_timer.start()

    def _check_solver_status(self):
        status_file = os.path.join(self.project_root, 'data', 'solver_status.txt')
        if not os.path.exists(status_file):
            return
        try:
            with open(status_file, 'r', encoding='utf-8') as sf:
                s = sf.read().strip()
        except Exception:
            return
        if s.lower() == 'running':
            self.status.showMessage('Solveur : en cours...')
            return
        # solver finished; present a dialog and update status
        if s.upper() == 'OPTIMAL':
            self.status.showMessage('Solveur : OPTIMAL')
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, 'Résultat solveur', 'Solution optimale trouvée.')
        elif s.upper() == 'INFEASIBLE':
            self.status.showMessage('Solveur : INFEASIBLE')
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Résultat solveur', 'Le modèle est INFEASIBLE.')
        elif s.upper() == 'UNBOUNDED':
            self.status.showMessage('Solveur : UNBOUNDED')
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Résultat solveur', 'Le modèle est UNBOUNDED.')
        elif s.upper() in ('FEASIBLE','DONE','DONE:OK'):
            self.status.showMessage('Solveur : solution trouvée (voire suboptimale)')
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, 'Résultat solveur', 'Le solveur a trouvé une solution.')
        else:
            self.status.showMessage(f'Solveur : {s}')
        # stop timer
        if hasattr(self, '_solver_timer'):
            self._solver_timer.stop()
        # re-enable editing once solver finished
        self.nodes_table.setEnabled(True)
        self.arcs_table.setEnabled(True)

    # -----------------------------
    # Immediate cell validation
    # -----------------------------
    def on_nodes_item_changed(self, item):
        # column 1 => Type must be non-numeric string
        if item is None:
            return
        col = item.column()
        txt = str(item.text()).strip()
        # helper
        def is_float(s):
            try:
                float(str(s).replace(',','.'))
                return True
            except Exception:
                return False

        if col == 1:
            # Type must be one of the expected types and not numeric
            allowed_types = ('source', 'destination', 'intermediaire')
            if txt == '':
                return
            if is_float(txt):
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, 'Erreur de saisie', "Le champ 'Type' ne doit pas être un nombre. Utilisez : source, destination ou intermediaire.")
                # highlight and clear
                self.nodes_table.blockSignals(True)
                item.setText('')
                item.setBackground(QBrush(QColor(255,200,200)))
                self.nodes_table.blockSignals(False)
                self.nodes_table.setCurrentCell(item.row(), col)
                self.nodes_table.editItem(item)
                return
            if txt.lower() not in allowed_types:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, 'Erreur de saisie', "Le champ 'Type' doit être l'un des suivants : source, destination, intermediaire (insensible à la casse).")
                # highlight and clear
                self.nodes_table.blockSignals(True)
                item.setText('')
                item.setBackground(QBrush(QColor(255,200,200)))
                self.nodes_table.blockSignals(False)
                self.nodes_table.setCurrentCell(item.row(), col)
                self.nodes_table.editItem(item)
                return
            else:
                # restore background
                item.setBackground(QBrush(QColor('white')))

        # If editing a supply column (may be dynamic index)
        header = self.nodes_table.horizontalHeaderItem(col).text() if self.nodes_table.horizontalHeaderItem(col) else ''
        if header.startswith('supply') or header == 'Offre/Demande':
            # Supply must be numeric or one of recognized labels
            allowed = ('offre','demande','o','d','supply','demand')
            if txt == '':
                return
            if is_float(txt):
                item.setBackground(QBrush(QColor('white')))
            elif txt.lower() in allowed:
                item.setBackground(QBrush(QColor('white')))
            else:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, 'Erreur de saisie', 'Le champ Offre/Demande doit être numérique ou "Offre"/"Demande". Veuillez corriger.')
                self.nodes_table.blockSignals(True)
                item.setText('')
                item.setBackground(QBrush(QColor(255,200,200)))
                self.nodes_table.blockSignals(False)
                self.nodes_table.setCurrentCell(item.row(), col)
                self.nodes_table.editItem(item)
                return

    def on_arcs_item_changed(self, item):
        if item is None:
            return
        col = item.column()
        txt = str(item.text()).strip()
        # numeric columns: cost,time,capacity,cap_cost or cost_<org>,time_<org>
        h = self.arcs_table.horizontalHeaderItem(col).text() if self.arcs_table.horizontalHeaderItem(col) else ''
        h_l = h.lower()
        if h_l not in ('origin','origine','dest','destination'):
            try:
                float(txt.replace(',','.'))
                item.setBackground(QBrush(QColor('white')))
            except Exception:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, 'Erreur de saisie', 'Ce champ doit être numérique (ex: 10.5). Veuillez corriger.')
                self.arcs_table.blockSignals(True)
                item.setText('')
                item.setBackground(QBrush(QColor(255,200,200)))
                self.arcs_table.blockSignals(False)
                self.arcs_table.setCurrentCell(item.row(), col)
                self.arcs_table.editItem(item)
                return
        else:
            # origin/dest must not be empty
            if txt == '':
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, 'Erreur de saisie', 'Les champs Origine / Destination ne peuvent pas être vides.')
                self.arcs_table.blockSignals(True)
                item.setBackground(QBrush(QColor(255,200,200)))
                self.arcs_table.blockSignals(False)
                self.arcs_table.setCurrentCell(item.row(), col)
                self.arcs_table.editItem(item)
                return


if __name__ == "__main__":
    # Create the QApplication and show the main window when the script is run
    try:
        app = QApplication(sys.argv)
        main_win = MainWindow()
        main_win.show()
        sys.exit(app.exec())
    except Exception as e:
        # Print errors (e.g., PyQt6 not installed) and exit
        print("Error launching GUI:", e)
        
        raise
