import sys
import random
import gurobipy as gp
from gurobipy import GRB
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QTabWidget, QInputDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QDialog, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsTextItem,
    QLabel, QScrollArea, QGroupBox, QFileDialog, QHeaderView, QTextEdit, QTabBar
)
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtGui import QPen, QColor, QBrush, QFont, QPainter, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF, QThread, pyqtSignal, QObject, QRectF, QRect
from PyQt5.QtWidgets import QProgressDialog
from pathlib import Path
from datetime import datetime
import pandas as pd
from openpyxl import load_workbook
import traceback
import re
from collections import defaultdict
import time

EXCEL_FILE = "tuteur_apprenant.xlsx"

# ---------- Normalization helpers (moved to module level) ----------
DAYS_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

def normalize_day_list(raw_list):
    result = set()
    for entry in raw_list:
        if not isinstance(entry, str):
            continue
        s = entry.strip().lower()
        # replace common abbreviations only as whole words
        abbrev_map = {
            'mon': 'monday', 'tue': 'tuesday', 'wed': 'wednesday',
            'thu': 'thursday', 'fri': 'friday', 'sat': 'saturday', 'sun': 'sunday'
        }
        for abbr, full in abbrev_map.items():
            s = re.sub(rf"\b{abbr}\b", full, s)
        # split on commas
        parts = [p.strip() for p in re.split('[,;]', s) if p.strip()]
        for p in parts:
            # range like monday-sunday
            if '-' in p:
                bounds = [b.strip() for b in p.split('-') if b.strip()]
                if len(bounds) >= 2 and bounds[0] in DAYS_ORDER and bounds[-1] in DAYS_ORDER:
                    i = DAYS_ORDER.index(bounds[0])
                    j = DAYS_ORDER.index(bounds[-1])
                    if i <= j:
                        for k in range(i, j+1):
                            result.add(DAYS_ORDER[k])
                    else:
                        # wrap-around (e.g., fri-mon)
                        for k in list(range(i, len(DAYS_ORDER))) + list(range(0, j+1)):
                            result.add(DAYS_ORDER[k])
                else:
                    # fallback: add whole token
                    result.add(p)
            else:
                # single day
                result.add(p)
    return set([d for d in result if d])

def normalize_hour_list(raw_list):
    # convert entries like '9-10', '9-10-11' -> {'9-10','10-11'}
    slots = set()
    for entry in raw_list:
        if not isinstance(entry, str):
            continue
        s = entry.strip()
        # split by comma or semicolon
        parts = [p.strip() for p in re.split('[,;]', s) if p.strip()]
        for p in parts:
            # extract numbers
            nums = re.findall(r"\d+", p)
            nums = [int(n) for n in nums]
            if len(nums) == 2:
                slots.add(f"{nums[0]}-{nums[1]}")
            elif len(nums) > 2:
                # treat as consecutive ranges: 9,10,11 -> 9-10,10-11
                for idx in range(len(nums)-1):
                    slots.add(f"{nums[idx]}-{nums[idx+1]}")
            else:
                # unknown format, add raw token
                slots.add(p)
    return set(slots)


def normalize_domain(s):
    """Normalize domain strings to a canonical form.

    Treat 'CSS', 'css', 'Css' as the same by converting to lower and
    returning Title case (first letter upper) for display/storage.
    Returns empty string for None/invalid input.
    """
    if not s:
        return ""
    try:
        ss = str(s).strip()
        if not ss:
            return ""
        # unify: remove extra whitespace, collapse internal spaces, lower then title
        ss = re.sub(r"\s+", " ", ss)
        return ss.lower().title()
    except Exception:
        return str(s)


# ---------- Analysis helpers for advanced diagnostics ----------

def analyze_coverage(tuteurs, apprenants, solution):
    """Analyse la couverture : tuteurs et apprenants couplés"""
    tuteurs_coupled = set(t for t in solution if solution[t])
    apprenants_coupled = set()
    for t in solution:
        for (a, _) in solution[t]:
            apprenants_coupled.add(a)
    
    total_couplings = sum(len(solution[t]) for t in solution)
    
    coverage = {
        "total_tuteurs": len(tuteurs),
        "tuteurs_couples": len(tuteurs_coupled),
        "taux_tuteurs": (len(tuteurs_coupled) / len(tuteurs) * 100) if tuteurs else 0,
        "tuteurs_non_couples": sorted(set(tuteurs.keys()) - tuteurs_coupled),
        
        "total_apprenants": len(apprenants),
        "apprenants_couples": len(apprenants_coupled),
        "taux_apprenants": (len(apprenants_coupled) / len(apprenants) * 100) if apprenants else 0,
        "apprenants_non_couples": sorted(set(apprenants.keys()) - apprenants_coupled),
        
        "total_couplings": total_couplings
    }
    return coverage


def detect_bottlenecks(tuteurs, apprenants, selected_tuteurs=None, selected_apprenants=None):
    """Détecte les goulets d'étranglement (domaines/jours/heures rares)"""
    
    if selected_tuteurs:
        T = [t for t in selected_tuteurs if t in tuteurs]
    else:
        T = list(tuteurs.keys())
    
    if selected_apprenants:
        A = [a for a in selected_apprenants if a in apprenants]
    else:
        A = list(apprenants.keys())
    
    # Count by domain
    domain_count_t = defaultdict(int)
    domain_count_a = defaultdict(int)
    for t in T:
        domain_count_t[tuteurs[t]["domain"].lower()] += 1
    for a in A:
        domain_count_a[apprenants[a]["domain"].lower()] += 1
    
    # Count by day
    day_count_t = defaultdict(int)
    day_count_a = defaultdict(int)
    for t in T:
        days = normalize_day_list(tuteurs[t]["days"])
        for d in days:
            day_count_t[d] += 1
    for a in A:
        days = normalize_day_list(apprenants[a]["days"])
        for d in days:
            day_count_a[d] += 1
    
    # Count by hour
    hour_count_t = defaultdict(int)
    hour_count_a = defaultdict(int)
    for t in T:
        hours = normalize_hour_list(tuteurs[t]["hours"])
        for h in hours:
            hour_count_t[h] += 1
    for a in A:
        hours = normalize_hour_list(apprenants[a]["hours"])
        for h in hours:
            hour_count_a[h] += 1
    
    bottlenecks = {
        "domains": {
            "tuteurs": dict(domain_count_t),
            "apprenants": dict(domain_count_a),
            "imbalance": max(domain_count_t.values()) - min(domain_count_t.values()) if domain_count_t else 0
        },
        "days": {
            "tuteurs": dict(day_count_t),
            "apprenants": dict(day_count_a),
            "rarest_day_t": min(day_count_t, key=day_count_t.get) if day_count_t else None,
            "rarest_day_a": min(day_count_a, key=day_count_a.get) if day_count_a else None
        },
        "hours": {
            "tuteurs": dict(hour_count_t),
            "apprenants": dict(hour_count_a),
            "rarest_hour_t": min(hour_count_t, key=hour_count_t.get) if hour_count_t else None,
            "rarest_hour_a": min(hour_count_a, key=hour_count_a.get) if hour_count_a else None
        }
    }
    return bottlenecks


# ---------------- Données par défaut ----------------
DEFAULT_TUTEURS = {
    "T1": {"domain": "Math", "days": ["Mon", "Wed"], "hours": ["9-10", "14-15"]},
    "T2": {"domain": "Physics", "days": ["Tue", "Thu"], "hours": ["10-11", "15-16"]},
    "T3": {"domain": "CS", "days": ["Mon", "Fri"], "hours": ["9-10", "13-14"]},
    "T4": {"domain": "Math", "days": ["Tue", "Thu"], "hours": ["10-11", "14-15"]}
}

DEFAULT_APPRENANTS = {
    "A1": {"domain": "Math", "days": ["Mon", "Wed"], "hours": ["9-10", "14-15"]},
    "A2": {"domain": "Physics", "days": ["Tue"], "hours": ["10-11"]},
    "A3": {"domain": "CS", "days": ["Fri"], "hours": ["13-14"]},
    "A4": {"domain": "Math", "days": ["Wed"], "hours": ["14-15"]},
    "A5": {"domain": "Physics", "days": ["Tue", "Thu"], "hours": ["15-16"]},
    "A6": {"domain": "CS", "days": ["Mon", "Fri"], "hours": ["9-10"]}
}

# ---------------- Excel functions ----------------
def save_to_excel(tuteurs, apprenants, filename=EXCEL_FILE):
    df_tut = pd.DataFrame([{
        "Tuteur": k,
        "Domain": v.get("domain", ""),
        "Days": ",".join(v.get("days", [])),
        "Hours": ",".join(v.get("hours", []))
    } for k, v in tuteurs.items()])
    df_app = pd.DataFrame([{
        "Apprenant": k,
        "Domain": v.get("domain", ""),
        "Days": ",".join(v.get("days", [])),
        "Hours": ",".join(v.get("hours", []))
    } for k, v in apprenants.items()])
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_tut.to_excel(writer, sheet_name="Tuteurs", index=False)
        df_app.to_excel(writer, sheet_name="Apprenants", index=False)

def load_from_excel(filename=EXCEL_FILE):
    if not Path(filename).exists():
        save_to_excel(DEFAULT_TUTEURS, DEFAULT_APPRENANTS)
        return DEFAULT_TUTEURS.copy(), DEFAULT_APPRENANTS.copy()
    xls = pd.ExcelFile(filename)
    df_tut = pd.read_excel(xls, sheet_name="Tuteurs")
    df_app = pd.read_excel(xls, sheet_name="Apprenants")
    tuteurs = {}
    for row in df_tut.values:
        if pd.notna(row[0]):  # Vérifier si le nom n'est pas NaN
            tuteurs[str(row[0])] = {
                "domain": normalize_domain(row[1]) if pd.notna(row[1]) else "",
                "days": str(row[2]).split(",") if pd.notna(row[2]) else [],
                "hours": str(row[3]).split(",") if pd.notna(row[3]) else []
            }
    
    apprenants = {}
    for row in df_app.values:
        if pd.notna(row[0]):  # Vérifier si le nom n'est pas NaN
            apprenants[str(row[0])] = {
                "domain": normalize_domain(row[1]) if pd.notna(row[1]) else "",
                "days": str(row[2]).split(",") if pd.notna(row[2]) else [],
                "hours": str(row[3]).split(",") if pd.notna(row[3]) else []
            }
    # If file is missing some defaults (corrupted or partial), merge missing default entries
    for k, v in DEFAULT_TUTEURS.items():
        if k not in tuteurs:
            t = v.copy()
            t["domain"] = normalize_domain(t.get("domain", ""))
            tuteurs[k] = t
    for k, v in DEFAULT_APPRENANTS.items():
        if k not in apprenants:
            a = v.copy()
            a["domain"] = normalize_domain(a.get("domain", ""))
            apprenants[k] = a

    return tuteurs, apprenants

def save_solution_to_excel(solution, filename=EXCEL_FILE):
    rows = []
    for t, app_list in solution.items():
        for a, info in app_list:
            rows.append({
                "Tuteur": t,
                "Apprenant": a,
                "Domain": info.get("domain", ""),
                "Days": ",".join(info.get("days", [])),
                "Hours": ",".join(info.get("hours", []))
            })
    df_solution = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y-%m-d_%H-%M-%S")
    try:
        with pd.ExcelWriter(filename, engine="openpyxl", mode='a', if_sheet_exists="replace") as writer:
            df_solution.to_excel(writer, sheet_name=f"Solution_{timestamp}", index=False)
    except:
        # Si le fichier n'existe pas, créer un nouveau
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_solution.to_excel(writer, sheet_name=f"Solution_{timestamp}", index=False)

# ---------------- Optimisation ----------------
def optimize_couplage(tuteurs, apprenants, selected_tuteurs=None, selected_apprenants=None, model_holder=None, time_limit=None):
    """
    Optimise le couplage entre tuteurs et apprenants
    """
    # Utiliser les sélections si spécifiées, sinon tous
    if selected_tuteurs:
        T = [t for t in selected_tuteurs if t in tuteurs]
    else:
        T = list(tuteurs.keys())
    
    if selected_apprenants:
        A = [a for a in selected_apprenants if a in apprenants]
    else:
        A = list(apprenants.keys())
    
    if not T or not A:
        return {}

    # (no debug prints)

    # Use module-level normalization helpers: `normalize_day_list`, `normalize_hour_list`
    
    try:
        model = gp.Model("Couplage")
        # expose model to caller if requested (allows abort)
        if isinstance(model_holder, dict):
            model_holder['model'] = model
        # set time limit if provided
        if time_limit is not None:
            try:
                model.setParam('TimeLimit', float(time_limit))
            except Exception:
                pass
        x = {}
        
        # Création de toutes les variables
        for i in T:
            for j in A:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        
        model.update()
        
        # Contrainte 1: Chaque tuteur peut être couplé avec au plus 1 apprenant
        for i in T:
            model.addConstr(gp.quicksum(x[i, j] for j in A) <= 1, name=f"cap_tuteur_{i}")
        
        # Contrainte 2: Chaque apprenant peut être couplé avec au plus 1 tuteur
        for j in A:
            model.addConstr(gp.quicksum(x[i, j] for i in T) <= 1, name=f"cap_apprenant_{j}")
        
        # Contraintes de compatibilité
        incompat_count = 0
        for i in T:
            for j in A:
                # Vérifier la compatibilité
                domain_match = tuteurs[i]["domain"].strip().lower() == apprenants[j]["domain"].strip().lower()

                # Nettoyer les jours et heures
                raw_t_days = tuteurs[i]["days"]
                raw_a_days = apprenants[j]["days"]
                raw_t_hours = tuteurs[i]["hours"]
                raw_a_hours = apprenants[j]["hours"]
                # (no debug prints of raw lists)

                # Normalize days and hours for robust matching
                t_days = normalize_day_list(raw_t_days)
                a_days = normalize_day_list(raw_a_days)
                t_hours = normalize_hour_list(raw_t_hours)
                a_hours = normalize_hour_list(raw_a_hours)

                days_common = t_days.intersection(a_days)
                hours_common = t_hours.intersection(a_hours)
                # (no debug prints of compatibility)

                # Si incompatible, forcer la variable à 0
                if not (domain_match and days_common and hours_common):
                    model.addConstr(x[i, j] == 0, name=f"incompat_{i}_{j}")
                    incompat_count += 1

        # (no debug prints of incompatibility counts)
        
        # Objectif : maximiser le nombre de couplages
        model.setObjective(gp.quicksum(x[i, j] for i in T for j in A), GRB.MAXIMIZE)
        
        # Résolution
        model.setParam('OutputFlag', 0)  # Désactiver la sortie console
        model.optimize()
        
        solution = {}
        
        if model.status == GRB.OPTIMAL:
            for i in T:
                solution[i] = []
                for j in A:
                    if x[i, j].X > 0.5:  # Si le couplage est sélectionné
                        # Use normalized day/hour lists for display so table shows common slotsf
                        try:
                            norm_t_days = normalize_day_list(tuteurs[i]["days"])
                            norm_a_days = normalize_day_list(apprenants[j]["days"])
                            days_common = sorted(list(norm_t_days.intersection(norm_a_days)))
                        except Exception:
                            days_common = list(set([d.strip() for d in tuteurs[i]["days"]])
                                               .intersection([d.strip() for d in apprenants[j]["days"]]))
                        try:
                            norm_t_hours = normalize_hour_list(tuteurs[i]["hours"])
                            norm_a_hours = normalize_hour_list(apprenants[j]["hours"])
                            hours_common = sorted(list(norm_t_hours.intersection(norm_a_hours)))
                        except Exception:
                            hours_common = list(set([h.strip() for h in tuteurs[i]["hours"]])
                                                .intersection([h.strip() for h in apprenants[j]["hours"]]))

                        solution[i].append((j, {
                            "domain": tuteurs[i]["domain"],
                            "days": days_common,
                            "hours": hours_common
                        }))
        
        # (no debug prints of solver status or variable values)

        return solution

    except gp.GurobiError as e:
        # Ne pas appeler de dialogues GUI depuis un thread de calcul; remonter l'erreur
        print(f"Erreur Gurobi: {e}")
        raise
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        raise


# ---------------- Background worker for optimization ----------------
class OptimizeWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, tuteurs, apprenants, selected_tuteurs, selected_apprenants, model_holder=None, time_limit=None):
        super().__init__()
        self.tuteurs = tuteurs
        self.apprenants = apprenants
        self.selected_tuteurs = selected_tuteurs
        self.selected_apprenants = selected_apprenants
        self.model_holder = model_holder
        self.time_limit = time_limit

    def run(self):
        try:
            result = optimize_couplage(
                self.tuteurs,
                self.apprenants,
                self.selected_tuteurs,
                self.selected_apprenants,
                model_holder=self.model_holder,
                time_limit=self.time_limit
            )
            self.finished.emit(result if result is not None else {})
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)

    def abort(self):
        # Try to abort the running Gurobi model if available
        try:
            if isinstance(self.model_holder, dict) and 'model' in self.model_holder and self.model_holder['model'] is not None:
                try:
                    self.model_holder['model'].abort()
                except Exception:
                    pass
        except Exception:
            pass


# ---------------- Multi-input Dialog ----------------
class MultiInputDialog(QDialog):
    def __init__(self, parent, title, default_domain="", default_days=None, default_hours=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 200)
        
        self.domain = default_domain
        self.days = default_days if default_days else []
        self.hours = default_hours if default_hours else []
        
        layout = QVBoxLayout()
        
        # Domain
        domain_layout = QHBoxLayout()
        domain_layout.addWidget(QLabel("Domaine:"))
        self.domain_input = QInputDialog()
        domain_input_text = QInputDialog()
        domain, ok1 = QInputDialog.getText(self, "Domaine", "Entrez le domaine:", text=self.domain)
        if ok1 and domain.strip():
            self.domain = domain.strip()
        
        # Days
        days_layout = QHBoxLayout()
        days_str, ok2 = QInputDialog.getText(self, "Jours", "Jours disponibles (séparés par des virgules, ex: Mon,Wed,Fri):",
                                            text=",".join(self.days))
        if ok2:
            self.days = [d.strip() for d in days_str.split(",") if d.strip()]
        
        # Hours
        hours_layout = QHBoxLayout()
        hours_str, ok3 = QInputDialog.getText(self, "Heures", "Heures disponibles (séparées par des virgules, ex: 9-10,14-15):",
                                             text=",".join(self.hours))
        if ok3:
            self.hours = [h.strip() for h in hours_str.split(",") if h.strip()]
        
        self.setLayout(layout)

    def get_values(self):
        return self.domain, self.days, self.hours

# ---------------- GUI ----------------
class CouplageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Couplage Tuteur/Apprenant")
        self.resize(1200, 800)
        self.tuteurs, self.apprenants = load_from_excel()
        self.solution = None
        self.selected_tuteurs = []
        self.selected_apprenants = []
        self.verbose = False
        self._line_items = {}  # mapping (t,a) -> QGraphicsLineItem
        self._ellipse_items = {"tuteurs": {}, "apprenants": {}}
        self._opt_worker = None
        self._opt_thread = None
        self._model_holder = None
        self.time_limit = None
        self.coverage = None  # for analysis
        self.bottlenecks = None  # for analysis
        self.solve_time = 0
        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Apprenants Tab ---
        self.app_tab = QWidget()
        app_layout = QVBoxLayout()
        self.app_tab.setLayout(app_layout)
        self.tabs.addTab(self.app_tab, "Étape 1: Apprenants")
        
        # Titre
        title_label = QLabel("Sélectionnez les apprenants:")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; color: #2196F3;")
        app_layout.addWidget(title_label)
        
        # Zone de défilement pour les apprenants
        scroll_app = QScrollArea()
        scroll_app.setWidgetResizable(True)
        scroll_widget_app = QWidget()
        self.app_checkbox_layout = QVBoxLayout(scroll_widget_app)
        self.app_checkbox_layout.setAlignment(Qt.AlignTop)
        scroll_app.setWidget(scroll_widget_app)
        app_layout.addWidget(scroll_app)
        
        self.app_checkboxes = {}
        self.refresh_apprenant_checkboxes()
        
        # Boutons CRUD
        btn_layout = QHBoxLayout()
        for text, func, color in [("Ajouter", self.add_apprenant, "#4CAF50"),
                                  ("Modifier", self.edit_apprenant, "#FFC107"),
                                  ("Supprimer", self.delete_apprenant, "#F44336")]:
            b = QPushButton(text)
            b.clicked.connect(func)
            b.setStyleSheet(f"background-color:{color}; color:white; font-weight:bold; padding:8px; border-radius:5px;")
            btn_layout.addWidget(b)
        import_btn = QPushButton("📥 Importer Excel")
        import_btn.setStyleSheet("background-color:#2196F3; color:white; padding:8px; border-radius:5px;")
        import_btn.clicked.connect(self.import_from_excel)
        btn_layout.addWidget(import_btn)
        app_layout.addLayout(btn_layout)
        
        # Next
        next_btn = QPushButton("Suivant: Tuteurs")
        next_btn.setStyleSheet("background-color:#2196F3; color:white; font-weight:bold; padding:8px; border-radius:5px;")
        next_btn.clicked.connect(self.goto_tuteur_tab)
        app_layout.addWidget(next_btn)

        # --- Tuteurs Tab ---
        self.tut_tab = QWidget()
        tut_layout = QVBoxLayout()
        self.tut_tab.setLayout(tut_layout)
        self.tabs.addTab(self.tut_tab, "Étape 2: Tuteurs")
        
        # Titre
        title_label2 = QLabel("Sélectionnez les tuteurs:")
        title_label2.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; color: #2196F3;")
        tut_layout.addWidget(title_label2)
        
        # Zone de défilement pour les tuteurs
        scroll_tut = QScrollArea()
        scroll_tut.setWidgetResizable(True)
        scroll_widget_tut = QWidget()
        self.tut_checkbox_layout = QVBoxLayout(scroll_widget_tut)
        self.tut_checkbox_layout.setAlignment(Qt.AlignTop)
        scroll_tut.setWidget(scroll_widget_tut)
        tut_layout.addWidget(scroll_tut)
        
        self.tut_checkboxes = {}
        self.refresh_tuteur_checkboxes()
        
        # Boutons CRUD
        btn_layout2 = QHBoxLayout()
        for text, func, color in [("Ajouter", self.add_tuteur, "#4CAF50"),
                                  ("Modifier", self.edit_tuteur, "#FFC107"),
                                  ("Supprimer", self.delete_tuteur, "#F44336")]:
            b = QPushButton(text)
            b.clicked.connect(func)
            b.setStyleSheet(f"background-color:{color}; color:white; font-weight:bold; padding:8px; border-radius:5px;")
            btn_layout2.addWidget(b)
        import_btn2 = QPushButton("📥 Importer Excel")
        import_btn2.setStyleSheet("background-color:#2196F3; color:white; padding:8px; border-radius:5px;")
        import_btn2.clicked.connect(self.import_from_excel)
        btn_layout2.addWidget(import_btn2)
        tut_layout.addLayout(btn_layout2)
        
        # Next
        next_btn2 = QPushButton("Suivant: Optimisation")
        next_btn2.setStyleSheet("background-color:#2196F3; color:white; font-weight:bold; padding:8px; border-radius:5px;")
        next_btn2.clicked.connect(self.goto_opt_tab)
        tut_layout.addWidget(next_btn2)

        # --- Optimisation Tab ---
        self.opt_tab = QWidget()
        opt_layout = QVBoxLayout()
        self.opt_tab.setLayout(opt_layout)
        self.tabs.addTab(self.opt_tab, "Étape 3: Optimisation")
        
        # Titre
        title_label3 = QLabel("Optimisation des couplages")
        title_label3.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; color: #9C27B0;")
        opt_layout.addWidget(title_label3)
        
        # Info sur la sélection
        self.selection_info = QLabel("")
        self.selection_info.setStyleSheet("padding: 5px; color: #666;")
        opt_layout.addWidget(self.selection_info)
        
        # Boutons
        opt_btn_layout = QHBoxLayout()
        b1 = QPushButton("🚀 Exécuter l'optimisation")
        b1.clicked.connect(self.run_optimization)
        b1.setStyleSheet("background-color:#9C27B0; color:white; font-weight:bold; padding:10px; border-radius:5px; font-size: 12px;")

        b2 = QPushButton("💾 Sauvegarder la solution")
        b2.clicked.connect(self.save_solution)
        b2.setStyleSheet("background-color:#607D8B; color:white; font-weight:bold; padding:10px; border-radius:5px; font-size: 12px;")

        b_clear = QPushButton("🧹 Effacer résultats")
        b_clear.clicked.connect(self.clear_results)
        b_clear.setStyleSheet("background-color:#9E9E9E; color:white; padding:8px; border-radius:5px;")

        b_cancel = QPushButton("✋ Annuler optimisation")
        b_cancel.clicked.connect(self.cancel_optimization)
        b_cancel.setStyleSheet("background-color:#F44336; color:white; padding:8px; border-radius:5px;")

        b_export = QPushButton("📷 Exporter PNG")
        b_export.clicked.connect(self.export_visualization)
        b_export.setStyleSheet("background-color:#4CAF50; color:white; padding:8px; border-radius:5px;")

        self.debug_checkbox = QCheckBox("Mode debug")
        self.debug_checkbox.stateChanged.connect(self.toggle_debug_mode)

        b_timelimit = QPushButton("⏱ TimeLimit")
        b_timelimit.clicked.connect(self.set_time_limit)
        b_timelimit.setStyleSheet("background-color:#607D8B; color:white; padding:8px; border-radius:5px;")

        opt_btn_layout.addWidget(b1)
        opt_btn_layout.addWidget(b2)
        opt_btn_layout.addWidget(b_clear)
        opt_btn_layout.addWidget(b_cancel)
        opt_btn_layout.addWidget(b_export)
        opt_btn_layout.addWidget(b_timelimit)
        opt_btn_layout.addWidget(self.debug_checkbox)
        opt_layout.addLayout(opt_btn_layout)

        
        # Table des résultats
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Tuteur", "Apprenant", "Domaine commun", "Jours communs", "Heures communes"])
        # Make all columns equal width
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(200)
        opt_layout.addWidget(self.table)
        # Connect table click for interaction (after table is created)
        self.table.cellClicked.connect(self.on_table_cell_clicked)
        
        # Visualisation 2D
        # Visualization (hidden by default)
        self.vis_label = QLabel("📊 Visualisation des couplages:")
        self.vis_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px; color: #2196F3;")
        self.vis_label.setVisible(False)
        opt_layout.addWidget(self.vis_label)

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.view.setFixedHeight(400)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setVisible(False)
        opt_layout.addWidget(self.view)

        # Button to toggle visualization visibility
        self.b_toggle_vis = QPushButton("Afficher la visualisation")
        self.b_toggle_vis.setStyleSheet("background-color:#03A9F4; color:white; padding:8px; border-radius:5px;")
        self.b_toggle_vis.clicked.connect(self.toggle_visualization)
        opt_btn_layout.addWidget(self.b_toggle_vis)

        # --- Rapport Tab ---
        self.rapport_tab = QWidget()
        rapport_layout = QVBoxLayout()
        self.rapport_tab.setLayout(rapport_layout)
        self.tabs.addTab(self.rapport_tab, "📋 Rapport")
        
        rapport_title = QLabel("Rapport d'Optimisation")
        rapport_title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; color: #FF9800;")
        rapport_layout.addWidget(rapport_title)
        
        self.rapport_text = QTextEdit()
        self.rapport_text.setReadOnly(True)
        self.rapport_text.setStyleSheet("font-family: monospace; font-size: 14px; padding: 10px;")
        rapport_layout.addWidget(self.rapport_text)
        
        b_refresh_rapport = QPushButton("🔄 Rafraîchir rapport")
        b_refresh_rapport.setStyleSheet("background-color:#FF9800; color:white; padding:8px; border-radius:5px;")
        b_refresh_rapport.clicked.connect(self.update_rapport)
        rapport_layout.addWidget(b_refresh_rapport)

        # --- Visualisation Avancée Tab ---
        self.vis_adv_tab = QWidget()
        vis_adv_layout = QVBoxLayout()
        self.vis_adv_tab.setLayout(vis_adv_layout)
        self.tabs.addTab(self.vis_adv_tab, "📊 Visualisation Avancée")
        
        vis_adv_title = QLabel("Heatmap & Analyse Graphique")
        vis_adv_title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; color: #4CAF50;")
        vis_adv_layout.addWidget(vis_adv_title)
        
        # Sub-tabs for different visualizations
        self.vis_subtabs = QTabWidget()

        vis_adv_layout.addWidget(self.vis_subtabs)
        
        # Heatmap Domains: SEULEMENT LA VISUALISATION GRAPHIQUE (tableau supprimé)
        self.heatmap_domain_widget = QWidget()
        hd_layout = QVBoxLayout(self.heatmap_domain_widget)

        # --- VIEW GRAPHIQUE SEULEMENT ---
        self.heatmap_domain_view = QGraphicsView()
        self.heatmap_domain_scene = QGraphicsScene()
        self.heatmap_domain_view.setScene(self.heatmap_domain_scene)
        self.heatmap_domain_view.setFixedHeight(400)  # Plus grand pour mieux voir
        self.heatmap_domain_view.setStyleSheet("background: white; border: 1px solid #ccc;")
        hd_layout.addWidget(self.heatmap_domain_view)

        self.vis_subtabs.addTab(self.heatmap_domain_widget, "Heatmap Domaines")
        
        # Heatmap Days
        self.heatmap_day_view = QGraphicsView()
        self.heatmap_day_scene = QGraphicsScene()
        self.heatmap_day_view.setScene(self.heatmap_day_scene)
        self.vis_subtabs.addTab(self.heatmap_day_view, "Heatmap Jours")
        
        # Heatmap Hours
        self.heatmap_hour_view = QGraphicsView()
        self.heatmap_hour_scene = QGraphicsScene()
        self.heatmap_hour_view.setScene(self.heatmap_hour_scene)
        self.vis_subtabs.addTab(self.heatmap_hour_view, "Heatmap Heures")
        
        # Bipartite graph
        self.bipartite_view = QGraphicsView()
        self.bipartite_scene = QGraphicsScene()
        self.bipartite_view.setScene(self.bipartite_scene)
        self.bipartite_view.setRenderHint(QPainter.Antialiasing)
        self.vis_subtabs.addTab(self.bipartite_view, "Graphe Bipartite")
        
        b_render_vis = QPushButton("🎨 Générer visualisations")
        b_render_vis.setStyleSheet("background-color:#4CAF50; color:white; padding:8px; border-radius:5px;")
        b_render_vis.clicked.connect(self.generate_advanced_visualizations)
        vis_adv_layout.addWidget(b_render_vis)

    # ---------------- Refresh checkboxes ----------------
    def refresh_apprenant_checkboxes(self):
        # Nettoyer le layout
        for i in reversed(range(self.app_checkbox_layout.count())):
            widget = self.app_checkbox_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        self.app_checkboxes.clear()
        
        if not self.apprenants:
            label = QLabel("Aucun apprenant disponible. Ajoutez-en un!")
            label.setStyleSheet("color: #666; padding: 10px;")
            self.app_checkbox_layout.addWidget(label)
            return
        
        # Ajouter un titre
        count_label = QLabel(f"Apprenants disponibles ({len(self.apprenants)}):")
        count_label.setStyleSheet("font-weight: bold; color: #333; padding: 5px;")
        self.app_checkbox_layout.addWidget(count_label)
        
        for a in sorted(self.apprenants.keys()):
            info = self.apprenants[a]
            cb = QCheckBox(f"{a} - Domaine: {info['domain']} | Jours: {', '.join(info['days'])} | Heures: {', '.join(info['hours'])}")
            cb.setChecked(True)  # Par défaut, tous sélectionnés
            cb.setStyleSheet("padding: 5px;")
            self.app_checkboxes[a] = cb
            self.app_checkbox_layout.addWidget(cb)
        
        

    def refresh_tuteur_checkboxes(self):
        # Nettoyer le layout
        for i in reversed(range(self.tut_checkbox_layout.count())):
            widget = self.tut_checkbox_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        self.tut_checkboxes.clear()
        
        if not self.tuteurs:
            label = QLabel("Aucun tuteur disponible. Ajoutez-en un!")
            label.setStyleSheet("color: #666; padding: 10px;")
            self.tut_checkbox_layout.addWidget(label)
            return
        
        # Ajouter un titre
        count_label = QLabel(f"Tuteurs disponibles ({len(self.tuteurs)}):")
        count_label.setStyleSheet("font-weight: bold; color: #333; padding: 5px;")
        self.tut_checkbox_layout.addWidget(count_label)
        
        for t in sorted(self.tuteurs.keys()):
            info = self.tuteurs[t]
            cb = QCheckBox(f"{t} - Domaine: {info['domain']} | Jours: {', '.join(info['days'])} | Heures: {', '.join(info['hours'])}")
            cb.setChecked(True)  # Par défaut, tous sélectionnés
            cb.setStyleSheet("padding: 5px;")
            self.tut_checkboxes[t] = cb
            self.tut_checkbox_layout.addWidget(cb)
        
      

    # ---------------- Navigation ----------------
    def goto_tuteur_tab(self):
        self.selected_apprenants = [a for a, cb in self.app_checkboxes.items() if cb.isChecked()]
        if not self.selected_apprenants:
            QMessageBox.warning(self, "Attention", "Sélectionnez au moins un apprenant")
            return
        self.tabs.setCurrentWidget(self.tut_tab)

    def goto_opt_tab(self):
        self.selected_tuteurs = [t for t, cb in self.tut_checkboxes.items() if cb.isChecked()]
        if not self.selected_tuteurs:
            QMessageBox.warning(self, "Attention", "Sélectionnez au moins un tuteur")
            return
        
        # Mettre à jour les informations de sélection
        self.selection_info.setText(f"Tuteurs sélectionnés: {len(self.selected_tuteurs)} | Apprenants sélectionnés: {len(self.selected_apprenants)}")
        self.tabs.setCurrentWidget(self.opt_tab)

    # ---------------- CRUD ----------------
    def add_apprenant(self):
        name, ok = QInputDialog.getText(self, "Ajouter un apprenant", "Entrez le nom de l'apprenant (ex: A5):")
        if not ok or not name.strip(): 
            return
        
        if name in self.apprenants:
            QMessageBox.warning(self, "Erreur", f"L'apprenant {name} existe déjà!")
            return
        
        # Contrôle du domaine : doit être une chaîne de caractères, pas des chiffres
        domain, ok1 = QInputDialog.getText(self, "Domaine", "Entrez le domaine (ex: Math, Physics, CS):")
        if not ok1 or not domain.strip():
            QMessageBox.warning(self, "Attention", "Le domaine est obligatoire")
            return
        
        # Vérifier que le domaine ne contient pas uniquement des chiffres
        if domain.strip().isdigit():
            QMessageBox.warning(self, "Erreur", "Le domaine ne doit pas être composé uniquement de chiffres")
            return
        
        # Vérifier que le domaine est bien une chaîne de caractères alphabétique
        # Autoriser les espaces, traits d'union, etc. mais pas que des chiffres
        if all(c.isdigit() for c in domain.strip() if c.isalnum()):
            QMessageBox.warning(self, "Erreur", "Le domaine doit contenir des lettres")
            return
        
        days_str, ok2 = QInputDialog.getText(self, "Jours", "Jours disponibles (séparés par des virgules, ex: Mon,Wed,Fri):")
        if not ok2:
            days_str = ""

        days = [d.strip() for d in days_str.split(",") if d.strip()]
        # Dictionnaire pour normaliser les jours
        day_mapping = {
            "mon": "Mon", "monday": "Monday",
            "tue": "Tue", "tuesday": "Tuesday",
            "wed": "Wed", "wednesday": "Wednesday",
            "thu": "Thu", "thursday": "Thursday",
            "fri": "Fri", "friday": "Friday",
            "sat": "Sat", "saturday": "Saturday",
            "sun": "Sun", "sunday": "Sunday",
            "lun": "Lun", "lundi": "Lundi",
            "mar": "Mar", "mardi": "Mardi",
            "mer": "Mer", "mercredi": "Mercredi",
            "jeu": "Jeu", "jeudi": "Jeudi",
            "ven": "Ven", "vendredi": "Vendredi",
            "sam": "Sam", "samedi": "Samedi",
            "dim": "Dim", "dimanche": "Dimanche"
        }

        # Validation et normalisation
        normalized_days = []
        for day in days:
            key = day.strip().lower()
            if key not in day_mapping:
                QMessageBox.warning(self, "Erreur", 
                                    f"Jour invalide: '{day}'. Jours acceptés:\n"
                                    f"Anglais: Mon, Tue, Wed, Thu, Fri, Sat, Sun\n"
                                    f"Français: Lun, Mar, Mer, Jeu, Ven, Sam, Dim")
                return
            normalized_days.append(day_mapping[key])

        # Ensuite, tu remplaces days par normalized_days
        days = normalized_days


        # Déplacer la validation juste après la saisie
        if days:
            valid_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
                        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                        "Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim",
                        "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

            for day in days:
                if day not in valid_days:
                    QMessageBox.warning(self, "Erreur", 
                                        f"Jour invalide: '{day}'. Jours acceptés:\n"
                                        f"Anglais: Mon, Tue, Wed, Thu, Fri, Sat, Sun\n"
                                        f"Français: Lun, Mar, Mer, Jeu, Ven, Sam, Dim")
                    return

        hours_str, ok3 = QInputDialog.getText(self, "Heures", "Heures disponibles (séparées par des virgules, ex: 9-10,14-15):")
        if not ok3:
            hours_str = ""
        hours = [h.strip() for h in hours_str.split(",") if h.strip()]
        # Validation des heures
        if hours:
            for hour_slot in hours:
                # Vérifier le format i-i+1 (ex: 9-10, 14-15)
                if "-" not in hour_slot:
                    QMessageBox.warning(self, "Erreur", 
                                    f"Format d'heure invalide: '{hour_slot}'. Format attendu: début-fin (ex: 9-10)")
                    return
                
                parts = hour_slot.split("-")
                if len(parts) != 2:
                    QMessageBox.warning(self, "Erreur", 
                                    f"Format d'heure invalide: '{hour_slot}'. Format attendu: début-fin (ex: 9-10)")
                    return
                
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    
                    # Vérifier que ce sont des entiers positifs
                    if start < 0 or end < 0:
                        QMessageBox.warning(self, "Erreur", 
                                        f"Heures invalides: '{hour_slot}'. Les heures doivent être positives")
                        return
                    
                    # Vérifier que end = start + 1 (format i-i+1)
                    if end != start + 1:
                        QMessageBox.warning(self, "Erreur", 
                                        f"Format d'heure invalide: '{hour_slot}'. Le format doit être i-i+1 (ex: 9-10, 14-15)")
                        return
                        
                except ValueError:
                    QMessageBox.warning(self, "Erreur", 
                                    f"Heures invalides: '{hour_slot}'. Les heures doivent être des nombres entiers")
                    return
        
        # Validation avec les fonctions existantes (garder pour compatibilité)
        if days and not normalize_day_list(days):
            QMessageBox.warning(self, "Erreur", "Format des jours invalide. Ex: Mon,Wed ou monday-sunday")
            return
        if hours and not normalize_hour_list(hours):
            QMessageBox.warning(self, "Erreur", "Format des heures invalide. Ex: 9-10,14-15 or 9-10-11")
            return
        
        self.apprenants[name] = {"domain": normalize_domain(domain.strip()), "days": days, "hours": hours}
        save_to_excel(self.tuteurs, self.apprenants)
        self.refresh_apprenant_checkboxes()
        QMessageBox.information(self, "Succès", f"Apprenant {name} ajouté avec succès!")

    def edit_apprenant(self):
        if not self.apprenants: 
            QMessageBox.warning(self, "Erreur", "Aucun apprenant à modifier")
            return
        
        a, ok = QInputDialog.getItem(self, "Modifier un apprenant", "Sélectionnez l'apprenant:", 
                                    sorted(self.apprenants.keys()), 0, False)
        if not ok: 
            return
        
        current_info = self.apprenants[a]
        
        # Contrôle du domaine : doit être une chaîne de caractères, pas des chiffres
        domain, ok1 = QInputDialog.getText(self, "Domaine", "Entrez le domaine:", text=current_info["domain"])
        if not ok1 or not domain.strip():
            QMessageBox.warning(self, "Attention", "Le domaine est obligatoire")
            return
        
        if domain.strip().isdigit():
            QMessageBox.warning(self, "Erreur", "Le domaine ne doit pas être composé uniquement de chiffres")
            return
        
        if all(c.isdigit() for c in domain.strip() if c.isalnum()):
            QMessageBox.warning(self, "Erreur", "Le domaine doit contenir des lettres")
            return
        
        # Saisie des jours
        days_str, ok2 = QInputDialog.getText(self, "Jours", "Jours disponibles:", text=", ".join(current_info["days"]))
        if not ok2:
            days_str = ", ".join(current_info["days"])
        
        days = [d.strip() for d in days_str.split(",") if d.strip()]
        # Dictionnaire pour normaliser les jours
        day_mapping = {
            "mon": "Mon", "monday": "Monday",
            "tue": "Tue", "tuesday": "Tuesday",
            "wed": "Wed", "wednesday": "Wednesday",
            "thu": "Thu", "thursday": "Thursday",
            "fri": "Fri", "friday": "Friday",
            "sat": "Sat", "saturday": "Saturday",
            "sun": "Sun", "sunday": "Sunday",
            "lun": "Lun", "lundi": "Lundi",
            "mar": "Mar", "mardi": "Mardi",
            "mer": "Mer", "mercredi": "Mercredi",
            "jeu": "Jeu", "jeudi": "Jeudi",
            "ven": "Ven", "vendredi": "Vendredi",
            "sam": "Sam", "samedi": "Samedi",
            "dim": "Dim", "dimanche": "Dimanche"
        }

        # Validation et normalisation
        normalized_days = []
        for day in days:
            key = day.strip().lower()
            if key not in day_mapping:
                QMessageBox.warning(self, "Erreur", 
                                    f"Jour invalide: '{day}'. Jours acceptés:\n"
                                    f"Anglais: Mon, Tue, Wed, Thu, Fri, Sat, Sun\n"
                                    f"Français: Lun, Mar, Mer, Jeu, Ven, Sam, Dim")
                return
            normalized_days.append(day_mapping[key])

        # Ensuite, tu remplaces days par normalized_days
        days = normalized_days

        
        # Validation immédiate des jours
        if days:
            valid_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
                        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                        "Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim",
                        "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
            
            for day in days:
                if day not in valid_days:
                    QMessageBox.warning(self, "Erreur", 
                                        f"Jour invalide: '{day}'. Jours acceptés:\n"
                                        f"Anglais: Mon, Tue, Wed, Thu, Fri, Sat, Sun\n"
                                        f"Français: Lun, Mar, Mer, Jeu, Ven, Sam, Dim")
                    return
        
        # Saisie des heures
        hours_str, ok3 = QInputDialog.getText(self, "Heures", "Heures disponibles:", text=", ".join(current_info["hours"]))
        if not ok3:
            hours_str = ", ".join(current_info["hours"])
        
        hours = [h.strip() for h in hours_str.split(",") if h.strip()]
        
        # Validation des heures
        if hours:
            for hour_slot in hours:
                if "-" not in hour_slot:
                    QMessageBox.warning(self, "Erreur", 
                                        f"Format d'heure invalide: '{hour_slot}'. Format attendu: début-fin (ex: 9-10)")
                    return
                parts = hour_slot.split("-")
                if len(parts) != 2:
                    QMessageBox.warning(self, "Erreur", 
                                        f"Format d'heure invalide: '{hour_slot}'. Format attendu: début-fin (ex: 9-10)")
                    return
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    if start < 0 or end < 0:
                        QMessageBox.warning(self, "Erreur", 
                                            f"Heures invalides: '{hour_slot}'. Les heures doivent être positives")
                        return
                    if end != start + 1:
                        QMessageBox.warning(self, "Erreur", 
                                            f"Format d'heure invalide: '{hour_slot}'. Le format doit être i-i+1 (ex: 9-10, 14-15)")
                        return
                except ValueError:
                    QMessageBox.warning(self, "Erreur", 
                                        f"Heures invalides: '{hour_slot}'. Les heures doivent être des nombres entiers")
                    return
        
        # Validation avec les fonctions existantes
        if days and not normalize_day_list(days):
            QMessageBox.warning(self, "Erreur", "Format des jours invalide. Ex: Mon,Wed ou monday-sunday")
            return
        if hours and not normalize_hour_list(hours):
            QMessageBox.warning(self, "Erreur", "Format des heures invalide. Ex: 9-10,14-15 or 9-10-11")
            return
        
        # Mise à jour des données
        self.apprenants[a] = {"domain": normalize_domain(domain.strip()), "days": days, "hours": hours}
        save_to_excel(self.tuteurs, self.apprenants)
        self.refresh_apprenant_checkboxes()
        QMessageBox.information(self, "Succès", f"Apprenant {a} modifié avec succès!")

    def delete_apprenant(self):
        if not self.apprenants: 
            QMessageBox.warning(self, "Erreur", "Aucun apprenant à supprimer")
            return
        
        a, ok = QInputDialog.getItem(self, "Supprimer un apprenant", "Sélectionnez l'apprenant:", 
                                    sorted(self.apprenants.keys()), 0, False)
        if not ok: 
            return
        
        reply = QMessageBox.question(self, "Confirmation", f"Voulez-vous vraiment supprimer l'apprenant {a}?", 
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.apprenants[a]
            save_to_excel(self.tuteurs, self.apprenants)
            self.refresh_apprenant_checkboxes()
            QMessageBox.information(self, "Succès", f"Apprenant {a} supprimé avec succès!")

    def add_tuteur(self):
        name, ok = QInputDialog.getText(self, "Ajouter un tuteur", "Entrez le nom du tuteur (ex: T5):")
        if not ok or not name.strip(): 
            return
        
        if name in self.tuteurs:
            QMessageBox.warning(self, "Erreur", f"Le tuteur {name} existe déjà!")
            return
        
        # --- Contrôle du domaine ---
        domain, ok1 = QInputDialog.getText(self, "Domaine", "Entrez le domaine (ex: Math, Physics, CS):")
        if not ok1 or not domain.strip():
            QMessageBox.warning(self, "Attention", "Le domaine est obligatoire")
            return
        
        if domain.strip().isdigit():
            QMessageBox.warning(self, "Erreur", "Le domaine ne doit pas être composé uniquement de chiffres")
            return
        
        if all(c.isdigit() for c in domain.strip() if c.isalnum()):
            QMessageBox.warning(self, "Erreur", "Le domaine doit contenir des lettres")
            return
        
        # --- Saisie des jours ---
        days_str, ok2 = QInputDialog.getText(self, "Jours", "Jours disponibles (séparés par des virgules, ex: Mon,Wed,Fri):")
        if not ok2:
            days_str = ""
        
        days = [d.strip() for d in days_str.split(",") if d.strip()]
        
        # --- Normalisation et validation des jours ---
        day_mapping = {
            "mon": "Mon", "monday": "Monday",
            "tue": "Tue", "tuesday": "Tuesday",
            "wed": "Wed", "wednesday": "Wednesday",
            "thu": "Thu", "thursday": "Thursday",
            "fri": "Fri", "friday": "Friday",
            "sat": "Sat", "saturday": "Saturday",
            "sun": "Sun", "sunday": "Sunday",
            "lun": "Lun", "lundi": "Lundi",
            "mar": "Mar", "mardi": "Mardi",
            "mer": "Mer", "mercredi": "Mercredi",
            "jeu": "Jeu", "jeudi": "Jeudi",
            "ven": "Ven", "vendredi": "Vendredi",
            "sam": "Sam", "samedi": "Samedi",
            "dim": "Dim", "dimanche": "Dimanche"
        }

        normalized_days = []
        for day in days:
            key = day.lower()
            if key not in day_mapping:
                QMessageBox.warning(self, "Erreur", f"Jour invalide: '{day}'.")
                return
            normalized_days.append(day_mapping[key])
        days = normalized_days  # remplacer par jours normalisés
        
        # --- Saisie des heures ---
        hours_str, ok3 = QInputDialog.getText(self, "Heures", "Heures disponibles (séparées par des virgules, ex: 9-10,14-15):")
        if not ok3:
            hours_str = ""
        
        hours = [h.strip() for h in hours_str.split(",") if h.strip()]
        
        # --- Validation des heures ---
        for hour_slot in hours:
            if "-" not in hour_slot:
                QMessageBox.warning(self, "Erreur", f"Format d'heure invalide: '{hour_slot}'")
                return
            parts = hour_slot.split("-")
            if len(parts) != 2:
                QMessageBox.warning(self, "Erreur", f"Format d'heure invalide: '{hour_slot}'")
                return
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                if start < 0 or end < 0:
                    QMessageBox.warning(self, "Erreur", f"Heures invalides: '{hour_slot}'. Les heures doivent être positives")
                    return
                if end != start + 1:
                    QMessageBox.warning(self, "Erreur", f"Format d'heure invalide: '{hour_slot}'. Doit être i-i+1")
                    return
            except ValueError:
                QMessageBox.warning(self, "Erreur", f"Heures invalides: '{hour_slot}'. Les heures doivent être des entiers")
                return
        
        # --- Sauvegarde ---
        self.tuteurs[name] = {"domain": normalize_domain(domain.strip()), "days": days, "hours": hours}
        save_to_excel(self.tuteurs, self.apprenants)
        self.refresh_tuteur_checkboxes()
        QMessageBox.information(self, "Succès", f"Tuteur {name} ajouté avec succès!")


    def edit_tuteur(self):
        if not self.tuteurs: 
            QMessageBox.warning(self, "Erreur", "Aucun tuteur à modifier")
            return
        
        t, ok = QInputDialog.getItem(self, "Modifier un tuteur", "Sélectionnez le tuteur:", 
                                    sorted(self.tuteurs.keys()), 0, False)
        if not ok: 
            return
        
        current_info = self.tuteurs[t]
        
        # --- Contrôle du domaine ---
        domain, ok1 = QInputDialog.getText(self, "Domaine", "Entrez le domaine:", text=current_info["domain"])
        if not ok1 or not domain.strip():
            QMessageBox.warning(self, "Attention", "Le domaine est obligatoire")
            return
        
        if domain.strip().isdigit():
            QMessageBox.warning(self, "Erreur", "Le domaine ne doit pas être composé uniquement de chiffres")
            return
        
        if all(c.isdigit() for c in domain.strip() if c.isalnum()):
            QMessageBox.warning(self, "Erreur", "Le domaine doit contenir des lettres")
            return
        
        # --- Saisie et normalisation des jours ---
        days_str, ok2 = QInputDialog.getText(self, "Jours", "Jours disponibles:", text=", ".join(current_info["days"]))
        if not ok2:
            days_str = ", ".join(current_info["days"])
        
        days = [d.strip() for d in days_str.split(",") if d.strip()]
        
        day_mapping = {
            "mon": "Mon", "monday": "Monday",
            "tue": "Tue", "tuesday": "Tuesday",
            "wed": "Wed", "wednesday": "Wednesday",
            "thu": "Thu", "thursday": "Thursday",
            "fri": "Fri", "friday": "Friday",
            "sat": "Sat", "saturday": "Saturday",
            "sun": "Sun", "sunday": "Sunday",
            "lun": "Lun", "lundi": "Lundi",
            "mar": "Mar", "mardi": "Mardi",
            "mer": "Mer", "mercredi": "Mercredi",
            "jeu": "Jeu", "jeudi": "Jeudi",
            "ven": "Ven", "vendredi": "Vendredi",
            "sam": "Sam", "samedi": "Samedi",
            "dim": "Dim", "dimanche": "Dimanche"
        }

        normalized_days = []
        for day in days:
            key = day.lower()
            if key not in day_mapping:
                QMessageBox.warning(self, "Erreur", f"Jour invalide: '{day}'.")
                return
            normalized_days.append(day_mapping[key])
        days = normalized_days  # remplacer par jours normalisés
        
        # --- Saisie et validation des heures ---
        hours_str, ok3 = QInputDialog.getText(self, "Heures", "Heures disponibles:", text=", ".join(current_info["hours"]))
        if not ok3:
            hours_str = ", ".join(current_info["hours"])
        
        hours = [h.strip() for h in hours_str.split(",") if h.strip()]
        
        for hour_slot in hours:
            if "-" not in hour_slot:
                QMessageBox.warning(self, "Erreur", f"Format d'heure invalide: '{hour_slot}'")
                return
            parts = hour_slot.split("-")
            if len(parts) != 2:
                QMessageBox.warning(self, "Erreur", f"Format d'heure invalide: '{hour_slot}'")
                return
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                if start < 0 or end < 0:
                    QMessageBox.warning(self, "Erreur", f"Heures invalides: '{hour_slot}'. Les heures doivent être positives")
                    return
                if end != start + 1:
                    QMessageBox.warning(self, "Erreur", f"Format d'heure invalide: '{hour_slot}'. Doit être i-i+1")
                    return
            except ValueError:
                QMessageBox.warning(self, "Erreur", f"Heures invalides: '{hour_slot}'. Les heures doivent être des entiers")
                return
        
        # --- Sauvegarde ---
        self.tuteurs[t] = {"domain": normalize_domain(domain.strip()), "days": days, "hours": hours}
        save_to_excel(self.tuteurs, self.apprenants)
        self.refresh_tuteur_checkboxes()
        QMessageBox.information(self, "Succès", f"Tuteur {t} modifié avec succès!")


    def delete_tuteur(self):
        if not self.tuteurs: 
            QMessageBox.warning(self, "Erreur", "Aucun tuteur à supprimer")
            return
        
        t, ok = QInputDialog.getItem(self, "Supprimer un tuteur", "Sélectionnez le tuteur:", 
                                    sorted(self.tuteurs.keys()), 0, False)
        if not ok: 
            return
        
        reply = QMessageBox.question(self, "Confirmation", f"Voulez-vous vraiment supprimer le tuteur {t}?", 
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.tuteurs[t]
            save_to_excel(self.tuteurs, self.apprenants)
            self.refresh_tuteur_checkboxes()
            QMessageBox.information(self, "Succès", f"Tuteur {t} supprimé avec succès!")

    def import_from_excel(self):
        # Let user pick an Excel file, then ask to Replace or Merge
        fname, _ = QFileDialog.getOpenFileName(self, "Importer depuis Excel", "", "Excel Files (*.xlsx *.xls);;All Files (*)")
        if not fname:
            return

        try:
            t_loaded, a_loaded = load_from_excel(fname)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier Excel:\n{e}")
            return

        # Ask user whether to replace or merge
        reply = QMessageBox.question(self, "Importer Excel",
                                     "Remplacer les données actuelles par celles du fichier ?\n'Yes' = Remplacer, 'No' = Fusionner (ajouter manquants)",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Replace
            self.tuteurs = t_loaded
            self.apprenants = a_loaded
        else:
            # Merge: add missing entries but don't overwrite existing
            for k, v in t_loaded.items():
                if k not in self.tuteurs:
                    self.tuteurs[k] = v
            for k, v in a_loaded.items():
                if k not in self.apprenants:
                    self.apprenants[k] = v

        # Save merged/replaced data back to default Excel file
        try:
            save_to_excel(self.tuteurs, self.apprenants)
        except Exception:
            pass

        # Refresh UI
        self.refresh_apprenant_checkboxes()
        self.refresh_tuteur_checkboxes()
        QMessageBox.information(self, "Importation", "Importation terminée.")

    # ---------------- Optimisation ----------------
    def run_optimization(self):
        if not self.selected_tuteurs or not self.selected_apprenants:
            QMessageBox.warning(self, "Erreur", "Sélectionnez au moins un tuteur et un apprenant")
            return

        # Désactiver les boutons pendant l'optimisation
        self.setEnabled(False)

        # Préparer la barre de progression non bloquante
        self.progress = QProgressDialog("Optimisation en cours...", None, 0, 0, self)
        self.progress.setWindowTitle("Optimisation")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()

        # Créer le worker et le thread
        self._model_holder = {}
        self._opt_thread = QThread()
        self._opt_worker = OptimizeWorker(self.tuteurs, self.apprenants, self.selected_tuteurs, self.selected_apprenants, model_holder=self._model_holder, time_limit=self.time_limit)
        self._opt_worker.moveToThread(self._opt_thread)

        # Connexions
        self._opt_thread.started.connect(self._opt_worker.run)
        self._opt_worker.finished.connect(self.on_optimization_finished)
        self._opt_worker.error.connect(self.on_optimization_error)
        self._opt_worker.finished.connect(self._opt_thread.quit)
        self._opt_worker.error.connect(self._opt_thread.quit)
        self._opt_thread.finished.connect(self._opt_thread.deleteLater)

        # Démarrer
        self._opt_thread.start()

    def clear_results(self):
        # Clear solution, table and visualization
        self.solution = None
        self.table.setRowCount(0)
        self.scene.clear()
        self.selection_info.setText("")

    def cancel_optimization(self):
        # Attempt to abort running optimization
        try:
            if self._opt_worker:
                self._opt_worker.abort()
            if hasattr(self, '_opt_thread') and self._opt_thread is not None:
                self._opt_thread.quit()
                self._opt_thread.wait(500)
        except Exception:
            pass

    def save_solution(self):
        """Sauvegarde la solution actuelle dans le fichier Excel."""
        if not self.solution:
            QMessageBox.warning(self, "Erreur", "Aucune solution à sauvegarder. Exécutez l'optimisation d'abord.")
            return

        # Demander à l'utilisateur de choisir le fichier Excel
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Sauvegarder la solution",
            "solution.xlsx",       # nom par défaut
            "Fichier Excel (*.xlsx)"
        )
        if not filename:  # Si l'utilisateur annule
            return

        total_couplages = sum(len(app_list) for app_list in self.solution.values()) if self.solution else 0
        try:
            save_solution_to_excel(self.solution, filename)  # passer le fichier choisi
            QMessageBox.information(self, "Sauvegarde", f"✅ Solution sauvegardée avec {total_couplages} couplage(s)")
        except Exception as e:
            QMessageBox.critical(self, "Erreur sauvegarde", f"Impossible de sauvegarder la solution:\n{e}")

    def export_visualization(self):
        # Export the QGraphicsScene to PNG
        if self.scene is None:
            QMessageBox.warning(self, "Erreur", "Rien à exporter")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Enregistrer la visualisation", "visualisation.png", "PNG Files (*.png);;All Files (*)")
        if not fname:
            return
        # Render scene to image
        rect = self.scene.itemsBoundingRect()
        image = QImage(int(rect.width())+20, int(rect.height())+20, QImage.Format_ARGB32)
        image.fill(QColor('white'))
        painter = QPainter(image)
        self.scene.render(painter, target=QRectF(image.rect()), source=rect)
        painter.end()
        image.save(fname)

    def toggle_debug_mode(self, state):
        self.verbose = bool(state)

    def set_time_limit(self):
        t, ok = QInputDialog.getText(self, "TimeLimit", "Entrez le time limit (secondes), laisser vide pour aucun:")
        if ok:
            try:
                self.time_limit = float(t) if t.strip() else None
            except Exception:
                QMessageBox.warning(self, "Erreur", "TimeLimit invalide")

    def on_table_cell_clicked(self, row, col):
        # When user clicks a row, highlight corresponding link in visualization
        try:
            t_item = self.table.item(row, 0)
            a_item = self.table.item(row, 1)
            if not t_item or not a_item:
                return
            t = t_item.text()
            a = a_item.text()
            self.highlight_pair(t, a)
        except Exception:
            pass

    def highlight_pair(self, t, a):
        # Reset all lines to default, then highlight the selected pair
        for key, line in list(self._line_items.items()):
            pen = QPen(QColor("red"), 2)
            pen.setStyle(Qt.DashLine)
            line.setPen(pen)
        # Highlight selected
        key = (t, a)
        if key in self._line_items:
            line = self._line_items[key]
            pen = QPen(QColor("orange"), 3)
            line.setPen(pen)

    def toggle_visualization(self):
        # Toggle vis_label and view visibility
        visible = not self.view.isVisible()
        self.view.setVisible(visible)
        self.vis_label.setVisible(visible)
        # Update button text if present in layout
        # find the button by text and toggle its label
        try:
            if hasattr(self, 'b_toggle_vis') and self.b_toggle_vis:
                self.b_toggle_vis.setText("Masquer la visualisation" if visible else "Afficher la visualisation")
        except Exception:
            pass

    def on_optimization_finished(self, solution):
        # Called in the main thread when worker finished
        try:
            self.solution = solution if solution is not None else {}
            self.populate_table()
            self.visualize_2D()
            total_couplages = sum(len(app_list) for app_list in self.solution.values()) if self.solution else 0
            total_tuteurs = len(self.selected_tuteurs)
            total_apprenants = len(self.selected_apprenants)

            # Afficher le résumé dans le label de sélection au lieu d'une fenêtre modale
            self.selection_info.setText(f"✅ Couplages: {total_couplages} | Tuteurs: {total_tuteurs} | Apprenants: {total_apprenants}")
        finally:
            try:
                if hasattr(self, 'progress') and self.progress:
                    self.progress.close()
            except:
                pass
            self.setEnabled(True)

    def on_optimization_error(self, tb):
        try:
            if hasattr(self, 'progress') and self.progress:
                self.progress.close()
        except:
            pass
        self.setEnabled(True)
        print(tb)
        QMessageBox.critical(self, "Erreur optimisation", f"Une erreur est survenue lors de l'optimisation:\n\nTrace:\n{tb}")

    def populate_table(self):
        self.table.setRowCount(0)
        
        if not self.solution:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("Aucun couplage trouvé"))
            for col in range(1, 5):
                self.table.setItem(0, col, QTableWidgetItem(""))
            return
        
        row_count = 0
        for t, app_list in self.solution.items():
            for a, info in app_list:
                row = self.table.rowCount()
                self.table.insertRow(row)
                
                self.table.setItem(row, 0, QTableWidgetItem(t))
                self.table.setItem(row, 1, QTableWidgetItem(a))
                self.table.setItem(row, 2, QTableWidgetItem(info.get("domain", "")))
                self.table.setItem(row, 3, QTableWidgetItem(", ".join(info.get("days", []))))
                self.table.setItem(row, 4, QTableWidgetItem(", ".join(info.get("hours", []))))
                
                # Colorer les lignes alternativement
                if row % 2 == 0:
                    for col in range(5):
                        if self.table.item(row, col):
                            self.table.item(row, col).setBackground(QColor(240, 240, 240))
                
                row_count += 1
        
        # Ajuster la largeur des colonnes
        self.table.resizeColumnsToContents()

    def visualize_2D(self):
        self.scene.clear()
        # reset stored graphics references
        self._line_items.clear()
        self._ellipse_items = {"tuteurs": {}, "apprenants": {}}
        
        if not self.selected_tuteurs or not self.selected_apprenants:
            return
        
        # Titre
        title = self.scene.addText("Visualisation des couplages Tuteur-Apprenant")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setPos(10, 10)
        
        # Légende
        legend_tuteur = self.scene.addRect(20, 50, 15, 15, brush=QBrush(QColor("blue")))
        legend_tuteur_text = self.scene.addText("Tuteur")
        legend_tuteur_text.setPos(40, 48)
        
        legend_apprenant = self.scene.addRect(20, 70, 15, 15, brush=QBrush(QColor("green")))
        legend_apprenant_text = self.scene.addText("Apprenant")
        legend_apprenant_text.setPos(40, 68)
        
        legend_couplage = self.scene.addLine(20, 90, 50, 90, QPen(QColor("red"), 2))
        legend_couplage.setPen(QPen(QColor("red"), 2, Qt.DashLine))
        legend_couplage_text = self.scene.addText("Couplage")
        legend_couplage_text.setPos(55, 85)
        
        # Calculer les positions
        tuteur_pos = {}
        apprenant_pos = {}
        
        # Positionner les tuteurs (colonne de gauche)
        x_tuteur = 100
        y_start = 150
        y_step = 80
        
        for i, t in enumerate(self.selected_tuteurs):
            y = y_start + i * y_step
            tuteur_pos[t] = (x_tuteur, y)
            
            # Cercle tuteur
            ellipse = QGraphicsEllipseItem(x_tuteur - 20, y - 20, 40, 40)
            ellipse.setBrush(QColor("blue"))
            ellipse.setPen(QPen(Qt.black, 1))
            self.scene.addItem(ellipse)
            # store ellipse for potential highlighting
            tip = ''
            if t in self.tuteurs:
                tip = f"{t} - {self.tuteurs[t].get('domain','')}\nJours: {', '.join(self.tuteurs[t].get('days',[]))}\nHeures: {', '.join(self.tuteurs[t].get('hours',[]))}"
            ellipse.setToolTip(tip)
            self._ellipse_items['tuteurs'][t] = ellipse
            
            # Nom du tuteur
            text = QGraphicsTextItem(t)
            text.setPos(x_tuteur - 15, y - 15)
            text.setDefaultTextColor(Qt.white)
            text.setFont(QFont("Arial", 8, QFont.Bold))
            self.scene.addItem(text)
            
            # Domaine du tuteur
            if t in self.tuteurs:
                domain_text = QGraphicsTextItem(self.tuteurs[t]["domain"])
                domain_text.setPos(x_tuteur - 20, y + 25)
                domain_text.setFont(QFont("Arial", 7))
                self.scene.addItem(domain_text)
        
        # Positionner les apprenants (colonne de droite)
        x_apprenant = 400
        for i, a in enumerate(self.selected_apprenants):
            y = y_start + i * y_step
            apprenant_pos[a] = (x_apprenant, y)
            
            # Cercle apprenant
            ellipse = QGraphicsEllipseItem(x_apprenant - 20, y - 20, 40, 40)
            ellipse.setBrush(QColor("green"))
            ellipse.setPen(QPen(Qt.black, 1))
            self.scene.addItem(ellipse)
            tip = ''
            if a in self.apprenants:
                tip = f"{a} - {self.apprenants[a].get('domain','')}\nJours: {', '.join(self.apprenants[a].get('days',[]))}\nHeures: {', '.join(self.apprenants[a].get('hours',[]))}"
            ellipse.setToolTip(tip)
            self._ellipse_items['apprenants'][a] = ellipse
            
            # Nom de l'apprenant
            text = QGraphicsTextItem(a)
            text.setPos(x_apprenant - 15, y - 15)
            text.setDefaultTextColor(Qt.white)
            text.setFont(QFont("Arial", 8, QFont.Bold))
            self.scene.addItem(text)
            
            # Domaine de l'apprenant
            if a in self.apprenants:
                domain_text = QGraphicsTextItem(self.apprenants[a]["domain"])
                domain_text.setPos(x_apprenant - 20, y + 25)
                domain_text.setFont(QFont("Arial", 7))
                self.scene.addItem(domain_text)
        
        # Dessiner les liens (couplages)
        if self.solution:
            for t, app_list in self.solution.items():
                for a, info in app_list:
                    if t in tuteur_pos and a in apprenant_pos:
                        x1, y1 = tuteur_pos[t]
                        x2, y2 = apprenant_pos[a]
                        
                        # Ligne avec style pointillé
                        line = QGraphicsLineItem(x1 + 20, y1, x2 - 20, y2)
                        pen = QPen(QColor("red"), 2)
                        pen.setStyle(Qt.DashLine)
                        line.setPen(pen)
                        self.scene.addItem(line)
                        # store line reference for interaction
                        self._line_items[(t, a)] = line
        
        # Statistiques
        total_couplages = sum(len(app_list) for app_list in self.solution.values()) if self.solution else 0
        stats_text = f"Couplages trouvés: {total_couplages}"
        stats = self.scene.addText(stats_text)
        stats.setPos(100, y_start + len(self.selected_tuteurs) * y_step + 30)
        stats.setFont(QFont("Arial", 10, QFont.Bold))
        stats.setDefaultTextColor(QColor("darkblue"))

    def update_rapport(self):
        """Génère et affiche le rapport d'optimisation"""
        if not self.solution:
            self.rapport_text.setText("❌ Aucune solution d'optimisation.\nExécutez d'abord l'optimisation.")
            return
        
        # Analyser la couverture
        self.coverage = analyze_coverage(self.tuteurs, self.apprenants, self.solution)
        
        # Détecter les goulets
        self.bottlenecks = detect_bottlenecks(self.tuteurs, self.apprenants, 
                                             self.selected_tuteurs, self.selected_apprenants)
        
        rapport_lines = [
            "=" * 70,
            "RAPPORT D'OPTIMISATION - COUPLAGE TUTEUR/APPRENANT",
            "=" * 70,
            "",
            "📊 STATISTIQUES GÉNÉRALES",
            "-" * 70,
            f"Temps de résolution: {self.solve_time:.2f} secondes",
            f"Nombre total de couplages: {self.coverage['total_couplings']}",
            "",
            "👥 COUVERTURE TUTEURS",
            "-" * 70,
            f"  Total: {self.coverage['total_tuteurs']}",
            f"  Couplés: {self.coverage['tuteurs_couples']} ({self.coverage['taux_tuteurs']:.1f}%)",
            f"  Non-couplés: {len(self.coverage['tuteurs_non_couples'])}",
            f"  Liste: {', '.join(self.coverage['tuteurs_non_couples']) if self.coverage['tuteurs_non_couples'] else 'Aucun'}",
            "",
            "📚 COUVERTURE APPRENANTS",
            "-" * 70,
            f"  Total: {self.coverage['total_apprenants']}",
            f"  Couplés: {self.coverage['apprenants_couples']} ({self.coverage['taux_apprenants']:.1f}%)",
            f"  Non-couplés: {len(self.coverage['apprenants_non_couples'])}",
            f"  Liste: {', '.join(self.coverage['apprenants_non_couples']) if self.coverage['apprenants_non_couples'] else 'Aucun'}",
            "",
            "🎯 ANALYSE GOULETS D'ÉTRANGLEMENT",
            "-" * 70,
            "Domaines (distribtion tuteurs):",
            f"  {dict(self.bottlenecks['domains']['tuteurs'])}",
            "Domaines (distribution apprenants):",
            f"  {dict(self.bottlenecks['domains']['apprenants'])}",
            "",
            "Jours critiques (moins représentés):",
            f"  Tuteurs: {self.bottlenecks['days']['rarest_day_t']}",
            f"  Apprenants: {self.bottlenecks['days']['rarest_day_a']}",
            "",
            "Heures critiques (moins représentées):",
            f"  Tuteurs: {self.bottlenecks['hours']['rarest_hour_t']}",
            f"  Apprenants: {self.bottlenecks['hours']['rarest_hour_a']}",
            "",
            "💡 RECOMMANDATIONS",
            "-" * 70,
        ]
        
        # Ajouter des recommandations basées sur l'analyse
        recommendations = []
        if self.coverage['taux_tuteurs'] < 50:
            recommendations.append("⚠ Ajouter plus de tuteurs pour augmenter la couverture")
        if self.coverage['taux_apprenants'] < 50:
            recommendations.append("⚠ Ajouter plus d'apprenants ou augmenter disponibilité tuteurs")
        
        if self.bottlenecks['days']['rarest_day_t']:
            rarest = self.bottlenecks['days']['rarest_day_t']
            recommendations.append(f"⚠ Très peu de tuteurs disponibles {rarest}: considérez d'en ajouter")
        
        if not recommendations:
            recommendations.append("✅ Configuration optimale - tous les tuteurs et apprenants sont bien répartis")
        
        rapport_lines.extend(recommendations)
        rapport_lines.extend(["", "=" * 70])
        
        self.rapport_text.setText("\n".join(rapport_lines))

    def generate_advanced_visualizations(self):
        """Génère les heatmaps et graphe bipartite avancé"""
        if not self.solution:
            QMessageBox.warning(self, "Erreur", "Exécutez d'abord l'optimisation")
            return
        
        if not self.bottlenecks:
            self.update_rapport()  # Générer les analyses si nécessaire
        
        # Heatmap Domaines
        self.draw_heatmap_domains()
        
        # Heatmap Jours
        self.draw_heatmap_days()
        
        # Heatmap Heures
        self.draw_heatmap_hours()
        
        # Graphe bipartite amélioré
        self.draw_bipartite_graph()
        
        

    def draw_heatmap_domains(self):
        """Dessine une heatmap des domaines (SEULEMENT VISUALISATION GRAPHIQUE)"""
        self.heatmap_domain_scene.clear()
        
        domains_t = self.bottlenecks['domains']['tuteurs']
        domains_a = self.bottlenecks['domains']['apprenants']

        # Group domains by normalized display name
        grouped_t = {}
        grouped_a = {}
        for k, v in domains_t.items():
            disp = normalize_domain(k)
            grouped_t[disp] = grouped_t.get(disp, 0) + v
        for k, v in domains_a.items():
            disp = normalize_domain(k)
            grouped_a[disp] = grouped_a.get(disp, 0) + v

        all_domains = sorted(set(list(grouped_t.keys()) + list(grouped_a.keys())))
        
        if not all_domains:
            # Afficher un message si pas de données
            no_data = self.heatmap_domain_scene.addText("Aucune donnée disponible")
            no_data.setFont(QFont("Arial", 12))
            no_data.setPos(100, 100)
            return
        
        # TITRE PRINCIPAL - PLUS GRAND ET PLUS HAUT
        title = self.heatmap_domain_scene.addText("Heatmap: Distribution par Domaine")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setPos(50, 10)
        
        # LÉGENDE - BIEN ESPACÉE
        # Indicateur couleur + texte pour Tuteurs
        color_t_rect = self.heatmap_domain_scene.addRect(100, 50, 20, 20)
        color_t_rect.setBrush(QColor(255, 200, 200))  # Rouge clair
        color_t_rect.setPen(QPen(Qt.black, 1))
        
        legend_t = self.heatmap_domain_scene.addText("Tuteurs")
        legend_t.setFont(QFont("Arial", 12, QFont.Bold))
        legend_t.setPos(125, 50)  # Espace après le carré de couleur
        
        # Indicateur couleur + texte pour Apprenants - BIEN SÉPARÉ
        color_a_rect = self.heatmap_domain_scene.addRect(250, 50, 20, 20)
        color_a_rect.setBrush(QColor(70, 70, 220))  # Bleu foncé
        color_a_rect.setPen(QPen(Qt.black, 1))
        
        legend_a = self.heatmap_domain_scene.addText("Apprenants")
        legend_a.setFont(QFont("Arial", 12, QFont.Bold))
        legend_a.setPos(275, 50)  # Espace après le carré de couleur
        
        # CONFIGURATION DE LA HEATMAP
        cell_size = 50  # Cellules plus grandes pour mieux voir
        x_offset = 150  # Position horizontale de départ
        y_offset = 90   # Position verticale (après la légende)
        column_gap = 100  # ESPACE IMPORTANT entre les deux colonnes
        
        # LIGNE DE SÉPARATION ENTRE COLONNES - BIEN VISIBLE
        separator_x = x_offset + cell_size + (column_gap / 2)
        separator = self.heatmap_domain_scene.addLine(
            separator_x, y_offset - 5, 
            separator_x, y_offset + len(all_domains) * cell_size + 5, 
            QPen(QColor(150, 150, 150), 3, Qt.SolidLine)
        )
        
        
       
        
        # DESSINER CHAQUE LIGNE (DOMAINE)
        for i, domain in enumerate(all_domains):
            # NOM DU DOMAINE - À GAUCHE
            domain_label = self.heatmap_domain_scene.addText(domain)
            domain_label.setFont(QFont("Arial", 11, QFont.Bold))
            domain_label.setPos(30, y_offset + i * cell_size + 15)  # Bien à gauche
            
            # CELLULE TUTEURS (COLONNE GAUCHE)
            count_t = grouped_t.get(domain, 0)
            max_count = max(max(domains_t.values()) if domains_t else 0, 
                          max(domains_a.values()) if domains_a else 0)
            if max_count <= 0:
                max_count = 1

            # Intensité couleur basée sur le nombre
            intensity_t = min(255, int(220 * count_t / max_count))
            rect_t = self.heatmap_domain_scene.addRect(x_offset, y_offset + i * cell_size, cell_size, cell_size)
            rect_t.setBrush(QColor(255, 255 - intensity_t, 255 - intensity_t))  # Rougeâtre
            rect_t.setPen(QPen(QColor(80, 80, 80), 2))  # Bordure plus épaisse
            
            # NOMBRE dans la cellule Tuteurs
            text_t = self.heatmap_domain_scene.addText(str(count_t))
            text_t.setFont(QFont("Arial", 12, QFont.Bold))
            # Centrage approximatif du texte
            text_width_t = len(str(count_t)) * 8
            text_t.setPos(x_offset + cell_size/2 - text_width_t/2 + 2, y_offset + i * cell_size + cell_size/2 - 10)
            
            # CELLULE APPRENANTS (COLONNE DROITE)
            count_a = grouped_a.get(domain, 0)
            intensity_a = min(255, int(220 * count_a / max_count))
            rect_a = self.heatmap_domain_scene.addRect(x_offset + cell_size + column_gap, y_offset + i * cell_size, cell_size, cell_size)
            rect_a.setBrush(QColor(255 - intensity_a, 255 - intensity_a, 255))  # Bleuâtre
            rect_a.setPen(QPen(QColor(80, 80, 80), 2))  # Bordure plus épaisse
            
            # NOMBRE dans la cellule Apprenants
            text_a = self.heatmap_domain_scene.addText(str(count_a))
            text_a.setFont(QFont("Arial", 12, QFont.Bold))
            # Centrage approximatif du texte
            text_width_a = len(str(count_a)) * 8
            text_a.setPos(x_offset + cell_size + column_gap + cell_size/2 - text_width_a/2 + 2, y_offset + i * cell_size + cell_size/2 - 10)

    def draw_heatmap_days(self):
        """Dessine une heatmap des jours avec le même design que celle des domaines"""
        self.heatmap_day_scene.clear()
        
        days_t = self.bottlenecks['days']['tuteurs']
        days_a = self.bottlenecks['days']['apprenants']
        
        # Ordre des jours avec normalisation
        days_order = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        days_display = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        
        # Garder seulement les jours pertinents
        relevant_indices = [i for i, day in enumerate(days_order) if day in days_t or day in days_a]
        relevant_days = [days_display[i] for i in relevant_indices]
        
        if not relevant_days:
            # Afficher un message si pas de données
            no_data = self.heatmap_day_scene.addText("Aucune donnée disponible")
            no_data.setFont(QFont("Arial", 12))
            no_data.setPos(100, 100)
            return
        
        # TITRE PRINCIPAL - MÊME STYLE QUE DOMAINES
        title = self.heatmap_day_scene.addText("Heatmap: Distribution par Jour")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setPos(50, 10)
        
        # LÉGENDE - MÊME STYLE QUE DOMAINES
        # Indicateur couleur + texte pour Tuteurs
        color_t_rect = self.heatmap_day_scene.addRect(100, 50, 20, 20)
        color_t_rect.setBrush(QColor(255, 200, 200))  # Rouge clair
        color_t_rect.setPen(QPen(Qt.black, 1))
        
        legend_t = self.heatmap_day_scene.addText("Tuteurs")
        legend_t.setFont(QFont("Arial", 12, QFont.Bold))
        legend_t.setPos(125, 50)
        
        # Indicateur couleur + texte pour Apprenants
        color_a_rect = self.heatmap_day_scene.addRect(250, 50, 20, 20)
        color_a_rect.setBrush(QColor(70, 70, 220))  # Bleu foncé
        color_a_rect.setPen(QPen(Qt.black, 1))
        
        legend_a = self.heatmap_day_scene.addText("Apprenants")
        legend_a.setFont(QFont("Arial", 12, QFont.Bold))
        legend_a.setPos(275, 50)
        
        # CONFIGURATION DE LA HEATMAP
        cell_size = 50  # Cellules plus grandes
        x_offset = 150  # Position horizontale de départ
        y_offset = 90   # Position verticale (après la légende)
        column_gap = 100  # Espace entre les colonnes
        
        # LIGNE DE SÉPARATION ENTRE COLONNES
        separator_x = x_offset + cell_size + (column_gap / 2)
        separator = self.heatmap_day_scene.addLine(
            separator_x, y_offset - 5, 
            separator_x, y_offset + len(relevant_days) * cell_size + 5, 
            QPen(QColor(150, 150, 150), 3, Qt.SolidLine)
        )
        
        # DESSINER CHAQUE LIGNE (JOUR)
        for i, (day_key, day_display) in enumerate(zip([days_order[idx] for idx in relevant_indices], relevant_days)):
            # NOM DU JOUR - À GAUCHE
            day_label = self.heatmap_day_scene.addText(day_display)
            day_label.setFont(QFont("Arial", 11, QFont.Bold))
            day_label.setPos(30, y_offset + i * cell_size + 15)
            
            # CELLULE TUTEURS (COLONNE GAUCHE)
            count_t = days_t.get(day_key, 0)
            max_count = max(max(days_t.values()) if days_t else 0, 
                        max(days_a.values()) if days_a else 0)
            if max_count <= 0:
                max_count = 1
            
            # Intensité couleur basée sur le nombre
            intensity_t = min(255, int(220 * count_t / max_count))
            rect_t = self.heatmap_day_scene.addRect(x_offset, y_offset + i * cell_size, cell_size, cell_size)
            rect_t.setBrush(QColor(255, 255 - intensity_t, 255 - intensity_t))  # Rougeâtre
            rect_t.setPen(QPen(QColor(80, 80, 80), 2))  # Bordure plus épaisse
            
            # NOMBRE dans la cellule Tuteurs
            text_t = self.heatmap_day_scene.addText(str(count_t))
            text_t.setFont(QFont("Arial", 12, QFont.Bold))
            text_width_t = len(str(count_t)) * 8
            text_t.setPos(x_offset + cell_size/2 - text_width_t/2 + 2, y_offset + i * cell_size + cell_size/2 - 10)
            
            # CELLULE APPRENANTS (COLONNE DROITE)
            count_a = days_a.get(day_key, 0)
            intensity_a = min(255, int(220 * count_a / max_count))
            rect_a = self.heatmap_day_scene.addRect(x_offset + cell_size + column_gap, y_offset + i * cell_size, cell_size, cell_size)
            rect_a.setBrush(QColor(255 - intensity_a, 255 - intensity_a, 255))  # Bleuâtre
            rect_a.setPen(QPen(QColor(80, 80, 80), 2))  # Bordure plus épaisse
            
            # NOMBRE dans la cellule Apprenants
            text_a = self.heatmap_day_scene.addText(str(count_a))
            text_a.setFont(QFont("Arial", 12, QFont.Bold))
            text_width_a = len(str(count_a)) * 8
            text_a.setPos(x_offset + cell_size + column_gap + cell_size/2 - text_width_a/2 + 2, y_offset + i * cell_size + cell_size/2 - 10)

    def draw_heatmap_hours(self):
            """Dessine une heatmap des heures avec le même design que celle des domaines"""
            self.heatmap_hour_scene.clear()
            
            hours_t = self.bottlenecks['hours']['tuteurs']
            hours_a = self.bottlenecks['hours']['apprenants']
            all_hours = sorted(set(list(hours_t.keys()) + list(hours_a.keys())))
            
            if not all_hours:
                # Afficher un message si pas de données
                no_data = self.heatmap_hour_scene.addText("Aucune donnée disponible")
                no_data.setFont(QFont("Arial", 12))
                no_data.setPos(100, 100)
                return
            
            # TITRE PRINCIPAL - MÊME STYLE QUE DOMAINES
            title = self.heatmap_hour_scene.addText("Heatmap: Distribution par Créneau Horaire")
            title.setFont(QFont("Arial", 16, QFont.Bold))
            title.setPos(50, 10)
            
            # LÉGENDE - MÊME STYLE QUE DOMAINES
            # Indicateur couleur + texte pour Tuteurs
            color_t_rect = self.heatmap_hour_scene.addRect(100, 50, 20, 20)
            color_t_rect.setBrush(QColor(255, 200, 200))  # Rouge clair
            color_t_rect.setPen(QPen(Qt.black, 1))
            
            legend_t = self.heatmap_hour_scene.addText("Tuteurs")
            legend_t.setFont(QFont("Arial", 12, QFont.Bold))
            legend_t.setPos(125, 50)
            
            # Indicateur couleur + texte pour Apprenants
            color_a_rect = self.heatmap_hour_scene.addRect(250, 50, 20, 20)
            color_a_rect.setBrush(QColor(70, 70, 220))  # Bleu foncé
            color_a_rect.setPen(QPen(Qt.black, 1))
            
            legend_a = self.heatmap_hour_scene.addText("Apprenants")
            legend_a.setFont(QFont("Arial", 12, QFont.Bold))
            legend_a.setPos(275, 50)
            
            # CONFIGURATION DE LA HEATMAP
            cell_size = 50  # Cellules plus grandes
            x_offset = 150  # Position horizontale de départ
            y_offset = 90   # Position verticale (après la légende)
            column_gap = 100  # Espace entre les colonnes
            
            # LIGNE DE SÉPARATION ENTRE COLONNES
            separator_x = x_offset + cell_size + (column_gap / 2)
            separator = self.heatmap_hour_scene.addLine(
                separator_x, y_offset - 5, 
                separator_x, y_offset + len(all_hours) * cell_size + 5, 
                QPen(QColor(150, 150, 150), 3, Qt.SolidLine)
            )
            
            # DESSINER CHAQUE LIGNE (HEURE)
            for i, hour in enumerate(all_hours):
                # FORMATAGE DE L'HEURE POUR L'AFFICHAGE
                if "-" in hour:
                    # Format "9-10"
                    parts = hour.split("-")
                    hour_display = f"{parts[0]}h-{parts[1]}h"
                else:
                    hour_display = f"{hour}h"
                
                # NOM DE L'HEURE - À GAUCHE
                hour_label = self.heatmap_hour_scene.addText(hour_display)
                hour_label.setFont(QFont("Arial", 11, QFont.Bold))
                hour_label.setPos(30, y_offset + i * cell_size + 15)
                
                # CELLULE TUTEURS (COLONNE GAUCHE)
                count_t = hours_t.get(hour, 0)
                max_count = max(max(hours_t.values()) if hours_t else 0, 
                            max(hours_a.values()) if hours_a else 0)
                if max_count <= 0:
                    max_count = 1
                
                # Intensité couleur basée sur le nombre
                intensity_t = min(255, int(220 * count_t / max_count))
                rect_t = self.heatmap_hour_scene.addRect(x_offset, y_offset + i * cell_size, cell_size, cell_size)
                rect_t.setBrush(QColor(255, 255 - intensity_t, 255 - intensity_t))  # Rougeâtre
                rect_t.setPen(QPen(QColor(80, 80, 80), 2))  # Bordure plus épaisse
                
                # NOMBRE dans la cellule Tuteurs
                text_t = self.heatmap_hour_scene.addText(str(count_t))
                text_t.setFont(QFont("Arial", 12, QFont.Bold))
                text_width_t = len(str(count_t)) * 8
                text_t.setPos(x_offset + cell_size/2 - text_width_t/2 + 2, y_offset + i * cell_size + cell_size/2 - 10)
                
                # CELLULE APPRENANTS (COLONNE DROITE)
                count_a = hours_a.get(hour, 0)
                intensity_a = min(255, int(220 * count_a / max_count))
                rect_a = self.heatmap_hour_scene.addRect(x_offset + cell_size + column_gap, y_offset + i * cell_size, cell_size, cell_size)
                rect_a.setBrush(QColor(255 - intensity_a, 255 - intensity_a, 255))  # Bleuâtre
                rect_a.setPen(QPen(QColor(80, 80, 80), 2))  # Bordure plus épaisse
                
                # NOMBRE dans la cellule Apprenants
                text_a = self.heatmap_hour_scene.addText(str(count_a))
                text_a.setFont(QFont("Arial", 12, QFont.Bold))
                text_width_a = len(str(count_a)) * 8
                text_a.setPos(x_offset + cell_size + column_gap + cell_size/2 - text_width_a/2 + 2, y_offset + i * cell_size + cell_size/2 - 10)
    def draw_bipartite_graph(self):
        """Dessine un graphe bipartite avancé avec connexions"""
        self.bipartite_scene.clear()
        
        title = self.bipartite_scene.addText("Graphe Bipartite: Tuteurs ↔ Apprenants (Couplages)")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setPos(10, 10)
        
        # Positions
        tuteur_y_step = 60
        apprenant_y_step = 60
        x_tuteur = 50
        x_apprenant = 400
        y_start = 80
        
        tuteur_pos = {}
        apprenant_pos = {}
        
        # Draw tuteurs
        for i, t in enumerate(sorted(self.selected_tuteurs)):
            y = y_start + i * tuteur_y_step
            tuteur_pos[t] = (x_tuteur, y)
            
            rect = self.bipartite_scene.addRect(x_tuteur - 20, y - 15, 40, 30)
            rect.setBrush(QColor(255, 200, 200))  # rouge clair
            rect.setPen(QPen(Qt.black, 1))
            
            text = self.bipartite_scene.addText(t)
            text.setFont(QFont("Arial", 9, QFont.Bold))
            text.setPos(x_tuteur - 12, y - 10)
            text.setDefaultTextColor(Qt.white)
        
        # Draw apprenants
        for i, a in enumerate(sorted(self.selected_apprenants)):
            y = y_start + i * apprenant_y_step
            apprenant_pos[a] = (x_apprenant, y)
            
            rect = self.bipartite_scene.addRect(x_apprenant - 20, y - 15, 40, 30)
            rect.setBrush(QColor(176, 224, 230))  # bleu clair
            rect.setPen(QPen(Qt.black, 1))
            
            text = self.bipartite_scene.addText(a)
            text.setFont(QFont("Arial", 9, QFont.Bold))
            text.setPos(x_apprenant - 12, y - 10)
            text.setDefaultTextColor(Qt.white)
        
        # Draw connections (couplages)
        if self.solution:
            for t in self.solution:
                for (a, _) in self.solution[t]:
                    if t in tuteur_pos and a in apprenant_pos:
                        x1, y1 = tuteur_pos[t]
                        x2, y2 = apprenant_pos[a]
                        line = self.bipartite_scene.addLine(x1 + 20, y1, x2 - 20, y2)
                        line.setPen(QPen(QColor(255, 100, 100), 2, Qt.SolidLine))
                        
                        # Ajouter une flèche ou un label
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        label = self.bipartite_scene.addText(f"✓")
                        label.setDefaultTextColor(QColor(255, 100, 100))
                        label.setFont(QFont("Arial", 10, QFont.Bold))
                        label.setPos(mid_x - 5, mid_y - 10)


        if not self.solution:
            QMessageBox.warning(self, "Erreur", "Exécutez d'abord l'optimisation")
            return
        
        total_couplages = sum(len(app_list) for app_list in self.solution.values())
        if total_couplages == 0:
            QMessageBox.warning(self, "Attention", "Aucun couplage à sauvegarder")
            return
        

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Style de l'application
    app.setStyle("Fusion")
    
    window = CouplageWindow()
    window.show()
    
    sys.exit(app.exec_())