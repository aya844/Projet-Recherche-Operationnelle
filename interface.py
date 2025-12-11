
import sys
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
    QDialog, QTextEdit
)
from PyQt6.QtCore import Qt


class ExplanationWindow(QDialog):
    def __init__(self, project_name, description, launch_function, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Description – {project_name}")
        self.setMinimumWidth(700)

        layout = QVBoxLayout()

        # Title
        title = QLabel(f"Projet : {project_name}")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Description
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setText(description)
        layout.addWidget(desc)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_continue = QPushButton("Continuer vers l'application")
        btn_close = QPushButton("Fermer")

        btn_continue.clicked.connect(lambda: self.start_project(launch_function))
        btn_close.clicked.connect(self.close)

        btn_layout.addWidget(btn_continue)
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def start_project(self, func):
        func()
        self.close()


class MainInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interface Générale – Projets d'Optimisation")
        self.setMinimumSize(600, 500)
        
        
        # Dictionnaire des descriptions de chaque projet
        self.project_descriptions = {
            1: (
                "Projet 1 :Optimisation du transport urgent d’organes pour transplantations\n"
                "\n"
                "Thématique : Ce projet s’inscrit dans la problématique du flux à coût minimum et de la Programmation Linéaire (PL), appliquée au domaine de la logistique. Il porte sur le routage optimal de marchandises à travers plusieurs dépôts afin de minimiser les coûts. Dans ce cas spécifique, le modèle est appliqué à un scénario critique de transport urgent d’organes destinés à la transplantation, combinant des contraintes logistiques strictes, une optimisation multicritère et un modèle de programmation linéaire adapté.\n"
                "Modèle : PL classique avec variables continues.\n"
                "\n"
                "Développé par : Eya Bargouth.\n"
            ),

            2: (
                "Projet 2 : Planification d’un planning d’équipes\n"
                "\n"
                "Objectif : Minimiser les heures supplémentaires.\n"
                "Modèle : PLNE (binaire).\n"
                "Particularités :\n"
                "- Contraintes d’équité et disponibilité.\n"
                "- Visualisation Gantt en sortie.\n"
                "\n"
                "Développé par : Membre 2.\n"
            ),

            3: (
                "Projet 3 : Affectation optimale machines-tâches\n"
                "\n"
                "Objectif : Réduire le temps total d’exécution.\n"
                "Modèle : PLNE.\n"
                "Particularités :\n"
                "- Matrice de compatibilité machines.\n"
                "- Temps d’exécution variables.\n"
                "\n"
                "Développé par : Membre 3.\n"
            ),

            4: (
                "Projet 4 : Optimisation de portefeuille\n"
                "\n"
                "Objectif : Maximiser le rendement sous contraintes de risque.\n"
                "Modèle : PL/PLNE selon type d'actifs.\n"
                "Particularités :\n"
                "- Ajout d'une frontière efficace.\n"
                "- Simulation Monte Carlo optionnelle.\n"
                "\n"
                "Développé par : Membre 4.\n"
            ),

            5: (
                "Projet 5 : Gestion optimale des stocks\n"
                "\n"
                "Objectif : Minimiser le coût total (commande + stockage).\n"
                "Modèle : PL avec contraintes de demande.\n"
                "Particularités :\n"
                "- Intégration d’un modèle EOQ.\n"
                "- Visualisation graphique des niveaux de stock.\n"
                "\n"
                "Développé par : Membre 5.\n"
            ),
        }


        # Dictionnaire des chemins vers vos applications indépendantes
        self.project_scripts = {
            1: "projet1/run.py",
            2: "projet2/run.py",
            3: "projet3/run.py",
            4: "projet4/run.py",
            5: "projet5/run.py",
        }

        layout = QVBoxLayout()

        title = QLabel("Plateforme de Résolution – Recherche Opérationnelle")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Sélectionnez un projet")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Création des 5 boutons
        for i in range(1, 6):
            btn = QPushButton(f"Projet {i}")
            btn.clicked.connect(lambda _, idx=i: self.show_project_popup(idx))
            layout.addWidget(btn)

        self.setLayout(layout)

    def show_project_popup(self, index):
        project_name = f"Projet {index}"

        description = self.project_descriptions.get(index, "Description non disponible.")

        def launch():
            self.launch_project(index)

        popup = ExplanationWindow(project_name, description, launch, self)
        popup.exec()

    def launch_project(self, index):
        """Lance l'application indépendante du projet via subprocess"""
        script = self.project_scripts.get(index)

        if script is None:
            print(f"Erreur : aucun script trouvé pour le projet {index}")
            return

        # Appelle le script Python externe correspondant au projet
        subprocess.Popen([sys.executable, script])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainInterface()
    window.show()
    sys.exit(app.exec())
