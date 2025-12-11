# Optimisation de Couplage Maximum - Projet RO

## Description
Cette application constitue un projet de **Recherche Opérationnelle (RO)** visant à résoudre le problème du **Couplage Maximum** à l'aide de techniques de **Programmation Linéaire en Nombres Entiers (PLNE)**. L'objectif est de former des paires optimales entre **tuteurs et apprenants**, en respectant les contraintes de compatibilité (domaines d'expertise, jours et heures disponibles).

L'application a été entièrement développée en **Python** avec une interface graphique conviviale utilisant **PyQt/PySide** et le solveur **Gurobi** pour la résolution des problèmes d'optimisation.

---

## Fonctionnalités

- **Saisie des données** : Entrée et modification des informations des tuteurs et apprenants via l'interface.
- **Optimisation** : Calcul du couplage maximum respectant toutes les contraintes (domaines, disponibilités, etc.).
- **Visualisation des résultats** : Affichage des paires optimales avec détails des jours et heures communs ainsi que des domaines.
- **Multithreading** : L'application reste réactive même pour de grandes instances PLNE.
- **Annulation de l'optimisation** : Possibilité d'interrompre un calcul en cours.

---

## Capture d'écran de l'application

Voici un exemple montrant les paires tuteur/apprenant générées après l'optimisation :

![Capture de l'application](capture_resultat.png)

---

## Structure du projet

RO_Couplage_Maximum/
│
├─ tuteur_apprenant.py # Fichier Python unique contenant tout le code
├─ capture_resultat.png # Capture d'écran des résultats
├─ README.md # Ce fichier
├─ remarques_Projet_RO.docx # Notes de projet (facultatif, non suivi dans GitHub)
├─ solution.xlsx # Exemple de données
└─ requirements.txt # Dépendances Python

yaml
Copier le code

---

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/zemni31/RO_Couplage_Maximum.git
cd RO_Couplage_Maximum
Installer les dépendances :

bash
Copier le code
pip install -r requirements.txt
Lancer l'application :

bash
Copier le code
python tuteur_apprenant.py
Dépendances principales
Python >= 3.10

PyQt5 ou PySide2

Gurobi

matplotlib (pour la visualisation graphique)

pandas, numpy (gestion de tableaux et données)
