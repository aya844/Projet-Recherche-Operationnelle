# Optimisation Logistique du Transport d'Organes

## À propos du projet

Ce projet s’inscrit dans la thématique **"Flux à Coût Minimum"** et utilise la **Programmation Linéaire (PL)** pour optimiser le transport rapide d’organes pour transplantations.  
L’objectif est de fournir un **outil d’aide à la décision logistique** capable de planifier le routage des organes de manière efficace, en respectant les délais critiques et en minimisant les coûts.

---

## Contexte

- Les organes doivent être transportés dans une **fenêtre de temps très courte** pour rester viables.
- Les flux sont **peu volumineux mais prioritaires**.
- Le réseau comprend trois types de nœuds :
  - **Sources** : Hôpitaux donneurs
  - **Nœuds intermédiaires** : Centres de tri ou dépôts spécialisés
  - **Destinations** : Centres de chirurgie / blocs opératoires

---

## Objectifs

- Livrer les organes **dans les délais impartis**.
- **Minimiser le coût total** du transport.
- Optimiser l’usage des capacités existantes et décider de l’**achat éventuel de ressources supplémentaires** (ambulances, vols prioritaires).

---

## Technologies

- **Python** : Langage principal
- **PyQt6** : Interface graphique
- **Gurobi** : Solveur de programmation linéaire
- **Pandas** : Gestion et traitement des fichiers CSV

---

## Fonctionnalités

- Saisie et gestion des nœuds et arcs du réseau (offre/demande, coûts, temps, capacités, coût d’augmentation)
 - Saisie et gestion des nœuds et arcs du réseau (offre/demande, coûts, temps, capacités, coût d’augmentation)
 - Support multi-type d'organes (colonnes supply_<organ>, cost_<organ>, time_<organ>) et véhicules entiers achetés (param `vehicle_capacity` + `cap_cost` par véhicule)
- Définition des paramètres alpha et beta pour pondérer coût et temps
- Validation des données avant optimisation
- Résolution de la PL et affichage :
  - Flux optimaux
  - Capacités additionnelles nécessaires
- Export et sauvegarde des résultats au format CSV

---

## Visualisation

- Graphique du réseau avec **flèches proportionnelles aux flux**
- Coloration des arcs selon **coût ou temps**
- Tableau récapitulatif des flux et ressources additionnelles

---

## Installation

```bash
git clone https://github.com/votre-utilisateur/transport-organes.git
pip install -r requirements.txt
python main.py
```

---

## Utilisation

1. Ajouter les nœuds et arcs via l’interface graphique
2. Définir les paramètres alpha et beta
3. Cliquer sur **"Résoudre"** pour obtenir les flux optimaux
4. Visualiser et exporter les résultats

---

## Bénéfices

- **Augmente le taux de succès des transplantations**
- **Réduit les coûts logistiques inutiles**
- **Interface intuitive** pour les opérateurs hospitaliers

---

## Auteur

- Eya Bargouth

---

