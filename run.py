import sys
import os

# Récupère le chemin absolu du dossier du projet
project_dir = os.path.dirname(os.path.abspath(__file__))

# Définir main.py comme fichier principal à lancer
main_file = os.path.join(project_dir, "main.py")

# Lancer le vrai main.py du projet
os.system(f"{sys.executable} \"{main_file}\"")