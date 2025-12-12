import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
main_file = os.path.join(project_dir, "main.py")

os.system(f"{sys.executable} \"{main_file}\"")
