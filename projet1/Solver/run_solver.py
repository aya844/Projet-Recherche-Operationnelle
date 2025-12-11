"""
Wrapper script to call the solver and open the GUI results window automatically.

This file is intended as a simple script that can be called from the IHM to run
the solver and open the results display. It uses the Python interpreter used to
invoke this script to run the solver script (`solveur_gurobi.py`) with `--show`.

Usage:
    python Solver/run_solver.py

"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(__file__)
SOLVER_SCRIPT = os.path.join(SCRIPT_DIR, 'solveur_gurobi.py')

if __name__ == '__main__':
    # Call the solver with the show flag so the results window opens automatically
    # Run solver without internal results window; we'll display `GUI/Resultats.py` afterward
    # Validate CSVs before running the solver
    validator = os.path.join(os.path.dirname(__file__), 'validator.py')
    if os.path.exists(validator):
        # try to auto-fix minor CSV typing problems to avoid solver failures
        rcv = subprocess.call([sys.executable, validator, '--dir', os.path.join(os.getcwd(),'data'), '--quiet', '--fix'])
        if rcv != 0:
            print('CSV validation failed; aborting solver run')
            sys.exit(rcv)

    # Mark status as 'running' so IHM/Resultats can display progress
    status_file = os.path.join(os.getcwd(), 'data', 'solver_status.txt')
    try:
        with open(status_file, 'w', encoding='utf-8') as sf:
            sf.write('running')
    except Exception:
        pass

    # Call the solver (without its internal GUI) - validation passed
    cmd = [sys.executable, SOLVER_SCRIPT]
    print("Running solver:", cmd)
    # run and wait for completion; the solver will open a results window if --show
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"Solver exited with code {rc}")
    else:
        print("Solver finished successfully.")
        # After successful run, open the GUI results window automatically (if present)
        gui_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GUI', 'Resultats.py')
        if os.path.exists(gui_script):
            print(f"Launching results GUI: {gui_script}")
            subprocess.Popen([sys.executable, gui_script])
        else:
            print(f"GUI script not found: {gui_script}")
