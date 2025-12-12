from Solver.solveur_gurobi import read_inputs

if __name__ == '__main__':
    dn, da, params = read_inputs()
    print('nodes:', dn.shape)
    print('arcs:', da.shape)
    print('params sample:', params)
