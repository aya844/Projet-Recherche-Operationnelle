from Solver.solveur_gurobi import read_inputs

if __name__ == '__main__':
    try:
        dn, da, params = read_inputs()
        print('nodes shapes:', dn.shape)
        print('arcs shapes:', da.shape)
        print('params:', params)
    except Exception as e:
        print('Error in read_inputs:', e)
