"""
Validator CLI for CSV files produced by the IHM.

This script checks that files `data/nodes.csv`, `data/arcs.csv`, `data/params.csv`:
 - exist
 - have the expected columns
 - have no empty cells for required columns
 - numeric fields can be parsed

Usage:
    python Solver/validator.py
    python Solver/validator.py --dir data
    python Solver/validator.py --quiet

Exit code 0 on success, 1 on errors.
"""
import argparse
import os
import sys
import pandas as pd

def _read_csv_with_fallback(path):
    encodings = ['utf-8', 'cp1252', 'latin-1']
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_exc = e
    # if all encodings fail, raise the last exception
    raise last_exc


def validate(dirpath='data', fix=False):
    errors = []
    nodes_file = os.path.join(dirpath, 'nodes.csv')
    arcs_file = os.path.join(dirpath, 'arcs.csv')
    params_file = os.path.join(dirpath, 'params.csv')

    if not os.path.exists(nodes_file):
        errors.append(f"Missing {nodes_file}")
        return errors
    if not os.path.exists(arcs_file):
        errors.append(f"Missing {arcs_file}")
        return errors
    # read with encoding fallback
    try:
        df_nodes = _read_csv_with_fallback(nodes_file)
    except Exception as e:
        errors.append(f"cannot read nodes.csv: {e}")
        return errors
    try:
        df_arcs = _read_csv_with_fallback(arcs_file)
    except Exception as e:
        errors.append(f"cannot read arcs.csv: {e}")
        return errors
    # check columns
    req_nodes = ['id','type','name','supply']
    if not set(req_nodes).issubset(df_nodes.columns):
        errors.append(f"nodes.csv missing columns; expected {req_nodes} got {list(df_nodes.columns)}")
    req_arcs = ['origin','dest','cost','time','capacity','cap_cost']
    if not set(req_arcs).issubset(df_arcs.columns):
        errors.append(f"arcs.csv missing columns; expected {req_arcs} got {list(df_arcs.columns)}")

    # nodes rows
    for idx, row in df_nodes.iterrows():
        for col in ['id','type','name']:
            val = str(row.get(col, '')).strip()
            if val == '' or pd.isna(val):
                errors.append(f"nodes.csv line {idx+1}: {col} empty")
        # supply numeric
        try:
            _ = float(row['supply'])
        except Exception:
            s = str(row.get('supply','')).strip().lower()
            if s in ('offre','o','supply'):
                if fix:
                    df_nodes.loc[idx,'supply'] = 1.0
                else:
                    errors.append(f"nodes.csv line {idx+1}: supply label '{row.get('supply')}' should be numeric or use --fix")
            elif s in ('demande','d','demand'):
                if fix:
                    df_nodes.loc[idx,'supply'] = -1.0
                else:
                    errors.append(f"nodes.csv line {idx+1}: supply label '{row.get('supply')}' should be numeric or use --fix")
            else:
                try:
                    # attempt to remove thousand separators
                    df_nodes.loc[idx,'supply'] = float(str(row['supply']).replace(',',''))
                except Exception:
                    errors.append(f"nodes.csv line {idx+1}: supply not numeric: '{row.get('supply')}'")

    # arcs rows
    for idx, row in df_arcs.iterrows():
        for col in ['origin','dest']:
            val = str(row.get(col, '')).strip()
            if val == '' or pd.isna(val):
                errors.append(f"arcs.csv line {idx+1}: {col} empty")
        for col in ['cost','time','capacity','cap_cost']:
            try:
                _ = float(row[col])
            except Exception:
                # try to coerce
                try:
                    df_arcs.loc[idx,col] = float(str(row.get(col,'')).replace(',',''))
                except Exception:
                    if fix:
                        df_arcs.loc[idx,col] = 0.0
                    else:
                        errors.append(f"arcs.csv line {idx+1}: {col} not numeric: '{row.get(col)}'")

    # params
    if os.path.exists(params_file):
        try:
            dfp = _read_csv_with_fallback(params_file)
        except Exception:
            try:
                dfp = _read_csv_with_fallback(params_file)
            except Exception:
                errors.append(f"params.csv cannot be read as CSV")
                return errors
    else:
        # not mandatory
        pass

    # If fix option was used and we changed frames, write them back
    if fix and len(errors) == 0:
        # if the frames were modified, write back
        df_nodes.to_csv(nodes_file, index=False, encoding='utf-8')
        df_arcs.to_csv(arcs_file, index=False, encoding='utf-8')
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='data')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--fix', action='store_true', help='Try to coerce common errors and rewrite CSVs')
    args = parser.parse_args(argv)
    err = validate(args.dir, fix=args.fix)
    if err:
        if not args.quiet:
            print("Validation errors:")
            for e in err:
                print(" - ", e)
        sys.exit(1)
    else:
        if not args.quiet:
            print("CSV validation: OK")
        sys.exit(0)


if __name__ == '__main__':
    main()
