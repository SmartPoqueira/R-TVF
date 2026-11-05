
import pandas as pd
import numpy as np
import os
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RES_2000 = os.path.join(BASE_DIR, 'src_2000/results_2000')

GS_HADS = os.path.join(RES_2000, '1_grid_search/grid_search_hads.csv')
GS_LADS = os.path.join(RES_2000, '1_grid_search/grid_search_lads.csv')

CV_HADS = os.path.join(RES_2000, '2_cross_validation/cv_results_hads.csv')
CV_LADS = os.path.join(RES_2000, '2_cross_validation/cv_results_lads.csv')

METHODS = ['DHR', 'FS-BSM', 'TBATS', 'ARMAX', 'GP-Celerite', 'R-TVF']
N_FOLDS = 5

def check_file_exists(path, name):
    if not os.path.exists(path):
        print(f"❌ [FAIL] Missing file: {name} ({path})")
        return False
    print(f"✓ [OK] Found {name}")
    return True

def audit_grid_search(path, stype, expected_stars):
    print(f"\n--- Auditing Step 1: Grid Search ({stype}) ---")
    if not check_file_exists(path, f"Grid Search {stype}"): return

    df = pd.read_csv(path)
    
    # 1. Check Stars
    stars = df['StarFile'].unique()
    n_stars = len(stars)
    print(f"  • Stars found: {n_stars} (Expected approx {expected_stars})")
    
    # 2. Check Methods (Should contain at least R-TVF and ARMAX which haveParams)
    methods_found = df['Method'].unique()
    print(f"  • Methods with params: {sorted(methods_found)}")
    
    # 3. Check for empty params
    missing_params = df['Params'].isnull().sum()
    if missing_params > 0:
        print(f"  ❌ [FAIL] Found {missing_params} rows with missing Params!")
    else:
        print(f"  ✓ [OK] All rows have parameters.")

    # 4. Success Rate (R2 > 0)
    success = df[df['R2'] > 0]
    print(f"  • Successful trials (R²>0): {len(success)}/{len(df)} ({len(success)/len(df)*100:.1f}%)")

def audit_cv(path, stype, expected_rows):
    print(f"\n--- Auditing Step 2: Cross Validation ({stype}) ---")
    if not check_file_exists(path, f"CV Results {stype}"): return

    df = pd.read_csv(path)
    
    # 1. Exact Row Count
    if len(df) == expected_rows:
        print(f"  ✓ [OK] Exact row count matched: {len(df)}")
    else:
        print(f"  ❌ [FAIL] Row count mismatch! Expected {expected_rows}, got {len(df)}")
        # Diagnostics
        counts = df.groupby('Method')['StarFile'].count()
        print(f"     Detailed counts per method:\n{counts}")

    # 2. Duplicates
    dupes = df.duplicated(subset=['StarFile', 'Method', 'Fold'])
    if dupes.any():
        print(f"  ❌ [FAIL] Found {dupes.sum()} duplicates!")
    else:
        print(f"  ✓ [OK] No duplicates found.")

    # 3. Method Coverage
    methods_present = sorted(df['Method'].unique())
    print(f"  • Methods present: {methods_present}")
    if len(methods_present) != 6:
        print(f"  ❌ [FAIL] Missing methods! Expected 6, got {len(methods_present)}")

    # 4. Repair Quality (Outliers)
    print("  • Checking Repair Quality:")
    
    # ARMAX Massive Negatives
    armax_fail = df[(df['Method'] == 'ARMAX') & (df['R2'] < -10)]
    if len(armax_fail) > 0:
        print(f"     ⚠ [WARN] Found {len(armax_fail)} ARMAX rows with R² < -10 (Potential unfixable outliers)")
        print(armax_fail[['StarFile', 'Fold', 'R2']].to_string(index=False))
    else:
        print(f"     ✓ [OK] ARMAX clean (No massive negatives).")

    # R-TVF Weak Positives / Negatives
    rtvf_fail = df[(df['Method'] == 'R-TVF') & (df['R2'] < 0.1)]
    if len(rtvf_fail) > 0:
        print(f"     ℹ [INFO] Found {len(rtvf_fail)} R-TVF rows < 0.1 (Acceptable if R² > 0)")
        print(rtvf_fail[['StarFile', 'Fold', 'R2']].to_string(index=False))
        
        neg_rtvf = rtvf_fail[rtvf_fail['R2'] < 0]
        if not neg_rtvf.empty:
            print(f"     ❌ [FAIL] Found NEGATIVE R-TVF R²! Repair Failed.")
        else:
            print(f"     ✓ [OK] All weak R-TVF are positive (valid).")
    else:
        print(f"     ✓ [OK] R-TVF clean (All > 0.1).")


if __name__ == "__main__":
    print("==================================================")
    print("       AUDIT REPORT: PIPELINE STEPS 1 & 2")
    print("==================================================")
    
    # HADS (33 stars)
    audit_grid_search(GS_HADS, 'HADS', 33)
    # 33 stars * 6 methods * 5 folds = 990
    audit_cv(CV_HADS, 'HADS', 990)

    print("\n" + "="*50)

    # LADS (39 stars)
    audit_grid_search(GS_LADS, 'LADS', 39)
    # 39 stars * 6 methods * 5 folds = 1170
    audit_cv(CV_LADS, 'LADS', 1170)
    
    print("\n==================================================")
