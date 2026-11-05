import pandas as pd
import sys
import os

def check_csv(path, expected_rows, name):
    print(f"Auditing {name} ({path})...")
    if not os.path.exists(path):
        print(f"FAILED: File not found: {path}")
        return False
        
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"FAILED: Could not read file. {e}")
        return False

    # 1. Count
    if len(df) != expected_rows:
        print(f"FAILED: Expected {expected_rows} rows, got {len(df)}.")
        return False
    print(f"✓ Count: {len(df)} (Correct)")

    # 2. Duplicates
    dupes = df.duplicated(subset=['StarFile', 'Method', 'Fold'])
    if dupes.any():
        print(f"FAILED: Found {dupes.sum()} duplicates.")
        return False
    print("✓ Duplicates: None")

    # 3. Outliers (Strict check)
    # R-TVF < 0.1
    rtvf_issues = df[(df['Method'] == 'R-TVF') & (df['R2'] < 0.1)]
    if not rtvf_issues.empty:
        print(f"⚠ WARNING: Found {len(rtvf_issues)} R-TVF results < 0.1.")
        print(rtvf_issues[['StarFile', 'Fold', 'R2']].to_string(index=False))
    else:
        print("✓ R-TVF: No results < 0.1")

    # ARMAX < -100
    armax_issues = df[(df['Method'] == 'ARMAX') & (df['R2'] < -100)]
    if not armax_issues.empty:
        print(f"⚠ WARNING: Found {len(armax_issues)} ARMAX results < -100.")
    else:
        print("✓ ARMAX: No massive errors (< -100)")
    
    return True

print("=== FINAL AUDIT (src_2000) ===")
hads_ok = check_csv('src_2000/results_2000/2_cross_validation/cv_results_hads.csv', 990, 'HADS')
print("-" * 20)
lads_ok = check_csv('src_2000/results_2000/2_cross_validation/cv_results_lads.csv', 1170, 'LADS')
print("=" * 20)

if hads_ok and lads_ok:
    print("AUDIT STATUS: PASS")
else:
    print("AUDIT STATUS: FAIL")
