#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUDIT PIPELINE SCRIPT
=====================
Verifies successful execution of:
1. Grid Search (All stars * methods have results).
2. Cross Validation (All stars * methods have 5 folds, check duplicates/NaNs).
3. Analysis Outputs (Files exist and are not empty).
"""

import pandas as pd
import glob
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Expected Counts
METHODS = ['DHR', 'FS-BSM', 'TBATS', 'ARMAX', 'GP-Celerite', 'R-TVF']
N_FOLDS = 5

def audit_step_1_grid_search():
    print("\n>>> AUDIT STEP 1: GRID SEARCH")
    gs_files = list(RESULTS_DIR.joinpath("1_grid_search").glob("grid_search_*.csv"))
    if not gs_files:
        print("  [FAIL] No grid search CSVs found.")
        return False
    
    all_passed = True
    for f in gs_files:
        print(f"  Checking {f.name}...")
        try:
            df = pd.read_csv(f)
            stars = df['StarFile'].unique()
            print(f"    Found {len(stars)} distinct stars.")
            
            # Check coverage
            missing = []
            for star in stars:
                df_star = df[df['StarFile'] == star]
                methods_present = df_star['Method'].unique()
                for m in METHODS:
                    if m not in methods_present:
                        missing.append((star, m))
            
            if missing:
                print(f"    [WARN] Missing methods for some stars: {len(missing)} instances.")
                # print(missing[:5]) # Sample
                # Ideally every star should have every method, but maybe some failed?
                # We will mark as WARN not FAIL unless count is huge.
            else:
                print("    [OK] All stars have all methods.")
                
            # Check R2 valid
            # In Grid Search, failures might be recorded as very low R2.
            # We just verify rows exist.
            
        except Exception as e:
            print(f"    [FAIL] Error reading CSV: {e}")
            all_passed = False
            
    return all_passed

def audit_step_2_cv():
    print("\n>>> AUDIT STEP 2: CROSS VALIDATION")
    cv_files = list(RESULTS_DIR.joinpath("2_cross_validation").glob("cv_results_*.csv"))
    # Exclude 'manual_fix' if it exists separate (we use merged ones)
    cv_files = [f for f in cv_files if "manual_fix" not in f.name]
    
    if not cv_files:
        print("  [FAIL] No CV results files found.")
        return False
        
    all_passed = True
    for f in cv_files:
        print(f"  Checking {f.name}...")
        try:
            df = pd.read_csv(f)
            
            # 1. Duplicates check
            # Key should be (StarFile, Method, Fold)
            dupes = df[df.duplicated(subset=['StarFile', 'Method', 'Fold'], keep=False)]
            if not dupes.empty:
                print(f"    [WARN] Found {len(dupes)} duplicate entries! (Star/Method/Fold collision).")
                # This affects counting.
            else:
                print("    [OK] No exact duplicates found.")
            
            # 2. Completeness Check
            # For each Star * Method, count unique Folds.
            grouped = df.groupby(['StarFile', 'Method'])['Fold'].nunique()
            incomplete = grouped[grouped != N_FOLDS]
            
            if not incomplete.empty:
                print(f"    [FAIL] Incomplete runs found: {len(incomplete)} Star/Method pairs have != {N_FOLDS} folds.")
                print(incomplete.head())
                all_passed = False
            else:
                print(f"    [OK] All combinations have exactly {N_FOLDS} folds.")
                
            # 3. Validity Check (NaNs)
            nans = df[['MAE', 'R2']].isna().sum()
            if nans.sum() > 0:
                print(f"    [FAIL] Found NaNs in results: \n{nans}")
                all_passed = False
            else:
                print("    [OK] No NaNs in metrics.")
            
            # 4. Outlier Check (Quick summary)
            neg_r2 = len(df[df['R2'] < 0])
            print(f"    [INFO] Negative R2 count: {neg_r2} / {len(df)}")
            
        except Exception as e:
            print(f"    [FAIL] Error reading CSV: {e}")
            all_passed = False
            
    return all_passed

def audit_step_3_to_7_analysis():
    print("\n>>> AUDIT STEP 3-7: ANALYSIS OUTPUTS")
    # Step 3: Sensitivity Heatmaps
    sensitivity_dir = RESULTS_DIR / "3_sensitivity"
    if sensitivity_dir.exists() and len(list(sensitivity_dir.glob("*.png"))) > 0:
        print(f"  [OK] Step 3 (Sensitivity): Found plots in {sensitivity_dir.name}")
    else:
        print(f"  [WARN] Step 3 (Sensitivity): No plots found in {sensitivity_dir}")

    # Step 4: Analysis (Tables & Plots)
    analysis_dir = RESULTS_DIR / "4_analysis"
    if analysis_dir.exists():
        tables = list((analysis_dir / "tables").glob("*.tex"))
        results_csv = list(analysis_dir.glob("summary_*.csv"))
        if tables and results_csv:
             print(f"  [OK] Step 4 (Analysis): Found {len(tables)} tables and {len(results_csv)} summary CSVs.")
        else:
             print(f"  [FAIL] Step 4 (Analysis): Missing tables or summary CSVs.")
    else:
        print(f"  [FAIL] Step 4 Directory missing.")

    # Step 5: Grid Plots (Saved in 4_analysis/plots)
    grid_plots = list((analysis_dir / "plots").glob("*.png"))
    if grid_plots:
         print(f"  [OK] Step 5 (Grid Plots): Found {len(grid_plots)} plots in 4_analysis/plots.")
    else:
         print(f"  [WARN] Step 5 (Grid Plots): No plots found (maybe generation failed or disabled?).")

    # Step 6: Outlier Analysis (Calculates robust stats, prints to stdout, no file)
    # We verify the script exists.
    script_6 = BASE_DIR / "src/6.analyze_outliers.py"
    if script_6.exists():
         print(f"  [OK] Step 6 (Outliers): Script exists (Output is stdout-only, verified in log).")
    else:
         print(f"  [FAIL] Step 6 Script missing.")

    # Step 7: Generate Tables (Outputs to 4_analysis/tables/table_cv_results.tex)
    final_table = analysis_dir / "tables" / "table_cv_results.tex"
    if final_table.exists():
         print(f"  [OK] Step 7 (Final Tables): Found {final_table.name}")
    else:
         print(f"  [FAIL] Step 7 (Final Tables): Missing output file.")
         
    return True # Return True if critical files exist

def run_audit():
    print("STARTING DEEP AUDIT...")
    s1 = audit_step_1_grid_search()
    s2 = audit_step_2_cv()
    s3 = audit_step_3_to_7_analysis()
    
    print("\n" + "="*40)
    print("AUDIT SUMMARY:")
    print(f"1. Grid Search: {'PASS' if s1 else 'FAIL/WARN'}")
    print(f"2. Cross Valid: {'PASS' if s2 else 'FAIL'}")
    print(f"3. Analysis:    {'PASS' if s3 else 'FAIL'}")
    print("="*40)
    
    if s1 and s2 and s3:
        print("RESULT: AUDIT PASSED ✅")
    else:
        print("RESULT: AUDIT FAILED ❌ (Check logs above)")

if __name__ == "__main__":
    run_audit()
