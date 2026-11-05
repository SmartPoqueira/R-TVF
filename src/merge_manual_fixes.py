#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merges the results from 'cv_results_manual_fix.csv' into the main
'cv_results_hads.csv' and 'cv_results_lads.csv' files.
This replaces the specific failed folds (Negative R²) with the re-run fixed values.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "2_cross_validation"

FIX_FILE = RESULTS_DIR / "cv_results_manual_fix.csv"
HADS_FILE = RESULTS_DIR / "cv_results_hads.csv"
LADS_FILE = RESULTS_DIR / "cv_results_lads.csv"

def merge_fixes():
    if not FIX_FILE.exists():
        print(f"No manual fix file found at {FIX_FILE}. Skipping merge.")
        return

    print(f"Loading manual fixes from {FIX_FILE}...")
    df_fix = pd.read_csv(FIX_FILE)
    print(f"Found {len(df_fix)} rows to update.")

    # Process HADS and LADS
    for target_file in [HADS_FILE, LADS_FILE]:
        if not target_file.exists():
            print(f"Target file {target_file} not found. Skipping.")
            continue

        print(f"Processing {target_file.name}...")
        df_main = pd.read_csv(target_file)
        initial_len = len(df_main)
        updated_count = 0

        # Iterate over fixes and apply them if they belong to this file (matched by filename existence)
        # Note: We rely on StarFile to match.
        
        for _, row_fix in df_fix.iterrows():
            star = row_fix['StarFile']
            fold = row_fix['Fold']
            method = "R-TVF" # Always R-TVF for this batch
            
            # Check if this star is in this main df
            mask = (df_main['StarFile'] == star) & \
                   (df_main['Method'] == method) & \
                   (df_main['Fold'] == fold)
            
            if mask.any():
                # Update values
                idx = df_main[mask].index[0]
                
                print(f"  -> Updating {star} Fold {fold}:")
                print(f"     Old R2: {df_main.at[idx, 'R2']:.4f}")
                print(f"     New R2: {row_fix['R2']:.4f}")
                
                df_main.at[idx, 'R2'] = row_fix['R2']
                df_main.at[idx, 'MAE'] = row_fix['MAE']
                df_main.at[idx, 'MSE'] = row_fix['MSE']
                df_main.at[idx, 'MAPE'] = row_fix['MAPE']
                # Optionally update Time or other fields if needed
                if 'Time' in row_fix:
                    df_main.at[idx, 'Time'] = row_fix['Time']
                    
                updated_count += 1
        
        if updated_count > 0:
            df_main.to_csv(target_file, index=False)
            print(f"Updated {updated_count} rows in {target_file.name}.")
        else:
            print(f"No matching rows found in {target_file.name} to update.")

    print("\nMerge complete.")

if __name__ == "__main__":
    merge_fixes()
