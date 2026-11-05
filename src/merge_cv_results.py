#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to MERGE the new optimized R-TVF results into the main CV result files.

1. Backs up existing 'cv_results_hads.csv' and 'cv_results_lads.csv'.
2. Loads the new 'cv_results_rtvf_optimized.csv'.
3. Removes old R-TVF rows from the main files.
4. Injects the new R-TVF rows.
5. Saves the updated files.
"""

import pandas as pd
import shutil
from pathlib import Path
import sys

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PARENT_DIR / "results" / "2_cross_validation"

FILE_HADS = RESULTS_DIR / "cv_results_hads.csv"
FILE_LADS = RESULTS_DIR / "cv_results_lads.csv"
FILE_NEW = RESULTS_DIR / "cv_results_rtvf_optimized.csv"

def merge_results():
    if not FILE_NEW.exists():
        print(f"Error: New results file not found: {FILE_NEW}")
        sys.exit(1)

    print(f"Loading new results from: {FILE_NEW.name}")
    df_new = pd.read_csv(FILE_NEW)
    print(f"  -> Total new rows: {len(df_new)}")

    # Process HADS and LADS separately
    for file_path, star_type in [(FILE_HADS, "HADS"), (FILE_LADS, "LADS")]:
        if not file_path.exists():
            print(f"Warning: {file_path.name} not found. Skipping.")
            continue

        print(f"\nProcessing {star_type}...")
        
        # 1. Backup
        backup_path = file_path.with_suffix('.csv.bak')
        shutil.copy(file_path, backup_path)
        print(f"  -> Backup created: {backup_path.name}")

        # 2. Load Original
        df_old = pd.read_csv(file_path)
        print(f"  -> Original rows: {len(df_old)}")

        # 3. Filter New Data for this StarType
        df_new_filtered = df_new[df_new['StarType'] == star_type]
        print(f"  -> New R-TVF rows to inject: {len(df_new_filtered)}")

        # 4. Remove OLD R-TVF rows
        df_merged = df_old[df_old['Method'] != 'R-TVF']
        rows_removed = len(df_old) - len(df_merged)
        print(f"  -> Removed {rows_removed} old R-TVF rows")

        # 5. Append NEW R-TVF rows
        df_merged = pd.concat([df_merged, df_new_filtered], ignore_index=True)
        print(f"  -> Final rows: {len(df_merged)}")

        # 6. Save
        df_merged.to_csv(file_path, index=False)
        print(f"  -> Saved updated file: {file_path.name}")

    print("\nMerge completed successfully for both HADS and LADS.")

if __name__ == "__main__":
    merge_results()
