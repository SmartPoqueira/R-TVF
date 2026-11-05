#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   2. CROSS VALIDATION - 5-Fold CV (Q1 JCR Methodology)                       ║
║   Delta Scuti Stars Missing Values Imputation                                ║
║   6 Métodos: ARMAX, GP-Celerite, DHR, FS-BSM, TBATS, R-TVF          ║
╚══════════════════════════════════════════════════════════════════════════════╝

METHODOLOGY (Standard Q1 JCR):
- Fixed seeds ensuring 100% reproducibility
- Each fold uses a DIFFERENT block position (simulating different observation gaps)
- Same hyperparameters for all folds (the best from grid search)
- Metrics: MAE, MSE, MAPE, R² per fold → mean ± std reported

This script:
1. Reads grid search results (grid_search_hads.csv, grid_search_lads.csv - now includes GP-Celerite)
2. Finds the best configuration for each (star, method) by maximizing R²
3. Runs 5-fold CV with that configuration (5 different block positions)
4. Saves per-fold results to cross_validation_results.csv

Usage:
    python 2.cross_validation.py [--star-type HADS|LADS]
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import warnings

# ============= PATHS =============
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
DATA_DIR = PARENT_DIR / "estrellas_test"
RESULTS_DIR = PARENT_DIR / "results" / "2_cross_validation"
GRID_SEARCH_DIR = PARENT_DIR / "results" / "1_grid_search"
PLOTS_DIR_HADS = RESULTS_DIR / "plots_hads"
PLOTS_DIR_LADS = RESULTS_DIR / "plots_lads"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR_HADS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR_LADS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PARENT_DIR))

# ============= IMPORTS =============
try:
    import lightCurve as lc
    import modelos_hads_lads as mh
    from runner_comparativa_3x2 import (
        load_dat, simulate_block_run, metrics_from_block,
        run_gp_imputation
    )
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# GP imports
try:
    import celerite2
    from celerite2 import terms
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    print("Warning: celerite2 not available. GP-Celerite will be skipped.")

warnings.filterwarnings("ignore")

# ============= CONFIGURATION =============
# FIXED SEEDS for 100% reproducibility (Q1 JCR standard)
# Each fold uses a deterministic seed: SEED_BASE + fold_index * 1000
SEED_BASE = 42
N_FOLDS = 5
NBLOCK = 1000

# Pre-computed fold seeds (deterministic for reproducibility)
FOLD_SEEDS = [SEED_BASE + i * 1000 for i in range(N_FOLDS)]
# Results: [42, 1042, 2042, 3042, 4042]

DATA_DIRS = {
    "HADS": DATA_DIR / "hads",
    "LADS": DATA_DIR / "lads"
}

FIXED_DL_PARAMS = {
    "window_size": 512,
    "batch_size": 32,
    "epochs": 200,
    "patience": 30
}

# ============= DETERMINISTIC SETUP =============
def set_global_seed(seed):
    """Set all random generators for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except: pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except: pass

# ============= LOAD BEST CONFIGURATIONS FROM CSV =============
def load_best_configs():
    """
    Load grid search CSVs and find best config per (star, method) by max R².
    """
    csv_files = [
        GRID_SEARCH_DIR / "grid_search_results.csv"
        # GP-Celerite results are now included in the above CSVs
    ]
    
    dfs = []
    for csv_path in csv_files:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"  ✓ {csv_path.name}: {len(df)} rows")
    
    if not dfs:
        raise FileNotFoundError("No grid search CSVs found!")
    
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Find best config per (star_type, star_file, method) by max R²
    best_configs = {}
    for (star_type, star_file, method), group in df_all.groupby(['StarType', 'StarFile', 'Method']):
        best_idx = group['R2'].idxmax()
        best_row = group.loc[best_idx]
        best_configs[(star_type, star_file, method)] = {
            'params': best_row['Params'],
            'best_r2': best_row['R2'],
            'best_mae': best_row['MAE']
        }
    
    print(f"  → Total: {len(df_all)} rows, {df_all.StarFile.nunique()} stars")
    print(f"  → Best configs loaded: {len(best_configs)}")
    return best_configs, df_all

# ============= CHECKPOINT =============
def load_completed_runs(csv_path):
    """Load completed runs for resume capability."""
    completed = set()
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                key = (row['StarType'], row['StarFile'], row['Method'], int(row['Fold']))
                completed.add(key)
        except: pass
    return completed

# ============= PLOT FUNCTION (same as 1.grid_search.py) =============
def save_imputation_plot(t, y_true, y_imp, start, end, star_name, method, fold, metrics, out_dir, y_std=None):
    """Save imputation plot for visual inspection, with optional uncertainty bands."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    pad = 200
    left = max(0, start - pad)
    right = min(len(t), end + pad)
    
    ax.plot(t[left:right], y_true[left:right], 'k.', alpha=0.3, markersize=2, label='Original', zorder=1)
    ax.plot(t[start:end], y_true[start:end], 'b-', alpha=0.6, linewidth=1, label='Ground truth', zorder=3)
    ax.plot(t[start:end], y_imp[start:end], 'r-', linewidth=1.5, label=f'{method}', zorder=4)
    if y_std is not None:
        ax.fill_between(t[start:end], y_imp[start:end] - 3*y_std[start:end], y_imp[start:end] + 3*y_std[start:end], color='#FFB6C1', alpha=1.0, zorder=2, label=f'3σ Band')
    ax.axvspan(t[start], t[end-1], color='yellow', alpha=0.15, zorder=0)
    
    ax.set_xlabel('Time (HJD)')
    ax.set_ylabel('Flux')
    ax.set_title(f"{star_name} - {method} - Fold {fold+1}\nMAE={metrics['MAE']:.3f} | R²={metrics['R2']:.3f}")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    safe_star = star_name.replace(' ', '_').replace('.dat', '')
    filename = f"{safe_star}_{method}_fold{fold}.png"
    filepath = out_dir / filename
    
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath

# ============= RUN SINGLE METHOD =============
def run_single_method(method, params_dict, t, y_true, y_missing, start, end, mask_block, star_type):
    y_imp = None
    y_std = None
    
    # Determine frequency band based on star_type
    if star_type == "HADS":
        fmin, fmax = 3.0, 30.0
    else:
        fmin, fmax = 5.0, 80.0
    
    if method == "GP-Celerite":
        gp_params = {
            "tau": float(params_dict.get("gp_tau", 50.0)),
            "noise": float(params_dict.get("gp_noise", 1e-3))
        }
        y_imp = run_gp_imputation(t, y_missing, start, end, gp_params)
        if y_imp is not None:
            y_imp = np.asarray(y_imp, float)
        
    elif method == "ARMAX":
        freqs = mh.ls_top_freqs(t[np.isfinite(y_missing)], y_missing[np.isfinite(y_missing)], M=int(params_dict.get("fourier_K", 3)), fmin=fmin, fmax=fmax)
        y_imp, _ = mh.armax_impute_block_multi(
            t, y_missing, start, end, freqs_cpd=freqs,
            p=int(params_dict.get("arma_p", 3)), q=int(params_dict.get("arma_q", 5)), trend="c"
        )
        y_imp = np.asarray(y_imp, float)
        
    elif method == "R-TVF":
        l2_val = float(params_dict.get("l2", 1e-2))
        y_imp, _, y_std = mh.tvar_fourier_impute_block_multi(
            t, y_missing, start, end,
            M=int(params_dict.get("fourier_K", 5)), degree=int(params_dict.get("degree", 3)),
            trend_deg=1, l2=l2_val,
            fmin=fmin, fmax=fmax, max_samples_ls=60000,
            return_std=True
        )
        y_imp = np.asarray(y_imp, float)
        y_std = np.asarray(y_std, float)
        
    elif method == "DHR":
        y_imp = mh.dhr_impute_block(t, y_missing, start, end,
                                    fmin=fmin, fmax=fmax, M=int(params_dict.get("fourier_K", 3)),
                                    p_ar=int(params_dict.get("p_ar", 2)),
                                    maxiter=20)
        y_imp = np.asarray(y_imp, float)
        
    elif method == "FS-BSM":
        y_imp = mh.trig_ss_impute_block(t, y_missing, start, end,
                                        fmin=fmin, fmax=fmax, M=int(params_dict.get("fourier_K", 3)),
                                        maxiter=20)
        y_imp = np.asarray(y_imp, float)
        
    elif method == "TBATS":
        y_imp = mh.tbats_ss_impute_block(t, y_missing, start, end,
                                         fmin=fmin, fmax=fmax, M=int(params_dict.get("fourier_K", 3)),
                                         p_ar=int(params_dict.get("p_ar", 1)),
                                         maxiter=500)
        y_imp = np.asarray(y_imp, float)
    
    return y_imp, y_std

# ============= CROSS VALIDATION (Q1 JCR) =============
def run_cross_validation(star_type_filter=None):
    """
    5-Fold Cross Validation following Q1 JCR methodology.
    
    For each fold:
    - Use deterministic seed (FOLD_SEEDS[fold])
    - Generate block at different position
    - Evaluate with best hyperparameters from grid search
    
    All seeds are fixed, ensuring 100% reproducibility.
    """
    print("\n" + "="*70)
    print("  2. CROSS VALIDATION - 5-Fold CV (Q1 JCR Methodology)")
    print("="*70)
    print(f"  N_FOLDS: {N_FOLDS}")
    print(f"  FOLD_SEEDS: {FOLD_SEEDS}")
    print(f"  BLOCK_SIZE: {NBLOCK}")
    print("="*70 + "\n")
    
    # Set initial seed
    set_global_seed(SEED_BASE)
    
    # Load best configs
    print("Loading best configurations from grid search...")
    best_configs, df_grid = load_best_configs()
    
    # Output file - SEPARATE for parallel execution
    if star_type_filter:
        out_csv = RESULTS_DIR / f"cv_results_{star_type_filter.lower()}.csv"
    else:
        out_csv = RESULTS_DIR / "cv_results_all.csv"
    
    completed_runs = load_completed_runs(out_csv)
    print(f"\n  Output: {out_csv.name}")
    print(f"  Already completed: {len(completed_runs)} runs")
    
    # Get unique stars
    stars_to_process = df_grid[['StarType', 'StarFile']].drop_duplicates()
    if star_type_filter:
        stars_to_process = stars_to_process[stars_to_process['StarType'] == star_type_filter]
    
    # Methods (6 total)
    methods = ['DHR', 'FS-BSM', 'TBATS', 'ARMAX', 'GP-Celerite', 'R-TVF']
    
    total_runs = len(stars_to_process) * len(methods) * N_FOLDS
    print(f"\n  Total planned runs: {total_runs}")
    print(f"  ({len(stars_to_process)} stars × {len(methods)} methods × {N_FOLDS} folds)\n")
    
    for _, star_row in stars_to_process.iterrows():
        star_type = star_row['StarType']
        star_file = star_row['StarFile']
        
        file_path = DATA_DIRS[star_type] / star_file
        if not file_path.exists():
            print(f"  ⚠ File not found: {star_file}")
            continue
        
        t, y_true = load_dat(file_path)
        print(f"\n★ {star_type} - {star_file} ({len(t)} points)")
        
        for method in methods:
            config_key = (star_type, star_file, method)
            if config_key not in best_configs:
                print(f"  ⚠ No config for {method}")
                continue
            
            best_cfg = best_configs[config_key]
            params_dict = json.loads(best_cfg['params'])
            
            print(f"  {method}: grid_R²={best_cfg['best_r2']:.4f}")
            
            for fold in range(N_FOLDS):
                run_key = (star_type, star_file, method, fold)
                
                if run_key in completed_runs:
                    print(f"    Fold {fold+1}: SKIP (exists)")
                    continue
                
                # Set DETERMINISTIC seed for this fold
                fold_seed = FOLD_SEEDS[fold]
                set_global_seed(fold_seed)
                
                # Generate missing block with this seed
                y_missing, start, end = simulate_block_run(y_true, block_n=NBLOCK, seed=fold_seed)
                mask_block = np.zeros_like(y_true, dtype=bool)
                mask_block[start:end] = True
                
                try:
                    t0 = time.perf_counter()
                    y_imp, y_std = run_single_method(method, params_dict, t, y_true, y_missing, start, end, mask_block, star_type)
                    elapsed = time.perf_counter() - t0
                    
                    if y_imp is not None:
                        metrics = metrics_from_block(y_true, y_imp, start, end)
                        
                        result_entry = {
                            "StarType": star_type,
                            "StarFile": star_file,
                            "Method": method,
                            "Fold": fold,
                            "FoldSeed": fold_seed,
                            "BlockStart": start,
                            "BlockEnd": end,
                            "Params": best_cfg['params'],
                            "MAE": metrics["MAE"],
                            "MSE": metrics["MSE"],
                            "MAPE": metrics["MAPE"],
                            "R2": metrics["R2"],
                            "Time": elapsed
                        }
                        
                        # Save incrementally
                        df_temp = pd.DataFrame([result_entry])
                        if out_csv.exists():
                            df_temp.to_csv(out_csv, mode='a', header=False, index=False)
                        else:
                            df_temp.to_csv(out_csv, index=False)
                        
                        # Save plot
                        plots_out_dir = PLOTS_DIR_HADS if star_type == "HADS" else PLOTS_DIR_LADS
                        save_imputation_plot(t, y_true, y_imp, start, end, 
                                           star_file.replace('.dat',''), method, fold, 
                                           metrics, plots_out_dir, y_std=y_std)
                        
                        print(f"    Fold {fold+1} (seed={fold_seed}): MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.4f}")
                    else:
                        print(f"    Fold {fold+1}: FAILED (null result)")
                        
                except Exception as e:
                    print(f"    Fold {fold+1}: ERROR - {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("  CROSS VALIDATION COMPLETE!")
    print(f"  Results: {out_csv}")
    print("="*70)
    
    if out_csv.exists():
        df_final = pd.read_csv(out_csv)
        print("\n  Summary (mean ± std per method):")
        summary = df_final.groupby('Method').agg({
            'MAE': ['mean', 'std'],
            'R2': ['mean', 'std']
        }).round(4)
        print(summary)

# ============= MAIN =============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Fold CV (Q1 JCR Methodology)")
    parser.add_argument("--star-type", choices=["HADS", "LADS"], help="Filter by star type")
    args = parser.parse_args()
    
    run_cross_validation(star_type_filter=args.star_type)
