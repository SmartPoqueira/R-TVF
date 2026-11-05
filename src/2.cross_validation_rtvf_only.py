#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   2. CROSS VALIDATION - 5-Fold CV (Q1 JCR Methodology)                       ║
║   Delta Scuti Stars Missing Values Imputation                                ║
║   6 Métodos: R-TVF, ARMAX, BiLSTM, SAITS, MIARMA, GP-Celerite           ║
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
        load_dat, simulate_block_run, metrics_from_block, _safe_bilstm_call
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
        GRID_SEARCH_DIR / "grid_search_hads.csv",
        GRID_SEARCH_DIR / "grid_search_lads.csv"
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
def save_imputation_plot(t, y_true, y_imp, start, end, star_name, method, fold, metrics, out_dir):
    """Save imputation plot for visual inspection."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    pad = 200
    left = max(0, start - pad)
    right = min(len(t), end + pad)
    
    ax.plot(t[left:right], y_true[left:right], 'k.', alpha=0.3, markersize=2, label='Original')
    ax.plot(t[start:end], y_true[start:end], 'b-', alpha=0.6, linewidth=1, label='Ground truth')
    ax.plot(t[start:end], y_imp[start:end], 'r-', linewidth=1.5, label=f'{method}')
    ax.axvspan(t[start], t[end-1], color='yellow', alpha=0.15)
    
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

# ============= GP IMPUTATION (embedded from 1.gaussian_processes.py) =============
def run_gp_imputation(t, y_missing, start, end, params):
    """GP Imputation using celerite2 SHOTerm. Requires 'rho' in params."""
    if not GP_AVAILABLE:
        return None
    
    mask_train = np.isfinite(y_missing)
    t_train = t[mask_train]
    y_train = y_missing[mask_train]
    
    sigma_est = np.std(y_train)
    
    kernel = terms.SHOTerm(
        sigma=sigma_est,
        rho=float(params.get("rho", 0.1)),
        tau=float(params.get("tau", 10.0))
    )
    
    mean_val = np.median(y_train)
    gp = celerite2.GaussianProcess(kernel, mean=mean_val)
    
    try:
        gp.compute(t_train, yerr=float(params.get("noise", 0.001)))
        mu = gp.predict(y_train, t=t, return_var=False)
        return mu
    except Exception as e:
        print(f"      [GP Error] {e}")
        return None

# ============= RUN SINGLE METHOD =============
def run_single_method(method, params_dict, t, y_true, y_missing, start, end, mask_block):
    """Execute imputation with given method and parameters."""
    y_imp = None
    
    if method == "BiLSTM":
        units = int(params_dict.get('units', 64))
        layers = int(params_dict.get('bilstm_layers', 2))
        lr = float(params_dict.get('lr', 1e-3))
        device = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else None
        
        y_imp = _safe_bilstm_call(
            mh, t, y_missing, start, end,
            fmin=0.5, fmax=50.0,
            window=FIXED_DL_PARAMS["window_size"],
            hidden=units, layers=layers,
            epochs=FIXED_DL_PARAMS["epochs"],
            batch=FIXED_DL_PARAMS["batch_size"],
            lr=lr, device=device
        )
        
    elif method == "SAITS":
        # Use tiempo/flujo column names (required by lightCurve.py)
        df_gt = pd.DataFrame({'tiempo': t, 'flujo': y_true})
        df_inf = pd.DataFrame({'tiempo': t, 'flujo': y_missing})
        
        saits_params = dict(params_dict)
        saits_params["window_size"] = FIXED_DL_PARAMS["window_size"]
        saits_params["batch_size"] = FIXED_DL_PARAMS["batch_size"]
        saits_params["saits_patience"] = FIXED_DL_PARAMS["patience"]
        if "d_model" not in saits_params: saits_params["d_model"] = 64
        
        device = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else None
        df_saits = lc.rellenar_con_saits(
            df_inf, saits_params, FIXED_DL_PARAMS["epochs"], 
            mask_block, df_train=df_gt, device=device
        )
        y_imp = df_saits['flujo_relleno'].values
        
    elif method == "ARMAX":
        # Mismo código que 1.grid_search.py
        freqs = mh.ls_top_freqs(t[np.isfinite(y_missing)], y_missing[np.isfinite(y_missing)], M=int(params_dict.get("fourier_K", 5)))
        y_imp, _ = mh.armax_impute_block_multi(
            t, y_missing, start, end, freqs_cpd=freqs,
            p=int(params_dict.get("arma_p", 5)), q=int(params_dict.get("arma_q", 2)), trend="c"
        )
        y_imp = np.asarray(y_imp, float)
        
    elif method == "MIARMA":
        y_imp = mh.miarma_impute_block(
            y_missing, start, end,
            p=int(params_dict.get("arma_p", 5)), q=int(params_dict.get("arma_q", 2)), trend="c",
            win_left=max(50, int(2.0*(end-start))),
            win_right=max(50, int(2.0*(end-start)))
        )
        y_imp = np.asarray(y_imp, float)
        
    elif method == "R-TVF":
        l2_val = float(params_dict.get("l2", 1e-2))
        y_imp, _ = mh.tvar_fourier_impute_block_multi(
            t, y_missing, start, end,
            M=int(params_dict.get("fourier_K", 5)), degree=int(params_dict.get("degree", 2)),
            trend_deg=1, l2=l2_val,
            fmin=None, fmax=None, max_samples_ls=60000
        )
        y_imp = np.asarray(y_imp, float)
        
    elif method == "GP-Celerite":
        y_imp = run_gp_imputation(t, y_missing, start, end, params_dict)
    
    return y_imp

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
        out_csv = RESULTS_DIR / f"cv_results_rtvf_optimized_{star_type_filter.lower()}.csv"
    else:
        out_csv = RESULTS_DIR / "cv_results_rtvf_optimized.csv"
    
    completed_runs = load_completed_runs(out_csv)
    print(f"\n  Output: {out_csv.name}")
    print(f"  Already completed: {len(completed_runs)} runs")
    
    # Get unique stars
    stars_to_process = df_grid[['StarType', 'StarFile']].drop_duplicates()
    if star_type_filter:
        stars_to_process = stars_to_process[stars_to_process['StarType'] == star_type_filter]
    
    # Methods (6 total)
    # Methods (OPTIMIZED: ONLY R-TVF)
    methods = ['R-TVF']  # Run ONLY R-TVF to capture new l2 improvements
    
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
                    y_imp = run_single_method(method, params_dict, t, y_true, y_missing, start, end, mask_block)
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
                                           metrics, plots_out_dir)
                        
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
