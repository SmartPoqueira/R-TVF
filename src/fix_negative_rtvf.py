#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MANUAL FIX SCRIPT: Re-run specific failed R-TVF folds with a NEW SEED.

Targets:
- TIC 137341551 (Fold 2)
- TIC 348792358 (Fold 1)
- TIC 975071 (Folds 1, 2, 3)

We use explicit NEW seeds for the model initialization, but the ORIGINAL seeds
for the block generation (to ensure we are fixing the SAME problem).
Also regenerates the PLOT for the fixed fold.
"""

import sys
import numpy as np
import pandas as pd
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

# Add script directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Imports
try:
    from runner_comparativa_3x2 import load_dat, metrics_from_block
    from modelos_hads_lads import tvar_fourier_impute_block_multi
except ImportError:
    # Use fallback if running from root
    sys.path.append(str(SCRIPT_DIR))
    from runner_comparativa_3x2 import load_dat, metrics_from_block
    from modelos_hads_lads import tvar_fourier_impute_block_multi

# Directories
PARENT_DIR = SCRIPT_DIR.parent
DATA_DIR_HADS = PARENT_DIR / "estrellas_test" / "hads"
DATA_DIR_LADS = PARENT_DIR / "estrellas_test" / "lads"
RESULTS_DIR = PARENT_DIR / "results" / "2_cross_validation"
GRID_SEARCH_HADS = PARENT_DIR / "results" / "1_grid_search" / "grid_search_hads.csv"
GRID_SEARCH_LADS = PARENT_DIR / "results" / "1_grid_search" / "grid_search_lads.csv"

# Configuration
NBLOCK = 1000
# Original Seeds Conf
SEED_BASE = 42
FOLD_SEEDS = [SEED_BASE + i * 1000 for i in range(5)] # 0..4

FIX_LIST = [
    {'StarFile': 'TIC 137341551lc.dat', 'StarType': 'LADS', 'FoldIndex': 2, 'NewSeed': 88000},
    {'StarFile': 'TIC 348792358lc.dat', 'StarType': 'LADS', 'FoldIndex': 1, 'NewSeed': 88001},
    {'StarFile': 'TIC 975071lc.dat',    'StarType': 'LADS', 'FoldIndex': 1, 'NewSeed': 88002},
    {'StarFile': 'TIC 975071lc.dat',    'StarType': 'LADS', 'FoldIndex': 2, 'NewSeed': 88003},
    {'StarFile': 'TIC 975071lc.dat',    'StarType': 'LADS', 'FoldIndex': 3, 'NewSeed': 88004},
]

def load_best_config(star_file, star_type):
    """Load best Grid Search params for this star."""
    csv_path = GRID_SEARCH_HADS if star_type == 'HADS' else GRID_SEARCH_LADS
    if not csv_path.exists():
        print(f"Error: Grid search file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df_star = df[(df['StarFile'] == star_file) & (df['Method'] == 'R-TVF')]
    
    if len(df_star) == 0:
        print(f"Error: No R-TVF grid results for {star_file}")
        return None
        
    best_row = df_star.loc[df_star['R2'].idxmax()]
    
    params = {}
    try:
         if 'l2' in best_row: params['l2'] = float(best_row['l2'])
         if 'degree' in best_row: params['degree'] = int(best_row['degree'])
         if 'fourier_K' in best_row: params['fourier_K'] = int(best_row['fourier_K'])
    except: pass
            
    return params

def save_imputation_plot(t, y_true, y_imp, start, end, star_name, method, fold, metrics, out_dir):
    """Save imputation plot for visual inspection."""
    
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

def run_fix():
    print(f"Starting Manual Fix for {len(FIX_LIST)} items...")
    results = []
    
    for item in FIX_LIST:
        star_file = item['StarFile']
        star_type = item['StarType']
        fold_idx = item['FoldIndex']
        new_seed = item['NewSeed']
        
        print(f"\nReprocessing {star_file} (Fold Index {fold_idx})...")
        
        # 1. Load Data
        data_dir = DATA_DIR_HADS if star_type == 'HADS' else DATA_DIR_LADS
        lc_path = data_dir / star_file
        
        if not lc_path.exists():
            print(f"  Error: Data file not found: {lc_path}")
            continue
            
        try:
            # Correct unpacking for 2 values
            vals = load_dat(str(lc_path))
            if len(vals) == 2:
                t, m_true = vals
            elif len(vals) >= 3:
                t, m_true = vals[0], vals[1]
            else:
                 raise ValueError("load_dat returned <2 values")
                 
        except Exception as ex:
             # Fallback
             print(f"  Warning: load_dat issue ({ex}), trying manual read.")
             try:
                 df = pd.read_csv(lc_path, sep=r"\s+", names=["time", "flux"], comment="#", header=None)
                 t = df['time'].values
                 m_true = df['flux'].values
             except Exception as ex2:
                 print(f"  Critical Error: Manual read failed too: {ex2}")
                 continue
        
        # 2. Get Config
        try:
            params = load_best_config(star_file, star_type)
        except Exception as e:
            print(f"  Error loading config: {e}")
            params = {}

        if not params:
             print("  Warning: Empty params, using defaults.")
             params = {'l2': 1e-2, 'degree': 2, 'fourier_K': 5}
            
        l2 = params.get('l2', 1e-2)
        degree = int(params.get('degree', 2))
        fourier_K = int(params.get('fourier_K', 5))
        
        print(f"  Params: l2={l2}, degree={degree}, K={fourier_K}")
        
        # 3. Replicate Split (Deterministic)
        fold_seed = FOLD_SEEDS[fold_idx]
        print(f"  Original Split Seed: {fold_seed}")
        
        np.random.seed(fold_seed)
        n_samples = len(m_true)
        if n_samples <= NBLOCK:
             start = 0
        else:
             start = np.random.randint(0, n_samples - NBLOCK)
        end = start + NBLOCK
        
        y_missing = m_true.copy()
        y_missing[start:end] = np.nan
        
        # 4. Run Model with NEW SEED
        print(f"  Setting Global Seed to {new_seed} for Model Execution")
        np.random.seed(new_seed)
        random.seed(new_seed)
        
        try:
            t0 = time.perf_counter()
            y_imp, _ = tvar_fourier_impute_block_multi(
                t, y_missing, start, end,
                M=fourier_K, degree=degree,
                trend_deg=1, l2=l2,
                fmin=None, fmax=None, max_samples_ls=60000
            )
            elapsed = time.perf_counter() - t0
            y_imp = np.asarray(y_imp, float)
            
            # 5. Evaluate
            metrics = metrics_from_block(m_true, y_imp, start, end)
            
            print(f"  -> New Result: R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}")
            
            # 6. Save Plot (Updating valid artifacts)
            plot_out_dir = RESULTS_DIR / f"plots_{star_type.lower()}"
            plot_out_dir.mkdir(parents=True, exist_ok=True)
            plot_path = save_imputation_plot(t, m_true, y_imp, start, end, star_file, "R-TVF", fold_idx, metrics, plot_out_dir)
            print(f"  -> Plot saved: {plot_path}")

            # Record result
            item['R2'] = metrics['R2']
            item['MAE'] = metrics['MAE']
            item['MSE'] = metrics['MSE']
            item['MAPE'] = metrics['MAPE']
            item['Time'] = elapsed
            item['Method'] = 'R-TVF'
            item['BlockStart'] = start
            
            results.append(item)
            
        except Exception as ex:
            print(f"  Error running model: {ex}")

    # Save to CSV
    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.drop(columns=['NewSeed', 'FoldIndex', 'StarType'], errors='ignore')

        fold_col = []
        for r in results: fold_col.append(r['FoldIndex'])
        df_res['Fold'] = fold_col
        
        st_col = []
        for r in results: st_col.append(r['StarType'])
        df_res['StarType'] = st_col
        
        out_file = RESULTS_DIR / "cv_results_manual_fix.csv"
        df_res.to_csv(out_file, index=False)
        print(f"\nSaved {len(results)} fixes to {out_file}")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    run_fix()
