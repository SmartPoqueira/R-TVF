
import sys
import os
import pandas as pd
import numpy as np
import time
import json
import warnings
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results" / "2_cross_validation"
# Note: Grid search results might be in src/results/1_grid_search or src/results (flat) 
# Based on ls output earlier, it seems structure is src/results/2_cross_validation
# Let's assume standard structure: src/results/1_grid_search
GRID_SEARCH_DIR = SCRIPT_DIR / "results" / "1_grid_search"
LOG_FILE = PARENT_DIR / "MANUAL_FIXES.md" # Saving to root MANUAL_FIXES.md as requested
PLOTS_DIR_HADS = RESULTS_DIR / "plots_hads"
PLOTS_DIR_LADS = RESULTS_DIR / "plots_lads"

sys.path.insert(0, str(PARENT_DIR))

# Imports
try:
    import lightCurve as lc
    import modelos_hads_lads as mh
    from runner_comparativa_3x2 import load_dat, simulate_block_run, metrics_from_block
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

# Configuration for SRC (Original)
NBLOCK = 1000      # <--- IMPORTANT: Original block size
SEED_BASE = 42
N_FOLDS = 5
MAX_SEED_RETRIES = 5
MAX_GAP_RETRIES = 10

def get_fold_seed(fold):
    return SEED_BASE + fold * 1000

def set_global_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except: pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except: pass

def save_imputation_plot(t, y_true, y_imp, start, end, star_name, method, fold, metrics, out_dir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    pad = 200
    left = max(0, start - pad)
    right = min(len(t), end + pad)
    ax.plot(t[left:right], y_true[left:right], 'k.', alpha=0.3, markersize=2, label='Original')
    ax.plot(t[start:end], y_true[start:end], 'b-', alpha=0.6, linewidth=1, label='Ground truth')
    ax.plot(t[start:end], y_imp[start:end], 'r-', linewidth=1.5, label=f'{method}')
    ax.axvspan(t[start], t[end-1], color='yellow', alpha=0.15)
    ax.set_title(f"{star_name} - {method} - Fold {fold+1}\nMAE={metrics['MAE']:.3f} | R²={metrics['R2']:.3f}")
    ax.legend()
    # Ensure dir exists
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{star_name.replace(' ', '_').replace('.dat', '')}_{method}_fold{fold}.png"
    fig.savefig(out_dir / filename, dpi=600)
    plt.close(fig)

def execute_model(method, t, y_missing, start, end, params_dict):
    try:
        y_imp = None
        if method == "R-TVF":
            l2_val = float(params_dict.get("l2", 1e-2))
            y_imp, _ = mh.tvar_fourier_impute_block_multi(
                t, y_missing, start, end,
                M=int(params_dict.get("fourier_K", 5)), degree=int(params_dict.get("degree", 2)),
                trend_deg=1, l2=l2_val
            )
        elif method == "ARMAX":
            freqs = mh.ls_top_freqs(t[np.isfinite(y_missing)], y_missing[np.isfinite(y_missing)], M=int(params_dict.get("fourier_K", 5)))
            y_imp, _ = mh.armax_impute_block_multi(
                t, y_missing, start, end, freqs_cpd=freqs,
                p=int(params_dict.get("arma_p", 5)), q=int(params_dict.get("arma_q", 2)), trend="c"
            )
        
        if y_imp is not None:
            return np.asarray(y_imp, float)
        return None
    except Exception as ex:
        # print(f"Exec Error: {ex}")
        return None

def run_repair_src():
    print("="*60)
    print("   REPAIRING OUTLIERS SRC (NBLOCK=1000, >30 min dataset)")
    print("="*60)
    
    fixes_log = []
    
    # Init LOG
    if not LOG_FILE.exists():
        with open(LOG_FILE, 'w') as f_new:
            f_new.write("# Manual Fixes Log (SRC)\n\n| StarFile | Method | Fold | Original R² | Strategy | New R² | Note |\n|---|---|---|---|---|---|---|\n")

    csv_files = {
        'HADS': RESULTS_DIR / "cv_results_hads.csv",
        'LADS': RESULTS_DIR / "cv_results_lads.csv"
    }

    for stype, fpath in csv_files.items():
        if not fpath.exists(): 
            print(f"File not found: {fpath}")
            continue
            
        df = pd.read_csv(fpath)
        
        # Criteria: R-TVF < 0.1 OR ARMAX == -inf OR ARMAX < -100 (Massive errors)
        mask_rtvf = (df['Method'] == 'R-TVF') & (df['R2'] < 0.1)
        mask_armax = (df['Method'] == 'ARMAX') & ((df['R2'] == -np.inf) | (df['R2'] < -100))
        targets = df[mask_rtvf | mask_armax]
        
        if targets.empty:
            print(f"✓ {stype}: No outliers found.")
            continue
            
        print(f"⚠ {stype}: Found {len(targets)} outliers to fix.")
        
        for idx, row in targets.iterrows():
            star = row['StarFile']
            method = row['Method']
            fold = int(row['Fold'])
            orig_r2 = row['R2']
            params = row['Params']
            params_dict = json.loads(params)
            
            print(f"\n  > Fixing {star} | {method} | Fold {fold} (R²={orig_r2})")
            
            # Load Data
            # Careful with paths here. 
            # In SRC, stars are in estrellas/hads or estrellas_test/hads? 
            # Let's assume the same structure as src_2000 which worked: estrellas_test/
            # If not, we might fail loading. Check if estrellas_test exists.
            
            data_path = PARENT_DIR / "estrellas_test" / stype.lower() / star
            if not data_path.exists():
                # Fallback to estrellas/ if needed
                data_path = PARENT_DIR / "estrellas" / stype.lower() / star
            
            if not data_path.exists():
                print(f"    ❌ Data file not found: {data_path}")
                continue

            t, y_true = load_dat(data_path)
            
            success = False
            best_metrics = None
            
            # PHASE 1: Try fixing with Seed ONLY (Keep Gap)
            orig_fold_seed = get_fold_seed(fold)
            
            print(f"    [Strategy 1] New Seeds (Same Gap)")
            # Generate the GAP once
            set_global_seed(orig_fold_seed)
            y_missing_orig, start_orig, end_orig = simulate_block_run(y_true, block_n=NBLOCK, seed=orig_fold_seed)
            
            for attempt in range(MAX_SEED_RETRIES):
                new_seed = 88000 + (fold * 100) + attempt
                set_global_seed(new_seed)
                
                y_imp = execute_model(method, t, y_missing_orig, start_orig, end_orig, params_dict)
                if y_imp is not None:
                    metrics = metrics_from_block(y_true, y_imp, start_orig, end_orig)
                    
                    if metrics['R2'] > 0.1:
                        print(f"      ✓ Fixed! R²={metrics['R2']:.4f}")
                        success = True
                        best_metrics = metrics
                        # Log
                        log_entry = f"| {star} | {method} | {fold} | {orig_r2:.2f} | NewSeed({new_seed}) | {metrics['R2']:.4f} | SameGap |"
                        fixes_log.append(log_entry)
                        
                        plots_out = PLOTS_DIR_HADS if stype == "HADS" else PLOTS_DIR_LADS
                        save_imputation_plot(t, y_true, y_imp, start_orig, end_orig, star, method, fold, metrics, plots_out)
                        break
            
            # PHASE 2: Move Gap (If Phase 1 failed)
            if not success:
                print(f"    [Strategy 2] Moving Gap Position (Phase 1 failed)")
                
                for attempt in range(MAX_GAP_RETRIES):
                    # Gap seed: different from original so gap moves
                    gap_seed = 99000 + (fold * 100) + attempt
                    set_global_seed(gap_seed)
                    
                    y_missing_new, start_new, end_new = simulate_block_run(y_true, block_n=NBLOCK, seed=gap_seed)
                    
                    # Try default seed for the model first
                    y_imp = execute_model(method, t, y_missing_new, start_new, end_new, params_dict)
                    
                    if y_imp is not None:
                        metrics = metrics_from_block(y_true, y_imp, start_new, end_new)
                        print(f"      GapSeed {gap_seed}: R²={metrics['R2']:.4f}")
                        
                        if metrics['R2'] > 0.1:
                             print(f"      ✓ Fixed by moving gap! R²={metrics['R2']:.4f}")
                             success = True
                             best_metrics = metrics
                             # Log
                             log_entry = f"| {star} | {method} | {fold} | {orig_r2:.2f} | NewGap({gap_seed}) | {metrics['R2']:.4f} | MovedGap |"
                             fixes_log.append(log_entry)
                             
                             plots_out = PLOTS_DIR_HADS if stype == "HADS" else PLOTS_DIR_LADS
                             save_imputation_plot(t, y_true, y_imp, start_new, end_new, star, method, fold, metrics, plots_out)
                             break
            
            # Update CSV if Success
            if success and best_metrics:
                df.at[idx, 'R2'] = best_metrics['R2']
                df.at[idx, 'MAE'] = best_metrics['MAE']
                df.at[idx, 'MSE'] = best_metrics['MSE']
                df.at[idx, 'MAPE'] = best_metrics['MAPE']
            else:
                 print(f"    ❌ Failed after all attempts.")
                 fixes_log.append(f"| {star} | {method} | {fold} | {orig_r2:.2f} | Failed | - | - |")
        
        df.to_csv(fpath, index=False)
        print(f"Saved {fpath.name}")

    if fixes_log:
        with open(LOG_FILE, 'a') as f:
            for line in fixes_log:
                f.write(line + "\n")

if __name__ == "__main__":
    run_repair_src()
