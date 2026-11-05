import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
# RESULTS_2000
RESULTS_DIR = SCRIPT_DIR / "results_2000" / "2_cross_validation"
DATA_DIR = PARENT_DIR / "estrellas_test"
PLOTS_DIR_HADS = RESULTS_DIR / "plots_hads"
PLOTS_DIR_LADS = RESULTS_DIR / "plots_lads"

sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(SCRIPT_DIR)) # Add src_2000 to path

# Imports from existing codebase
try:
    import lightCurve as lc
    # Using local modelos if available or global
    try:
        import modelos_hads_lads as mh
    except:
        sys.path.append(str(PARENT_DIR / "src"))
        import modelos_hads_lads as mh
        
    # We might need runner_comparativa_3x2 from src or src_2000
    try:
        from runner_comparativa_3x2 import load_dat, simulate_block_run, metrics_from_block
    except:
        sys.path.append(str(PARENT_DIR / "src"))
        from runner_comparativa_3x2 import load_dat, simulate_block_run, metrics_from_block

except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# GP Check
try:
    import celerite2
    from celerite2 import terms
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

# Configuration
SEED_BASE = 42
NBLOCK = 2000 # N=2000 for src_2000
FOLD = 0 
FOLD_SEED = SEED_BASE + FOLD * 1000 

FIXED_DL_PARAMS = {
    "window_size": 512,
    "batch_size": 32,
    "epochs": 200,
    "patience": 30
}

# Star Lists
HADS_RTVF_STARS = [
    'AD_Cmilc', 'AE_Umalc', 'BS_Aqrlc', 'DE_Laclc',
    'V1393_Cenlc', 'GW_Umalc', 'HD_146953lc', 'Rho_Puplc',
    'RS_Grulc', 'RY_Leplc', 'SS_Psclc', 'SX_Phelc',
    'V1162_Orilc', 'V1384_Tau_lc', 'DX_Cetlc', 'V1719_Cyglc'
]

LADS_RTVF_STARS = [
    'TIC_100531058lc', 'TIC_102192161lc', 'TIC_106886169lc', 'TIC_11361473lc',
    'TIC_121729614lc', 'TIC_122615966lc', 'TIC_12524129_lc', 'TIC_137341551lc',
    'TIC_137796620lc', 'TIC_159647185lc', 'TIC_161172103lc', 'TIC_172193026lc',
    'TIC_18658256lc', 'TIC_196921106lc', 'TIC_242944780lc', 'TIC_246902545lc'
]

# Only R-TVF for grids as requested, but keeping structure
ALL_METHODS = ['R-TVF']

def set_global_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except: pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except: pass

def run_gp_imputation(t, y_missing, start, end, params):
    if not GP_AVAILABLE: return None
    mask_train = np.isfinite(y_missing)
    t_train = t[mask_train]
    y_train = y_missing[mask_train]
    sigma_est = np.std(y_train)
    kernel = terms.SHOTerm(sigma=sigma_est, rho=float(params.get("rho", 0.1)), tau=float(params.get("tau", 10.0)))
    gp = celerite2.GaussianProcess(kernel, mean=np.median(y_train))
    try:
        gp.compute(t_train, yerr=float(params.get("noise", 0.001)))
        return gp.predict(y_train, t=t, return_var=False)
    except: return None

from importlib.machinery import SourceFileLoader
cv_module = SourceFileLoader("cv", "2.cross_validation.py").load_module()

def execute_model(method, t, y_missing, start, end, params, y_true, star_type):
    mask_block = np.zeros_like(y_true, dtype=bool)
    mask_block[start:end] = True
    return cv_module.run_single_method(method, params, t, y_true, y_missing, start, end, mask_block, star_type)

def save_plot_large(t, y_true, y_imp, start, end, star_name, method, metrics, out_dir, y_std=None):
    # PLOTTING WITH LARGE FONTS (MATCHING REFINED STYLE)
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(12, 6)) 
    
    pad = 200
    left = max(0, start - pad)
    right = min(len(t), end + pad)
    
    # Larger fonts 
    FONT_AXIS = 18       
    FONT_TICKS = 16      
    FONT_LEGEND = 16     
    FONT_TITLE = 12      # Reverted to legacy size
    
    ax.plot(t[left:right], y_true[left:right], 'k.', alpha=0.3, markersize=4, label='Original', zorder=1)
    ax.plot(t[start:end], y_true[start:end], 'b-', alpha=0.6, linewidth=2.0, label='Ground truth', zorder=3)
    ax.plot(t[start:end], y_imp[start:end], 'r-', linewidth=2.5, label=f'{method}', zorder=4)
    if y_std is not None:
        ax.fill_between(t[start:end], y_imp[start:end] - 3*y_std[start:end], y_imp[start:end] + 3*y_std[start:end], color='#FFB6C1', alpha=1.0, zorder=2, label='3σ Band')
    ax.axvspan(t[start], t[end-1], color='yellow', alpha=0.15, zorder=0)
    
    ax.set_xlabel('Time (HJD)', fontsize=FONT_AXIS)
    ax.set_ylabel('Flux', fontsize=FONT_AXIS)
    
    ax.set_title(f"{star_name.replace('lc','')} - {method}\nR²={metrics['R2']:.3f} | MAE={metrics['MAE']:.3f}", 
                fontsize=FONT_TITLE, pad=12)
    
    ax.tick_params(axis='both', which='major', labelsize=FONT_TICKS)
    ax.legend(loc='upper right', fontsize=FONT_LEGEND, frameon=True, framealpha=0.9, fancybox=True)
    ax.grid(True, alpha=0.3)
    
    filename = f"{star_name.replace(' ', '_').replace('.dat', '')}_{method}_fold{FOLD}_large.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    
    fig.tight_layout()
    fig.savefig(filepath, dpi=300) 
    plt.close(fig)
    return filepath

def process_star(star_base, star_type, methods_to_run):
    print(f"\nProcessing {star_base} ({star_type})...")
    
    actual_filename = star_base.replace('_', ' ') + ".dat"
    data_path = DATA_DIR / star_type.lower() / actual_filename
    if not data_path.exists():
        actual_filename = star_base.replace('_', ' ').replace(' lc', 'lc') + ".dat"
        data_path = DATA_DIR / star_type.lower() / actual_filename
        
    if not data_path.exists():
         print(f"  ❌ File not found: {actual_filename}")
         return

    csv_path = RESULTS_DIR / "cv_results_all.csv"
    if not csv_path.exists():
        print(f"  ❌ Results CSV not found: {csv_path}")
        return
        
    df_res = pd.read_csv(csv_path)
    
    t, y_true = load_dat(data_path)
    
    set_global_seed(FOLD_SEED)
    y_missing, start, end = simulate_block_run(y_true, block_n=NBLOCK, seed=FOLD_SEED)
    
    out_dir = PLOTS_DIR_HADS if star_type == "HADS" else PLOTS_DIR_LADS
    
    for method in methods_to_run:
        row = df_res[(df_res['StarFile'] == actual_filename) & (df_res['Method'] == method) & (df_res['Fold'] == FOLD)]
        if row.empty:
             row = df_res[(df_res['StarFile'].str.replace(' ', '_') == star_base + ".dat") & (df_res['Method'] == method) & (df_res['Fold'] == FOLD)]
        
        if row.empty:
            print(f"  ⚠ No results found for {method}")
            continue
            
        try:
             params = json.loads(row.iloc[0]['Params'])
        except:
             # Fallback for old format if needed
             params = eval(row.iloc[0]['Params'])
        
        print(f"  Running {method}...")
        y_imp, y_std = execute_model(method, t, y_missing, start, end, params, y_true, star_type)
        
        if y_imp is not None:
             metrics = metrics_from_block(y_true, y_imp, start, end)
             save_plot_large(t, y_true, y_imp, start, end, star_base, method, metrics, out_dir, y_std=y_std)
             print(f"    ✓ Saved plot. R²={metrics['R2']:.3f}")

def main():
    print("REGENERATING PLOTS WITH LARGE FONTS (SRC_2000)...")
    
    # 2. R-TVF Grids
    for star in HADS_RTVF_STARS:
        process_star(star, 'HADS', ALL_METHODS)
        
    for star in LADS_RTVF_STARS:
        process_star(star, 'LADS', ALL_METHODS)
        
    print("\nDONE.")

if __name__ == "__main__":
    main()
