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
RESULTS_DIR = PARENT_DIR / "results" / "2_cross_validation"
DATA_DIR = PARENT_DIR / "estrellas_test"
PLOTS_DIR_HADS = RESULTS_DIR / "plots_hads"
PLOTS_DIR_LADS = RESULTS_DIR / "plots_lads"

sys.path.insert(0, str(PARENT_DIR))

# Imports from existing codebase
try:
    import lightCurve as lc
    import modelos_hads_lads as mh
    from runner_comparativa_3x2 import load_dat, simulate_block_run, metrics_from_block
    # We import run_single_method from 2.cross_validation to reuse logic
    # Note: We need to make sure we can import it. 
    # Since 2.cross_validation.py has dashes/dots, we might need importlib or just copy the logic.
    # Copying logic is safer to avoid execution on import.
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
NBLOCK = 1000
FOLD = 0 # Grid plots use Fold 0
FOLD_SEED = SEED_BASE + FOLD * 1000 # 42

FIXED_DL_PARAMS = {
    "window_size": 512,
    "batch_size": 32,
    "epochs": 200,
    "patience": 30
}

# Star Lists
# From 8.generate_rtvf_grid.py
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

# Comparison Stars (All Methods)
# From 5.generate_grid_plots.py
HADS_COMPARISON = 'HD_146953lc'
LADS_COMPARISON = 'TIC_452982723lc'

ALL_METHODS = ['MIARMA', 'ARMAX', 'GP-Celerite', 'BiLSTM', 'SAITS', 'R-TVF']

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

def execute_model(method, t, y_missing, start, end, params, y_true):
    # Re-implementing simplified run_single_method logic
    y_imp = None
    mask_block = np.zeros_like(y_true, dtype=bool)
    mask_block[start:end] = True
    
    if method == "ARMAX":
        freqs = mh.ls_top_freqs(t[np.isfinite(y_missing)], y_missing[np.isfinite(y_missing)], M=int(params.get("fourier_K", 10)))
        y_imp, _ = mh.armax_impute_block_multi(t, y_missing, start, end, freqs_cpd=freqs, p=int(params.get("arma_p", 5)), q=int(params.get("arma_q", 2)), trend="c")
    elif method == "MIARMA":
        y_imp = mh.miarma_impute_block(y_missing, start, end, p=int(params.get("arma_p", 5)), q=int(params.get("arma_q", 2)), trend="c", win_left=max(50, int(2.0*(end-start))), win_right=max(50, int(2.0*(end-start))))
    elif method == "R-TVF":
        y_imp, _ = mh.tvar_fourier_impute_block_multi(t, y_missing, start, end, M=int(params.get("fourier_K", 5)), degree=int(params.get("degree", 2)), trend_deg=1, l2=float(params.get("l2", 1e-2)))
    elif method == "GP-Celerite":
        if GP_AVAILABLE:
            if 'rho' not in params:
                freqs = mh.ls_top_freqs(t[np.isfinite(y_missing)], y_missing[np.isfinite(y_missing)], M=1)
                params['rho'] = float(1.0 / freqs[0])
            y_imp = run_gp_imputation(t, y_missing, start, end, params)
    elif method == "BiLSTM":
        from runner_comparativa_3x2 import _safe_bilstm_call
        fmax = 0.5 / np.median(np.diff(t))
        fmin = 1.0/(t[-1]-t[0])
        y_imp = _safe_bilstm_call(mh, t, y_missing, start, end, fmin=fmin, fmax=fmax, K_grid=(1,2,3), window=FIXED_DL_PARAMS["window_size"], hidden=int(params.get("units", 64)), layers=int(params.get("bilstm_layers", 2)), epochs=FIXED_DL_PARAMS["epochs"], batch=FIXED_DL_PARAMS["batch_size"], lr=float(params.get("lr", 1e-3)), stride=4, max_windows=80000, seed=SEED_BASE, device='cuda:1')
    elif method == "SAITS":
        df_inf = pd.DataFrame({'tiempo': t, 'flujo': y_missing})
        df_gt = pd.DataFrame({'tiempo': t, 'flujo': y_true})
        saits_params = dict(params)
        saits_params.update({"window_size": 512, "batch_size": 32, "saits_patience": 30})
        if "d_model" not in saits_params: saits_params["d_model"] = 64
        df_saits = lc.rellenar_con_saits(df_inf, saits_params, 200, mask_block, df_train=df_gt, device='cuda:1')
        y_imp = df_saits['flujo_relleno'].values

    if y_imp is not None:
        return np.asarray(y_imp, float)
    return None

def save_plot_large(t, y_true, y_imp, start, end, star_name, method, metrics, out_dir):
    # PLOTTING WITH ORIGINAL SMALL FONTS (RESTORATION)
    plt.style.use('default') 
    fig, ax = plt.subplots(figsize=(12, 5)) 
    
    pad = 200
    left = max(0, start - pad)
    right = min(len(t), end + pad)
    
    # Original Small Fonts
    FONT_AXIS = 12       
    FONT_TICKS = 10      
    FONT_LEGEND = 10     
    FONT_TITLE = 12      
    
    ax.plot(t[left:right], y_true[left:right], 'k.', alpha=0.3, markersize=2, label='Original')
    ax.plot(t[start:end], y_true[start:end], 'b-', alpha=0.6, linewidth=1, label='Ground truth')
    ax.plot(t[start:end], y_imp[start:end], 'r-', linewidth=1.5, label=f'{method}')
    ax.axvspan(t[start], t[end-1], color='yellow', alpha=0.15)
    
    ax.set_xlabel('Time (HJD)', fontsize=FONT_AXIS)
    ax.set_ylabel('Flux', fontsize=FONT_AXIS)
    
    # Original Title Format
    # star_name - Method \n Params... MAE... R2...
    # We'll keep it clean but small
    ax.set_title(f"{star_name} - {method}\nR²={metrics['R2']:.3f} | MAE={metrics['MAE']:.3f}", 
                fontsize=FONT_TITLE)
    
    ax.tick_params(axis='both', which='major', labelsize=FONT_TICKS)
    
    # Original Legend
    ax.legend(loc='upper right', fontsize=FONT_LEGEND)
    
    ax.grid(True, alpha=0.3)
    
    filename = f"{star_name.replace(' ', '_').replace('.dat', '')}_{method}_fold{FOLD}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    
    fig.tight_layout()
    # Save with high DPI (Original was 150-300, keeping 300 for quality but small fonts)
    fig.savefig(filepath, dpi=300) 
    plt.close(fig)
    return filepath

def process_star(star_base, star_type, methods_to_run):
    print(f"\nProcessing {star_base} ({star_type})...")
    
    # Identify file and results csv
    star_file = f"{star_base}.dat" 
    # Handle naming inconsistencies if any. The lists usually have 'lc' but files have '.dat'
    # Actually checking the lists: 'HD_146953lc' -> file 'HD 146953lc.dat'
    # 'AD_Cmilc' -> 'AD Cmilc.dat'
    # 'TIC_100531058lc' -> 'TIC 100531058lc.dat'
    
    actual_filename = star_base.replace('_', ' ') + ".dat"
    if star_base.startswith("TIC") and " lc" in actual_filename:
         # TIC names sometimes are "TIC 123 lc.dat" or "TIC 123lc.dat"
         # Lets try both
         pass
         
    data_path = DATA_DIR / star_type.lower() / actual_filename
    if not data_path.exists():
        # Try alternative TIC naming
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
    
    # Set Seed and Gap
    set_global_seed(FOLD_SEED)
    y_missing, start, end = simulate_block_run(y_true, block_n=NBLOCK, seed=FOLD_SEED)
    
    out_dir = PLOTS_DIR_HADS if star_type == "HADS" else PLOTS_DIR_LADS
    
    for method in methods_to_run:
        # Find best params
        # We look for the entry with the best R2 in the results, or just take the first one for this method/star
        # The best one is implicitly what was run.
        # However, 2.cross_validation saves ALL folds. We can just look up Fold 0 for this star/method.
        
        row = df_res[(df_res['StarFile'] == actual_filename) & (df_res['Method'] == method) & (df_res['Fold'] == FOLD)]
        if row.empty:
             # Try fuzzy match on filename
             row = df_res[(df_res['StarFile'].str.replace(' ', '_') == star_base + ".dat") & (df_res['Method'] == method) & (df_res['Fold'] == FOLD)]
        
        if row.empty:
            print(f"  ⚠ No results found for {method}")
            continue
            
        params = json.loads(row.iloc[0]['Params'])
        
        print(f"  Running {method}...")
        y_imp = execute_model(method, t, y_missing, start, end, params, y_true)
        
        if y_imp is not None:
             metrics = metrics_from_block(y_true, y_imp, start, end)
             save_plot_large(t, y_true, y_imp, start, end, star_base, method, metrics, out_dir)
             print(f"    ✓ Saved plot. R²={metrics['R2']:.3f} (Old: {row.iloc[0]['R2']:.3f})")

def main():
    print("REGENERATING PLOTS WITH LARGE FONTS...")
    
    # 1. Comparison Stars (All Methods) -- SKIPPED as per user request to preserve existing plots
    # process_star(HADS_COMPARISON.replace('lc', 'lc'), 'HADS', ALL_METHODS)
    # process_star(LADS_COMPARISON.replace('lc', 'lc'), 'LADS', ALL_METHODS)
    
    # 2. R-TVF Grids (R-TVF Only)
    for star in HADS_RTVF_STARS:
        process_star(star, 'HADS', ['R-TVF'])
        
    for star in LADS_RTVF_STARS:
        process_star(star, 'LADS', ['R-TVF'])
        
    print("\nDONE.")

if __name__ == "__main__":
    main()
