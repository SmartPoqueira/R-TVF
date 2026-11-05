#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   1. GRID SEARCH - Q1 JCR BENCHMARK (ROBUST BASELINES & DL)                  ║
║   Delta Scuti Stars Missing Values Imputation                                ║
║   Models: Classical (MIARMA, Fourier) vs Deep Learning (BiLSTM, SAITS)       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python 1.grid_search.py [--dry-run]
"""

import sys
import os
import time
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para ejecución en servidor/nohup
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# ============= PATHS =============
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
DATA_DIR = PARENT_DIR / "estrellas_test"
RESULTS_DIR = SCRIPT_DIR / "results_2000" / "1_grid_search"
PLOTS_DIR_HADS = RESULTS_DIR / "plots_hads"
PLOTS_DIR_LADS = RESULTS_DIR / "plots_lads"
CONFIGS_DIR = SCRIPT_DIR / "configs"
LOGS_DIR = SCRIPT_DIR / "logs"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR_HADS.mkdir(parents=True, exist_ok=True)
PLOTS_DIR_LADS.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PARENT_DIR))

# ============= IMPORTS =============
try:
    import lightCurve as lc
    import modelos_hads_lads as mh
    from runner_comparativa_3x2 import (
        load_dat, simulate_random_points, simulate_block_run,
        metrics_from_mask, metrics_from_block, _safe_bilstm_call,
        run_gp_imputation
    )
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# GP Imports
try:
    import celerite2
    from celerite2 import terms
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    print("Warning: celerite2 not available. GP-Celerite will be skipped.")

warnings.filterwarnings("ignore")

# ============= GPU SETUP =============
# GPU device será configurado por argumento --gpu
GPU_DEVICE = None  # Se configura en main

# ============= DETERMINISTIC SEED =============
SEED = 42

def set_deterministic_seed(seed=SEED):
    """Configura todos los generadores para ejecución reproducible."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available() and GPU_DEVICE is not None:
            gpu_idx = int(GPU_DEVICE.split(':')[-1]) if ':' in GPU_DEVICE else int(GPU_DEVICE)
            torch.cuda.set_device(gpu_idx)
            print(f"✓ PyTorch (deterministic) GPU: {torch.cuda.get_device_name(gpu_idx)} (cuda:{gpu_idx})")
        elif torch.cuda.is_available():
            print(f"✓ PyTorch (deterministic) GPU: {torch.cuda.get_device_name(0)}")
    except: pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"✓ TensorFlow (deterministic) GPU: {gpus[0]}")
    except: pass
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Semilla global: {seed}")

set_deterministic_seed(SEED)

# ============= CONFIGURATION =============

STARS = {
    "HADS": [
        "AD Cmilc.dat", "AE Umalc.dat", "AN Lynlc.dat", "BE Lynlc.dat", "BL Camlc.dat",
        "BS Aqrlc.dat", "CY Aqrlc.dat", "DE Laclc.dat", "DX Cetlc.dat", "EH Liblc.dat",
        "GP Andlc.dat", "GW Umalc.dat", "HD 146953lc.dat", "KZ Hyalc.dat", "PT Comlc.dat",
        "Rho Puplc.dat", "RS Grulc.dat", "RY Leplc.dat", "SS Psclc.dat", "SX Phelc.dat",
        "V 524 Andlc.dat", "V1162 Orilc.dat", "V1384 Tau lc.dat", "V1393 Cenlc.dat",
        "V1719 Cyglc.dat", "V2455 Cyglc.dat", "V2855 Orilc.dat", "V367 Camlc.dat",
        "VX Hyalc.dat", "VZ Cnclc.dat", "XX Cyglc.dat", "YZ Boolc.dat", "ZZ Miclc.dat"
    ],
    "LADS": [
        "TIC 100531058lc.dat", "TIC 102192161lc.dat", "TIC 106886169lc.dat", "TIC 11361473lc.dat",
        "TIC 121729614lc.dat", "TIC 122615966lc.dat", "TIC 12524129 lc.dat", "TIC 137341551lc.dat",
        "TIC 137796620lc.dat", "TIC 159647185lc.dat", "TIC 161172103lc.dat", "TIC 172193026lc.dat",
        "TIC 18658256lc.dat", "TIC 196921106lc.dat", "TIC 242944780lc.dat", "TIC 246902545lc.dat",
        "TIC 253296458lc.dat", "TIC 255548143lc.dat", "TIC 26957587lc.dat", "TIC 269697721lc.dat",
        "TIC 274038922lc.dat", "TIC 30624832lc.dat", "TIC 329153513lc.dat", "TIC 34197596lc.dat",
        "TIC 34737955 lc.dat", "TIC 348772511lc.dat", "TIC 348792358lc.dat", "TIC 403114672lc.dat",
        "TIC 408906554lc.dat", "TIC 448892817lc.dat", "TIC 452982723lc.dat", "TIC 458689740lc.dat",
        "TIC 459908110lc.dat", "TIC 7245720lc.dat", "TIC 7808834lc.dat", "TIC 79659787lc.dat",
        "TIC 81003lc.dat", "TIC 9591460lc.dat", "TIC 975071lc.dat"
    ]
}

# Subdirectorios para cada tipo de estrella
DATA_DIRS = {
    "HADS": DATA_DIR / "hads",
    "LADS": DATA_DIR / "lads"
}

SCENARIOS = {
    "block_runs": {"nblock": 2000}
}

# PARÁMETROS FIJOS PARA DEEP LEARNING
FIXED_DL_PARAMS = {
    "window_size": 512,
    "batch_size": 32,
    "epochs": 200,
    "patience": 30
}

# =============================================================================
#  GRID GENERATION HELPERS (FACTORIAL DESIGNS + LR SEARCH)
# =============================================================================

def generate_gp_celerite_grid():
    """Factorial Design for GP-Celerite."""
    return [{"gp_tau": tau, "gp_noise": noise} for tau, noise in itertools.product([10.0, 50.0, 100.0], [1e-3, 1e-4])]

def generate_armax_grid():
    """Factorial Design for ARMAX."""
    return [{"arma_p": p, "arma_q": q, "fourier_K": k} for p, q, k in itertools.product([1, 3], [1, 2], [10, 30])]

def generate_fourier_grid():
    """Factorial Design for proposal R-TVF."""
    return [{"fourier_K": k, "degree": d, "l2": l2} for k, d, l2 in itertools.product([10, 30, 60], [1, 3], [0.1, 1e-2, 1e-3, 0.0])]

def generate_dhr_grid():
    """Factorial Design for DHR SS baseline.
    K: number of dominant frequencies (harmonic density)
    p_ar: AR order for residual component (structural flexibility)
    """
    return [{"fourier_K": k, "p_ar": p} for k, p in itertools.product([3, 5], [1, 2])]

def generate_trig_ss_grid():
    """Factorial Design for FS-BSM baseline (Hindrayanto et al., 2010).
    K: number of trigonometric seasonal components
    maxiter: Kalman filter EM convergence budget
    """
    return [{"fourier_K": k, "maxiter": m} for k, m in itertools.product([3, 5], [20, 50])]

def generate_tbats_ss_grid():
    """Factorial Design for TBATS baseline (De Livera et al., 2011).
    K: number of Fourier seasonal periods
    p_ar: AR order for single-source-of-error residual (0 = canonical TBATS, 1 = TBATS+AR)
    """
    return [{"fourier_K": k, "p_ar": p} for k, p in itertools.product([3, 5], [0, 1])]

GRID = {
    "block_runs": {
        "ARMAX": generate_armax_grid(),
        "GP-Celerite": generate_gp_celerite_grid(),
        "DHR": generate_dhr_grid(),
        "FS-BSM": generate_trig_ss_grid(),
        "TBATS": generate_tbats_ss_grid(),
        "R-TVF": generate_fourier_grid(),
    }
}

# Dry run grid
DRY_RUN_GRID = {
    "block_runs": {
        "ARMAX": [{"arma_p": 1, "arma_q": 1, "fourier_K": 3}],
        "GP-Celerite": [{"gp_tau": 50.0, "gp_noise": 1e-3}],
        "DHR": [{"fourier_K": 3, "p_ar": 1}],
        "FS-BSM": [{"fourier_K": 3}],
        "TBATS": [{"fourier_K": 3, "p_ar": 1}],
        "R-TVF": [{"fourier_K": 3, "degree": 1, "l2": 1e-2}],
    }
}

def load_completed_runs(csv_path):
    completed = set()
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                key = (row['StarFile'], row['Scenario'], row['Method'], row['Params'])
                completed.add(key)
        except: pass
    return completed

def save_imputation_plot(t, y_true, y_imp, start, end, star_name, method, params_str, metrics, out_dir):
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
    title_params = params_str[:60] + "..." if len(params_str) > 60 else params_str
    ax.set_title(f"{star_name} - {method}\n{title_params}\nMAE={metrics['MAE']:.3f} | R²={metrics['R2']:.3f}")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
def run_single_method(method, params, t, y_true, y_missing, start, end, mask_block, star_type):
    y_imp = None
    
    # Determine frequency band based on star_type
    if star_type == "HADS":
        fmin, fmax = 3.0, 30.0
    else:
        fmin, fmax = 5.0, 80.0
    
    if method == "GP-Celerite":
        gp_params = {
            "tau": float(params.get("gp_tau", 50.0)),
            "noise": float(params.get("gp_noise", 1e-3))
        }
        y_imp = run_gp_imputation(t, y_missing, start, end, gp_params)
        if y_imp is not None:
            y_imp = np.asarray(y_imp, float)
        
    elif method == "ARMAX":
        freqs = mh.ls_top_freqs(t[np.isfinite(y_missing)], y_missing[np.isfinite(y_missing)], M=params["fourier_K"], fmin=fmin, fmax=fmax)
        y_imp, _ = mh.armax_impute_block_multi(
            t, y_missing, start, end, freqs_cpd=freqs,
            p=params["arma_p"], q=params["arma_q"], trend="c"
        )
        y_imp = np.asarray(y_imp, float)
        
    elif method == "R-TVF":
        l2_val = params.get("l2", 1e-2)
        y_imp, _ = mh.tvar_fourier_impute_block_multi(
            t, y_missing, start, end,
            M=params["fourier_K"], degree=params["degree"],
            trend_deg=1, l2=l2_val,
            fmin=fmin, fmax=fmax, max_samples_ls=60000
        )
        y_imp = np.asarray(y_imp, float)
        
    elif method == "DHR":
        y_imp = mh.dhr_impute_block(t, y_missing, start, end, fmin=fmin, fmax=fmax, M=params["fourier_K"], p_ar=params["p_ar"], maxiter=20)
        y_imp = np.asarray(y_imp, float)
        
    elif method == "FS-BSM":
        y_imp = mh.trig_ss_impute_block(t, y_missing, start, end, fmin=fmin, fmax=fmax, M=params["fourier_K"], maxiter=20)
        y_imp = np.asarray(y_imp, float)
        
    elif method == "TBATS":
        y_imp = mh.tbats_ss_impute_block(t, y_missing, start, end, fmin=fmin, fmax=fmax, M=params["fourier_K"], p_ar=params["p_ar"], maxiter=500)
        y_imp = np.asarray(y_imp, float)

    return y_imp

def run_grid_search(dry_run=False, star_type_filter=None):
    """
    star_type_filter: 'HADS', 'LADS', or None (all)
    """
    results = []
    grid_to_use = DRY_RUN_GRID if dry_run else GRID
    
    # Output file based on filter
    if star_type_filter:
        out_csv = RESULTS_DIR / f"grid_search_{star_type_filter.lower()}.csv"
    else:
        out_csv = RESULTS_DIR / "grid_search_results.csv"
    completed_runs = load_completed_runs(out_csv)
    
    total_runs = 0
    for scen in grid_to_use.values():
        for lst in scen.values(): total_runs += len(lst)
    total_runs *= sum(len(v) for v in STARS.values())
    
    current_run = 0
    print(f"\n{'='*70}")
    print(f"  Q1 JCR GRID SEARCH - {total_runs} combinaciones totales")
    print(f"  Métodos: {list(grid_to_use['block_runs'].keys())}")
    print(f"{'='*70}\n")
    
    for star_type, star_files in STARS.items():
        # Filter by star type if specified
        if star_type_filter and star_type != star_type_filter:
            continue
            
        for star_file in star_files:
            file_path = DATA_DIRS[star_type] / star_file
            if not file_path.exists(): 
                print(f"  ⚠ File not found: {star_file}")
                continue
            
            print(f"\n★ Processing {star_type} - {star_file}")
            t, y_true = load_dat(file_path)

            if "block_runs" in grid_to_use:
                scenario = "block_runs"
                nblock = SCENARIOS[scenario]["nblock"]
                
                # Semilla fija para consistencia
                y_missing, start, end = simulate_block_run(y_true, block_n=nblock, seed=SEED)
                mask_block = np.zeros_like(y_true, dtype=bool)
                mask_block[start:end] = True
                
                for method, params_list in grid_to_use[scenario].items():
                    for params in params_list:
                        current_run += 1
                        params_key = json.dumps(params, sort_keys=True)
                        run_key = (star_file, scenario, method, params_key)
                        
                        if run_key in completed_runs:
                            print(f"  [{current_run}/{total_runs}] SKIP {method} (Ya existe)")
                            continue
                        
                        print(f"  [{current_run}/{total_runs}] Running {method}")
                        print(f"      » Params: {params}")
                        
                        try:
                            t0 = time.perf_counter()
                            y_imp = run_single_method(method, params, t, y_true, y_missing, start, end, mask_block, star_type)
                            
                            if y_imp is not None:
                                metrics = metrics_from_block(y_true, y_imp, start, end)
                                elapsed = time.perf_counter() - t0
                                
                                result_entry = {
                                    "StarType": star_type,
                                    "StarFile": star_file,
                                    "Scenario": scenario,
                                    "Method": method,
                                    "Params": params_key,
                                    "MAE": metrics["MAE"],
                                    "MSE": metrics["MSE"],
                                    "MAPE": metrics["MAPE"],
                                    "R2": metrics["R2"],
                                    "Time": elapsed
                                }
                                results.append(result_entry)
                                
                                df_temp = pd.DataFrame(results)
                                if out_csv.exists():
                                    df_temp.to_csv(out_csv, mode='a', header=False, index=False)
                                else:
                                    df_temp.to_csv(out_csv, index=False)
                                results = []
                                
                                star_name = star_file.replace('.dat', '')
                                plots_out_dir = PLOTS_DIR_HADS if star_type == "HADS" else PLOTS_DIR_LADS
                                save_imputation_plot(t, y_true, y_imp, start, end, 
                                                     star_name, method, params_key, 
                                                     metrics, plots_out_dir)
                                
                                print(f"      ✓ DONE: MAE={metrics['MAE']:.4f} | R²={metrics['R2']:.4f} | {elapsed:.1f}s")
                            else:
                                print(f"      - Skipped/Null result")
                                
                        except Exception as e:
                            print(f"      ✗ ERROR: {e}")
                            import traceback
                            traceback.print_exc()

    print("\n✓ Grid Search completado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--star-type", choices=["HADS", "LADS"], default=None,
                        help="Filter by star type: HADS or LADS")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index to use (0 for RTX 3090, 1 for RTX 2080)")
    args = parser.parse_args()
    
    # Configurar GPU global
    if args.gpu is not None:
        GPU_DEVICE = args.gpu
        print(f"\n{'='*70}")
        print(f"  Using GPU {GPU_DEVICE}")
        print(f"{'='*70}")
    
    # Reinicializar semilla con la GPU correcta
    set_deterministic_seed(SEED)
    
    run_grid_search(dry_run=args.dry_run, star_type_filter=args.star_type)