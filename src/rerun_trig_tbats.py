"""
rerun_trig_tbats.py
Re-runs ONLY FS-BSM and TBATS grid search + CV for N=1000 on all 72 stars.
- Uses IDENTICAL seeds/folds/block placement as the original run.
- Strips existing FS-BSM / TBATS rows from both CSVs and replaces them.
- All other method rows are untouched.
"""
import sys, os, json, argparse, warnings, itertools, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
BASE_DIR    = SCRIPT_DIR.parent
DATA_DIR    = BASE_DIR / "estrellas_test"
RESULTS_DIR = BASE_DIR / "results" / "2_cross_validation"
GRID_CSV    = BASE_DIR / "results" / "1_grid_search" / "grid_search_results.csv"
CV_CSV      = RESULTS_DIR / "cv_results_all.csv"
PLOTS_HADS  = RESULTS_DIR / "plots_hads"
PLOTS_LADS  = RESULTS_DIR / "plots_lads"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(BASE_DIR))

import modelos_hads_lads as mh
from runner_comparativa_3x2 import load_dat, simulate_block_run, metrics_from_block

# ── Constants (MUST match original pipeline) ───────────────────────────────────
SEED_BASE  = 42
NBLOCK     = 1000
N_FOLDS    = 5
FOLD_SEEDS = [SEED_BASE + i * 1000 for i in range(N_FOLDS)]  # [42,1042,2042,3042,4042]

TARGET_METHODS = ["FS-BSM", "TBATS"]   # These are the names in the CSV

# ── Grid definitions (same as 1.grid_search.py) ────────────────────────────────
GRID = {
    "FS-BSM": [{"fourier_K": k} for k in [3, 5]],
    "TBATS":  [{"fourier_K": k, "p_ar": p} for k, p in itertools.product([3, 5], [1, 2])],
}

STARS = {
    "HADS": sorted((DATA_DIR / "hads").glob("*.dat")),
    "LADS": sorted((DATA_DIR / "lads").glob("*.dat")),
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def freq_bounds(t, star_type):
    dt   = np.median(np.diff(t))
    fmax = 0.5 / dt
    fmin = 1.0 / (t[-1] - t[0])
    return fmin, fmax

def run_method(method, params, t, y_missing, start, end, fmin, fmax):
    try:
        if method == "FS-BSM":
            y_imp = mh.trig_ss_impute_block(t, y_missing, start, end,
                                             fmin=fmin, fmax=fmax,
                                             M=params["fourier_K"], maxiter=20)
        elif method == "TBATS":
            y_imp = mh.tbats_ss_impute_block(t, y_missing, start, end,
                                              fmin=fmin, fmax=fmax,
                                              M=params["fourier_K"],
                                              p_ar=params.get("p_ar", 1), maxiter=100)
        else:
            return None
        return np.asarray(y_imp, float)
    except Exception as e:
        print(f"      ✗ {method} error: {e}")
        return None

def save_plot(t, y_true, y_imp, start, end, star_name, method, metrics, out_dir, fold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pad = 200
    left  = max(0, start - pad)
    right = min(len(t), end + pad)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t[left:right], y_true[left:right], 'k.', alpha=0.3, ms=2, label='Original')
    ax.plot(t[start:end],  y_true[start:end],  'b-', alpha=0.6, lw=1,   label='Ground truth')
    ax.plot(t[start:end],  y_imp[start:end],   'r-',             lw=1.5, label=method)
    ax.axvspan(t[start], t[end-1], color='yellow', alpha=0.15)
    ax.set_xlabel('Time (HJD)'); ax.set_ylabel('Flux')
    ax.set_title(f"{star_name} – {method}  |  R²={metrics['R2']:.3f}  MAE={metrics['MAE']:.1f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{star_name.replace(' ','_')}_{method}_fold{fold}.png"
    fig.tight_layout(); fig.savefig(out_dir / fname, dpi=150); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Grid Search (only FS-BSM / TBATS)
# ══════════════════════════════════════════════════════════════════════════════
def run_grid_search():
    print("\n" + "="*70)
    print("  STEP 1 – GRID SEARCH (FS-BSM & TBATS warm-start)")
    print("="*70)

    GRID_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load existing grid CSV; drop FS-BSM and TBATS rows
    if GRID_CSV.exists():
        df_old = pd.read_csv(GRID_CSV)
        df_keep = df_old[~df_old['Method'].isin(TARGET_METHODS)]
        print(f"  Kept {len(df_keep)} existing rows; dropped {len(df_old)-len(df_keep)} Trig/TBATS rows.")
    else:
        df_keep = pd.DataFrame()

    new_rows = []
    set_seed(SEED_BASE)

    for star_type, star_files in STARS.items():
        fmin_global = fmax_global = None
        for star_path in star_files:
            star_file = star_path.name
            print(f"\n  [{star_type}] {star_file}")
            try:
                t, y_true = load_dat(star_path)
            except Exception as e:
                print(f"    ✗ Load error: {e}"); continue

            fmin_f, fmax_f = freq_bounds(t, star_type)

            # Use fold 0 for grid search (same as original pipeline)
            fold_seed = FOLD_SEEDS[0]
            set_seed(fold_seed)
            y_missing, start, end = simulate_block_run(y_true, block_n=NBLOCK, seed=fold_seed)

            for method, param_list in GRID.items():
                best_r2, best_row = -np.inf, None
                for params in param_list:
                    params_key = json.dumps(params, sort_keys=True)
                    t0 = time.time()
                    y_imp = run_method(method, params, t.copy(), y_missing.copy(),
                                       start, end, fmin_f, fmax_f)
                    elapsed = time.time() - t0
                    if y_imp is None: continue
                    m = metrics_from_block(y_true, y_imp, start, end)
                    row = {
                        'StarFile': star_file, 'StarType': star_type,
                        'Scenario': 'block_runs', 'Method': method,
                        'Params': params_key,
                        'MAE': m['MAE'], 'MSE': m['MSE'],
                        'MAPE': m['MAPE'], 'R2': m['R2'],
                        'Time': elapsed,
                    }
                    new_rows.append(row)
                    if m['R2'] > best_r2:
                        best_r2 = m['R2']; best_row = row
                    print(f"    {method} {params_key[:40]}  R²={m['R2']:.4f}  {elapsed:.1f}s")

    df_new = pd.DataFrame(new_rows)
    df_all = pd.concat([df_keep, df_new], ignore_index=True)
    df_all.to_csv(GRID_CSV, index=False)
    print(f"\n  ✓ Grid CSV updated → {GRID_CSV} ({len(df_all)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Cross-Validation (only FS-BSM / TBATS, all 5 folds)
# ══════════════════════════════════════════════════════════════════════════════
def run_cross_validation():
    print("\n" + "="*70)
    print("  STEP 2 – CROSS-VALIDATION (FS-BSM & TBATS warm-start, 5 folds)")
    print("="*70)

    # Load grid to get best params per star/method
    if not GRID_CSV.exists():
        print("  ✗ Grid CSV not found – run grid search first."); return
    df_grid = pd.read_csv(GRID_CSV)

    # Load existing CV CSV; drop FS-BSM and TBATS rows
    if CV_CSV.exists():
        df_old_cv = pd.read_csv(CV_CSV)
        df_keep_cv = df_old_cv[~df_old_cv['Method'].isin(TARGET_METHODS)]
        print(f"  Kept {len(df_keep_cv)} CV rows; dropped {len(df_old_cv)-len(df_keep_cv)} rows.")
    else:
        df_keep_cv = pd.DataFrame()

    new_cv_rows = []

    for star_type, star_files in STARS.items():
        for star_path in star_files:
            star_file = star_path.name
            print(f"\n  [{star_type}] {star_file}")
            try:
                t, y_true = load_dat(star_path)
            except Exception as e:
                print(f"    ✗ Load error: {e}"); continue

            fmin_f, fmax_f = freq_bounds(t, star_type)

            for method in TARGET_METHODS:
                # Get best params from grid search for this star/method
                df_star = df_grid[
                    (df_grid['StarFile'] == star_file) &
                    (df_grid['Method'] == method)
                ]
                if df_star.empty:
                    print(f"    ⚠ No grid results for {method}, skipping.")
                    continue
                best_idx  = df_star['R2'].idxmax()
                best_params = json.loads(df_star.loc[best_idx, 'Params'])

                for fold, fold_seed in enumerate(FOLD_SEEDS):
                    set_seed(fold_seed)
                    y_missing, start, end = simulate_block_run(
                        y_true, block_n=NBLOCK, seed=fold_seed)

                    t0 = time.time()
                    y_imp = run_method(method, best_params, t.copy(), y_missing.copy(),
                                       start, end, fmin_f, fmax_f)
                    elapsed = time.time() - t0

                    if y_imp is None:
                        print(f"    ✗ {method} fold {fold} failed"); continue

                    m = metrics_from_block(y_true, y_imp, start, end)
                    print(f"    {method} fold {fold}  R²={m['R2']:.4f}  MAE={m['MAE']:.1f}  {elapsed:.1f}s")

                    new_cv_rows.append({
                        'StarFile': star_file, 'StarType': star_type,
                        'Method': method, 'Fold': fold,
                        'Params': json.dumps(best_params, sort_keys=True),
                        'MAE': m['MAE'], 'MSE': m['MSE'],
                        'MAPE': m['MAPE'], 'R2': m['R2'],
                        'Time': elapsed,
                    })

                    # Save plot
                    out_dir = PLOTS_HADS if star_type == "HADS" else PLOTS_LADS
                    try:
                        save_plot(t, y_true, y_imp, start, end,
                                  star_file.replace('.dat',''), method, m, out_dir, fold)
                    except Exception as e:
                        print(f"    ⚠ Plot error: {e}")

    df_new_cv = pd.DataFrame(new_cv_rows)
    df_all_cv = pd.concat([df_keep_cv, df_new_cv], ignore_index=True)
    df_all_cv.to_csv(CV_CSV, index=False)
    print(f"\n  ✓ CV CSV updated → {CV_CSV} ({len(df_all_cv)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Regenerate analysis + tables
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis():
    import subprocess
    print("\n" + "="*70)
    print("  STEP 3 – Re-running analysis scripts")
    print("="*70)
    scripts = ["3.sensitivity_analysis.py", "4.analysis_and_plots.py", "7.generate_tables.py",
               "5.generate_grid_plots.py"]
    for s in scripts:
        sp = SCRIPT_DIR / s
        if sp.exists():
            print(f"  Running {s}...")
            r = subprocess.run(["python3", str(sp)], capture_output=True, text=True, cwd=str(SCRIPT_DIR))
            if r.returncode != 0:
                print(f"    ✗ Error:\n{r.stderr[:500]}")
            else:
                print(f"    ✓ Done")
        else:
            print(f"    ⚠ Not found: {s}")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_grid_search()
    run_cross_validation()
    run_analysis()
    print("\n✓ ALL DONE – FS-BSM & TBATS warm-start re-run complete.")
