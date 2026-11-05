import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Allow importing from src
sys.path.append(str(Path(__file__).parent))
from runner_comparativa_3x2 import load_dat, simulate_block_run, metrics_from_block
import modelos_hads_lads as mh

def run_test():
    star_file = "TIC 137796620lc.dat"
    file_path = Path(__file__).resolve().parent.parent / "db" / "lads" / star_file
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    t, y_true = load_dat(file_path)
    
    # 2. Simulate 1000 block gap
    y_missing, start, end = simulate_block_run(y_true, block_n=1000, seed=42)
    
    # 3. Define params
    fmin, fmax = 5.0, 80.0
    M = 5
    p_ar = 1
    degree = 3
    l2 = 1e-2
    
    results = []
    print(f"Executing test for {star_file} (LADS)...")
    
    # ----- 1. R-TVF (Separated times) -----
    print(f"\n--- Running R-TVF ---")
    # Measure just Periodogram time
    t0_ls = time.perf_counter()
    mask = np.isfinite(y_missing)
    freqs = mh.ls_top_freqs(t[mask], y_missing[mask], M=M, fmin=fmin, fmax=fmax)
    time_ls = time.perf_counter() - t0_ls
    
    # Measure total execution
    t0_total = time.perf_counter()
    y_rtvf, _ = mh.tvar_fourier_impute_block_multi(
        t, y_missing, start, end, M=M, degree=degree, l2=l2, fmin=fmin, fmax=fmax, max_samples_ls=60000
    )
    time_total = time.perf_counter() - t0_total
    
    metrics = metrics_from_block(y_true, y_rtvf, start, end)
    results.append({
        "Method": "R-TVF",
        "MAE": metrics["MAE"], "R2": metrics["R2"],
        "Time_LS (s)": round(time_ls, 3),
        "Time_Fit (s)": round(time_total - time_ls, 3),
        "Time_Total (s)": round(time_total, 3)
    })
    
    # ----- 2. DHR -----
    print(f"--- Running DHR ---")
    t0 = time.perf_counter()
    y_dhr = mh.dhr_impute_block(t, y_missing, start, end, fmin=fmin, fmax=fmax, M=M, p_ar=2, maxiter=20)
    time_dhr = time.perf_counter() - t0
    metrics = metrics_from_block(y_true, y_dhr, start, end)
    results.append({
        "Method": "DHR", "MAE": metrics["MAE"], "R2": metrics["R2"],
        "Time_LS (s)": "-", "Time_Fit (s)": "-", "Time_Total (s)": round(time_dhr, 3)
    })

    # ----- 3. FS-BSM -----
    print(f"--- Running FS-BSM ---")
    t0 = time.perf_counter()
    y_trig = mh.trig_ss_impute_block(t, y_missing, start, end, fmin=fmin, fmax=fmax, M=M, maxiter=20)
    time_trig = time.perf_counter() - t0
    metrics = metrics_from_block(y_true, y_trig, start, end)
    results.append({
        "Method": "FS-BSM", "MAE": metrics["MAE"], "R2": metrics["R2"],
        "Time_LS (s)": "-", "Time_Fit (s)": "-", "Time_Total (s)": round(time_trig, 3)
    })

    # ----- 4. TBATS -----
    print(f"--- Running TBATS ---")
    t0 = time.perf_counter()
    y_tbats = mh.tbats_ss_impute_block(t, y_missing, start, end, fmin=fmin, fmax=fmax, M=M, p_ar=1, maxiter=20)
    time_tbats = time.perf_counter() - t0
    metrics = metrics_from_block(y_true, y_tbats, start, end)
    results.append({
        "Method": "TBATS", "MAE": metrics["MAE"], "R2": metrics["R2"],
        "Time_LS (s)": "-", "Time_Fit (s)": "-", "Time_Total (s)": round(time_tbats, 3)
    })
    
    df = pd.DataFrame(results)
    print("\n================== RESULTS TABLE ==================")
    print(df.to_markdown(index=False))
    print("===================================================")
    
if __name__ == "__main__":
    run_test()
