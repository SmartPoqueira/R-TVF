import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import lightCurve as lc
import modelos_hads_lads as mh

def replot_rtvf():
    print("Re-plotting R-TVF with uncertainty bands for N=1000...")
    
    cv_csv = SCRIPT_DIR.parent / "results" / "2_cross_validation" / "cv_results_all.csv"
    if not cv_csv.exists():
        print(f"Error: {cv_csv} not found.")
        return
        
    df = pd.read_csv(cv_csv)
    df_rtvf = df[(df["Method"] == "R-TVF") & (df["Fold"] == 0) & (df["StarFile"].isin(["TIC 452982723lc.dat", "HD 146953lc.dat"]))].copy()
    
    hads_dir = SCRIPT_DIR.parent / "results" / "2_cross_validation" / "plots_hads"
    lads_dir = SCRIPT_DIR.parent / "results" / "2_cross_validation" / "plots_lads"
    
    # We only need fold 0 for the grid plots usually, but let's do all folds quickly
    
    for i, row in df_rtvf.iterrows():
        star = row["StarFile"]
        stype = row["StarType"]
        fold = int(row["Fold"])
        start = int(row["BlockStart"])
        end = int(row["BlockEnd"])
        metrics = {"MAE": row["MAE"], "R2": row["R2"]}
        params = eval(row["Params"])
        
        # Load star
        star_path = SCRIPT_DIR.parent / "estrellas_test" / stype.lower() / star
        if not star_path.exists(): continue
        
        df_star = mh.load_dat(str(star_path))
        t = df_star['time'].values
        y_true = df_star['flux'].values
        
        y_missing = y_true.copy()
        y_missing[start:end] = np.nan
        
        fmin, fmax = (3.0, 30.0) if stype == "HADS" else (5.0, 80.0)
        
        y_imp, _, y_std = mh.tvar_fourier_impute_block_multi(
            t, y_missing, start, end,
            M=int(params.get("fourier_K", 5)), degree=int(params.get("degree", 3)),
            trend_deg=1, l2=float(params.get("l2", 1e-2)),
            fmin=fmin, fmax=fmax, max_samples_ls=60000,
            return_std=True
        )
        
        # Plot
        out_dir = hads_dir if stype == "HADS" else lads_dir
        
        fig, ax = plt.subplots(figsize=(12, 5))
        pad = 200
        left = max(0, start - pad)
        right = min(len(t), end + pad)
        
        ax.plot(t[left:right], y_true[left:right], 'k.', alpha=0.3, markersize=2, label='Original', zorder=1)
        ax.plot(t[start:end], y_true[start:end], 'b-', alpha=0.6, linewidth=1, label='Ground truth', zorder=3)
        ax.plot(t[start:end], y_imp[start:end], 'r-', linewidth=1.5, label='R-TVF', zorder=4)
        
        # Band
        ax.fill_between(t[start:end], y_imp[start:end] - 3*y_std[start:end], 
                        y_imp[start:end] + 3*y_std[start:end], color='#FFB6C1', alpha=1.0, zorder=2, label='3σ Band')
                        
        ax.axvspan(t[start], t[end-1], color='yellow', alpha=0.15, zorder=0)
        
        ax.set_xlabel('Time (HJD)')
        ax.set_ylabel('Flux')
        ax.set_title(f"{star} - R-TVF - Fold {fold+1}\nMAE={metrics['MAE']:.3f} | R²={metrics['R2']:.3f}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        safe_star = star.replace(' ', '_').replace('.dat', '')
        filename = f"{safe_star}_R-TVF_fold{fold}.png"
        filepath = out_dir / filename
        
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        
        # Generate LARGE font version
        fig_l, ax_l = plt.subplots(figsize=(12, 6))
        FONT_AXIS = 18
        FONT_TICKS = 16
        FONT_LEGEND = 16
        FONT_TITLE = 12
        
        ax_l.plot(t[left:right], y_true[left:right], 'k.', alpha=0.3, markersize=4, label='Original', zorder=1)
        ax_l.plot(t[start:end], y_true[start:end], 'b-', alpha=0.6, linewidth=2.0, label='Ground truth', zorder=3)
        ax_l.plot(t[start:end], y_imp[start:end], 'r-', linewidth=2.5, label='R-TVF', zorder=4)
        
        # Band on large version
        ax_l.fill_between(t[start:end], y_imp[start:end] - 3*y_std[start:end], 
                         y_imp[start:end] + 3*y_std[start:end], color='#FFB6C1', alpha=1.0, zorder=2, label='3σ Band')
                         
        ax_l.axvspan(t[start], t[end-1], color='yellow', alpha=0.15, zorder=0)
        
        ax_l.set_xlabel('Time (HJD)', fontsize=FONT_AXIS)
        ax_l.set_ylabel('Flux', fontsize=FONT_AXIS)
        ax_l.set_title(f"{star.replace('lc','')} - R-TVF\nR²={metrics['R2']:.3f} | MAE={metrics['MAE']:.3f}", 
                     fontsize=FONT_TITLE, pad=12)
        ax_l.tick_params(axis='both', which='major', labelsize=FONT_TICKS)
        ax_l.legend(loc='upper right', fontsize=FONT_LEGEND, frameon=True, framealpha=0.9, fancybox=True)
        ax_l.grid(True, alpha=0.3)
        
        filename_l = f"{safe_star}_R-TVF_fold{fold}_large.png"
        filepath_l = out_dir / filename_l
        fig_l.tight_layout()
        fig_l.savefig(filepath_l, dpi=300)
        plt.close(fig_l)
        
        print(f"Updated {filename} and {filename_l}")

if __name__ == '__main__':
    replot_rtvf()
