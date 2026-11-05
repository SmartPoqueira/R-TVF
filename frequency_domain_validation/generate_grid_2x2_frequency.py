#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════════════════
  2x2 Frequency-Domain Comparison Grid Plot (600 DPI Publication Quality)
  Balanced 2 HADS and 2 LADS:
    - HD 146953 (HADS)
    - VZ Cnc (HADS)
    - TIC 242944780 (LADS)
    - TIC 452982723 (LADS)
  Integer X-axis ticks, f1 with exactly 3 decimals, legend 'R-TVF (Ours)'.
══════════════════════════════════════════════════════════════════════════════
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ANTONIO_DIR = BASE_DIR.parent / "pasar a Antonio"
ANTONIO_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR.parent.parent / "major review (do not touch)" / "estrellas_test"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent.parent / "first_submission (do not touch)" / "estrellas_test"

sys.path.insert(0, str(BASE_DIR))
import validate_frequency_comparison as vfc

# 4 Balanced Stars for the 2x2 Grid (2 HADS, 2 LADS)
GRID_STARS = [
    {"type": "HADS", "folder": "hads", "file": "HD 146953lc.dat", "name": "HD 146953", "fmin": 3.0, "fmax": 25.0},
    {"type": "HADS", "folder": "hads", "file": "VZ Cnclc.dat", "name": "VZ Cnc", "fmin": 3.0, "fmax": 25.0},
    {"type": "LADS", "folder": "lads", "file": "TIC 242944780lc.dat", "name": "TIC 242944780", "fmin": 40.0, "fmax": 70.0},
    {"type": "LADS", "folder": "lads", "file": "TIC 452982723lc.dat", "name": "TIC 452982723", "fmin": 8.0, "fmax": 28.0},
]

COLOR_GROUND_TRUTH = '#000000'  # Solid Black
COLOR_UNIMPUTED    = '#f77f00'  # Orange
COLOR_RTVF         = '#e63946'  # Crimson Red
COLOR_PEAK_LINE    = '#2a9d8f'  # Teal Green

def main():
    print("=" * 80)
    print("  GENERATING 2x2 BALANCED (2 HADS, 2 LADS) FREQUENCY SPECTRUM GRID")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    sub_len = 4500
    gap_start = 500
    gap_len = 1000
    gap_end = gap_start + gap_len
    
    for idx, star in enumerate(GRID_STARS):
        ax = axes[idx]
        sname = star["name"]
        stype = star["type"]
        sfile = star["file"]
        sfolder = star["folder"]
        fmin = star["fmin"]
        fmax = star["fmax"]
        
        file_path = DATA_DIR / sfolder / sfile
        data = np.loadtxt(file_path)
        t_full, y_full = data[:, 0], data[:, 1]
        
        # Global R-TVF fit
        y_gapped_full = y_full.copy()
        y_gapped_full[gap_start:gap_end] = np.nan
        y_rec_full, _, _, _ = vfc.rtvf_impute(
            t_full, y_gapped_full, gap_start, gap_end, K=30, degree=1, l2=0.1, fmin=fmin, fmax=fmax
        )
        
        # Extract subportion
        t = t_full[:sub_len]
        y_orig = y_full[:sub_len]
        y_gapped = y_gapped_full[:sub_len]
        y_rec = y_rec_full[:sub_len]
        
        # Periodograms
        freq_band, p_orig = vfc.compute_ls_spectrum(t, y_orig, fmin, fmax, samples_per_peak=15)
        _, p_gapped = vfc.compute_ls_spectrum(t, y_gapped, fmin, fmax, samples_per_peak=15)
        _, p_rtvf = vfc.compute_ls_spectrum(t, y_rec, fmin, fmax, samples_per_peak=15)
        
        f_dom = float(freq_band[np.argmax(p_orig)])
        
        # Plot curves
        ax.plot(freq_band, p_orig, color=COLOR_GROUND_TRUTH, lw=2.5, label='Ground Truth', alpha=0.95, zorder=3)
        ax.plot(freq_band, p_gapped, color=COLOR_UNIMPUTED, lw=1.8, label='Unimputed', alpha=0.9, zorder=2)
        ax.plot(freq_band, p_rtvf, color=COLOR_RTVF, lw=1.8, label='R-TVF (Ours)', alpha=0.95, zorder=4)
        
        # Marker for dominant mode (3 decimals)
        ax.axvline(f_dom, color=COLOR_PEAK_LINE, ls='--', alpha=0.6, lw=1.4)
        ax.text(f_dom + 0.015*(fmax-fmin), max(p_orig)*0.88, f"$f_1 = {f_dom:.3f}$ c/d", 
                color=COLOR_PEAK_LINE, fontsize=12.5, fontweight='bold')
        
        # Integer X-axis tick format
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=7))
        
        # Styling & Limits
        ax.set_xlim(fmin, fmax)
        ax.set_xlabel('Frequency (cycles/day)', fontsize=14, fontweight='medium')
        ax.set_ylabel('Power', fontsize=14, fontweight='medium')
        ax.set_title(f"{sname} ({stype})", fontsize=15.5, fontweight='bold', pad=10)
        ax.tick_params(axis='both', which='major', labelsize=12.5)
        ax.legend(loc='upper right', frameon=True, fontsize=12.5, framealpha=0.92)
        ax.grid(True, linestyle='--', alpha=0.45)
        
        print(f"[{idx+1}/4] {sname} ({stype}) processed | f1 = {f_dom:.3f} c/d")
        
    fig.tight_layout(pad=3.0)
    
    out_plots_png = PLOTS_DIR / "grid_2x2_frequency_spectrum.png"
    out_antonio_png = ANTONIO_DIR / "grid_2x2_frequency_spectrum.png"
    out_pdf = PLOTS_DIR / "grid_2x2_frequency_spectrum.pdf"
    
    fig.savefig(out_plots_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_antonio_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved updated 600 DPI 2x2 Grid Plots to:\n- {out_plots_png}\n- {out_antonio_png}\n- {out_pdf}")

if __name__ == "__main__":
    main()
