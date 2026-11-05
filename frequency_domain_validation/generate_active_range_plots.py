#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Active Pulsation Range Plots for Antonio
Focusing on the actual stellar pulsation band so all modes and sidelobes are clearly visible.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR.parent / "pasar a Antonio"

DATA_DIR = BASE_DIR.parent.parent / "major review (do not touch)" / "estrellas_test"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent.parent / "first_submission (do not touch)" / "estrellas_test"

sys.path.insert(0, str(BASE_DIR))
import validate_frequency_comparison as vfc

# Active pulsation bands for crystal clear visibility
STARS = [
    {"type": "HADS", "file": "AD Cmilc.dat", "name": "AD CMi", "fmin": 3.0, "fmax": 30.0, "f_plot_min": 6.0, "f_plot_max": 26.0, "start": 500, "span": 1000},
    {"type": "HADS", "file": "HD 146953lc.dat", "name": "HD 146953", "fmin": 3.0, "fmax": 30.0, "f_plot_min": 3.0, "f_plot_max": 25.0, "start": 500, "span": 1000},
    {"type": "LADS", "file": "TIC 100531058lc.dat", "name": "TIC 100531058", "fmin": 5.0, "fmax": 80.0, "f_plot_min": 45.0, "f_plot_max": 72.0, "start": 500, "span": 1000},
    {"type": "LADS", "file": "TIC 246902545lc.dat", "name": "TIC 246902545", "fmin": 5.0, "fmax": 80.0, "f_plot_min": 35.0, "f_plot_max": 65.0, "start": 500, "span": 1000},
    {"type": "LADS", "file": "TIC 452982723lc.dat", "name": "TIC 452982723", "fmin": 5.0, "fmax": 80.0, "f_plot_min": 8.0, "f_plot_max": 28.0, "start": 500, "span": 1000},
]

COLOR_GROUND_TRUTH = '#000000'
COLOR_UNIMPUTED    = '#f77f00'
COLOR_RTVF         = '#e63946'

def main():
    for idx, star in enumerate(STARS):
        sname = star["name"]
        stype = star["type"]
        sfile = star["file"]
        start = star["start"]
        end = start + star["span"]
        fmin_plot = star["f_plot_min"]
        fmax_plot = star["f_plot_max"]
        
        file_path = DATA_DIR / stype.lower() / sfile
        data = np.loadtxt(file_path)
        t, y_orig = data[:, 0], data[:, 1]
        N = len(t)
        
        y_gapped = y_orig.copy()
        y_gapped[start:end] = np.nan
        
        y_rtvf, y_hat, y_std, freqs = vfc.rtvf_impute(
            t, y_gapped, start, end, K=30, degree=1, l2=0.1, fmin=star["fmin"], fmax=star["fmax"]
        )
        
        freq_band, p_orig_band = vfc.compute_ls_spectrum(t, y_orig, fmin=fmin_plot, fmax=fmax_plot, samples_per_peak=15)
        _, p_gapped_band = vfc.compute_ls_spectrum(t, y_gapped, fmin=fmin_plot, fmax=fmax_plot, samples_per_peak=15)
        _, p_rtvf_band = vfc.compute_ls_spectrum(t, y_rtvf, fmin=fmin_plot, fmax=fmax_plot, samples_per_peak=15)
        
        fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={'height_ratios': [1, 1.25]})
        
        pad = 600
        t_l = max(0, start - pad)
        t_r = min(N, end + pad)
        ax_t.plot(t[t_l:t_r], y_orig[t_l:t_r], color='#6c757d', alpha=0.45, lw=1.0, label='Ground Truth')
        ax_t.axvspan(t[start], t[end-1], color='#ffeaa7', alpha=0.5, label='Gap')
        ax_t.plot(t[start:end], y_rtvf[start:end], color=COLOR_RTVF, lw=1.6, label='R-TVF')
        ax_t.fill_between(
            t[start:end],
            y_rtvf[start:end] - 3.0 * y_std[start:end],
            y_rtvf[start:end] + 3.0 * y_std[start:end],
            color=COLOR_RTVF, alpha=0.22, label='3σ Band'
        )
        ax_t.set_xlabel('Time (BJD)')
        ax_t.set_ylabel('Flux ($e^-$/s)')
        ax_t.set_title(f"{sname} ({stype}) — Time-Domain Reconstruction", fontweight='bold')
        ax_t.legend(loc='upper right', frameon=True, fontsize=10)
        ax_t.grid(True, linestyle='--', alpha=0.4)
        
        ax_f.plot(freq_band, p_orig_band, color=COLOR_GROUND_TRUTH, lw=2.2, label='Ground Truth', alpha=0.95, zorder=3)
        ax_f.plot(freq_band, p_gapped_band, color=COLOR_UNIMPUTED, lw=1.6, label='Unimputed', alpha=0.9, zorder=2)
        ax_f.plot(freq_band, p_rtvf_band, color=COLOR_RTVF, lw=1.6, label='R-TVF', alpha=0.95, zorder=4)
        ax_f.set_xlim(fmin_plot, fmax_plot)
        ax_f.set_xlabel('Frequency (cycles/day)')
        ax_f.set_ylabel('Power')
        ax_f.set_title(f"Active Pulsation Spectrum Range ({fmin_plot:.0f} to {fmax_plot:.0f} cycles/day)", fontsize=11.5, fontweight='bold')
        ax_f.legend(loc='upper right', frameon=True, fontsize=10)
        ax_f.grid(True, linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        safe_name = sname.replace(" ", "_")
        out_path = OUT_DIR / f"RANGO_ACTIVO_{safe_name}_{stype}.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Generated active range plot: {out_path.name}")

if __name__ == "__main__":
    main()
