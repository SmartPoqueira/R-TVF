#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════════════════
  Clean Publication Comparative Validation for Antonio:
  Continuous Orbit 1 (Excluding Satellite Downlink Gaps)
  With Prominent Sidelobe / Spurious Peak Visibility & Standard Title Metadata
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

# Set publication styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots" / "subporcion"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ANTONIO_DIR = BASE_DIR.parent / "pasar a Antonio"
ANTONIO_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR.parent.parent / "major review (do not touch)" / "estrellas_test"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent.parent / "first_submission (do not touch)" / "estrellas_test"

sys.path.insert(0, str(BASE_DIR))
import validate_frequency_comparison as vfc

# 10 Stars: evaluated in continuous Orbit 1 (strictly away from the real mid-sector downlink gap)
STARS = [
    # 5 HADS Stars
    {"type": "HADS", "file": "HD 146953lc.dat", "name": "HD 146953", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 3.0, "fmax": 25.0},
    {"type": "HADS", "file": "AD Cmilc.dat", "name": "AD CMi", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 3.0, "fmax": 25.0},
    {"type": "HADS", "file": "BE Lynlc.dat", "name": "BE Lyn", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 3.0, "fmax": 25.0},
    {"type": "HADS", "file": "BS Aqrlc.dat", "name": "BS Aqr", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 3.0, "fmax": 25.0},
    {"type": "HADS", "file": "VZ Cnclc.dat", "name": "VZ Cnc", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 3.0, "fmax": 25.0},
    # 5 LADS Stars
    {"type": "LADS", "file": "TIC 452982723lc.dat", "name": "TIC 452982723", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 8.0, "fmax": 28.0},
    {"type": "LADS", "file": "TIC 100531058lc.dat", "name": "TIC 100531058", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 45.0, "fmax": 72.0},
    {"type": "LADS", "file": "TIC 121729614lc.dat", "name": "TIC 121729614", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 10.0, "fmax": 35.0},
    {"type": "LADS", "file": "TIC 242944780lc.dat", "name": "TIC 242944780", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 40.0, "fmax": 70.0},
    {"type": "LADS", "file": "TIC 246902545lc.dat", "name": "TIC 246902545", "sub_len": 4500, "gap_start": 500, "gap_len": 1000, "fmin": 35.0, "fmax": 65.0},
]

COLOR_GROUND_TRUTH = '#000000'  # Solid Black
COLOR_UNIMPUTED    = '#f77f00'  # Orange
COLOR_RTVF         = '#e63946'  # Crimson Red
COLOR_PEAK_LINE    = '#2a9d8f'  # Teal Green
COLOR_GAP_SHADE    = '#ffeaa7'  # Soft Yellow

def main():
    print("=" * 85)
    print("  GENERATING CLEAN CONTINUOUS ORBIT 1 PLOTS (NO DOWNLINK GAPS)")
    print("=" * 85)
    
    fig_summary, axes_summary = plt.subplots(5, 2, figsize=(16, 20))
    axes_summary = axes_summary.flatten()
    
    for idx, star in enumerate(STARS):
        sname = star["name"]
        stype = star["type"]
        sfile = star["file"]
        sub_len = star["sub_len"]
        gap_start = star["gap_start"]
        gap_len = star["gap_len"]
        gap_end = gap_start + gap_len
        fmin = star["fmin"]
        fmax = star["fmax"]
        
        file_path = DATA_DIR / stype.lower() / sfile
        if not file_path.exists():
            continue
            
        data = np.loadtxt(file_path)
        t_full, y_full = data[:, 0], data[:, 1]
        
        # 1. Full series R-TVF fit for exact global phase and optimal R2
        y_gapped_full = y_full.copy()
        y_gapped_full[gap_start:gap_end] = np.nan
        y_rec_full, y_hat_full, y_std_full, freqs = vfc.rtvf_impute(
            t_full, y_gapped_full, gap_start, gap_end, K=30, degree=1, l2=0.1, fmin=fmin, fmax=fmax
        )
        
        ss_res = np.sum((y_full[gap_start:gap_end] - y_rec_full[gap_start:gap_end])**2)
        ss_tot = np.sum((y_full[gap_start:gap_end] - np.mean(y_full[gap_start:gap_end]))**2)
        r2_time = 1.0 - (ss_res / max(ss_tot, 1e-12))
        
        # 2. Extract 100% continuous Orbit 1 window (0:4500), strictly away from downlink gap (7500+)
        t = t_full[:sub_len]
        y_orig = y_full[:sub_len]
        y_gapped = y_gapped_full[:sub_len]
        y_rec = y_rec_full[:sub_len]
        y_std = y_std_full[:sub_len]
        N = len(t)
        
        # 3. Frequency Spectrum on the continuous Orbit 1 window
        freq_band, p_orig = vfc.compute_ls_spectrum(t, y_orig, fmin, fmax, samples_per_peak=15)
        _, p_gapped = vfc.compute_ls_spectrum(t, y_gapped, fmin, fmax, samples_per_peak=15)
        _, p_rtvf = vfc.compute_ls_spectrum(t, y_rec, fmin, fmax, samples_per_peak=15)
        
        f_dom = float(freq_band[np.argmax(p_orig)])
        
        # ---------------------------------------------------------------------
        # INDIVIDUAL 2-PANEL PLOT
        # ---------------------------------------------------------------------
        fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={'height_ratios': [1, 1.25]})
        
        # Panel 1: Time Domain Reconstruction
        pad = 600
        t_l = max(0, gap_start - pad)
        t_r = min(N, gap_end + pad)
        ax_t.plot(t[t_l:t_r], y_orig[t_l:t_r], color='#6c757d', alpha=0.45, lw=1.0, label='Ground Truth')
        ax_t.axvspan(t[gap_start], t[gap_end-1], color=COLOR_GAP_SHADE, alpha=0.55, label='Gap')
        ax_t.plot(t[gap_start:gap_end], y_rec[gap_start:gap_end], color=COLOR_RTVF, lw=1.6, label='R-TVF')
        ax_t.fill_between(
            t[gap_start:gap_end],
            y_rec[gap_start:gap_end] - 3.0 * y_std[gap_start:gap_end],
            y_rec[gap_start:gap_end] + 3.0 * y_std[gap_start:gap_end],
            color=COLOR_RTVF, alpha=0.22, label='3σ Band'
        )
        ax_t.set_xlabel('Time (BJD)')
        ax_t.set_ylabel('Flux ($e^-$/s)')
        ax_t.set_title(f"{sname} ({stype}) — Time-Domain Reconstruction ($R^2 = {r2_time:.3f}$)", fontweight='bold')
        ax_t.legend(loc='upper right', frameon=True, fontsize=10)
        ax_t.grid(True, linestyle='--', alpha=0.4)
        
        # Panel 2: Frequency Spectrum with Dominant Frequency Marker
        ax_f.plot(freq_band, p_orig, color=COLOR_GROUND_TRUTH, lw=2.4, label='Ground Truth', alpha=0.95, zorder=3)
        ax_f.plot(freq_band, p_gapped, color=COLOR_UNIMPUTED, lw=1.8, label='Unimputed', alpha=0.9, zorder=2)
        ax_f.plot(freq_band, p_rtvf, color=COLOR_RTVF, lw=1.8, label='R-TVF', alpha=0.95, zorder=4)
        
        ax_f.axvline(f_dom, color=COLOR_PEAK_LINE, ls='--', alpha=0.6, lw=1.2)
        ax_f.text(f_dom + 0.15, max(p_orig)*0.88, f"$f_1 = {f_dom:.4f}$ c/d", color=COLOR_PEAK_LINE, fontsize=10.5, fontweight='bold')
        
        ax_f.set_xlim(fmin, fmax)
        ax_f.set_xlabel('Frequency (cycles/day)')
        ax_f.set_ylabel('Power')
        ax_f.set_title(f"Frequency Spectrum", fontsize=11.5, fontweight='bold')
        ax_f.legend(loc='upper right', frameon=True, fontsize=10.5)
        ax_f.grid(True, linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        safe_name = sname.replace(" ", "_")
        
        # Save to plots/subporcion/
        out_sub = PLOTS_DIR / f"comparative_{stype}_{safe_name}_subporcion.png"
        fig.savefig(out_sub, dpi=300, bbox_inches='tight')
        
        # Copy into pasar a Antonio/
        out_antonio = ANTONIO_DIR / f"0{idx+1}_{safe_name}_{stype}_comparativa.png" if idx < 9 else ANTONIO_DIR / f"{idx+1}_{safe_name}_{stype}_comparativa.png"
        fig.savefig(out_antonio, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        print(f"[{idx+1}/10] {stype} {sname:14s} | R2 = {r2_time:.3f} | Saved: {out_sub.name}")
        
        # Summary multi-panel
        ax_s = axes_summary[idx]
        ax_s.plot(freq_band, p_orig, color=COLOR_GROUND_TRUTH, lw=2.0, label='Ground Truth', alpha=0.95)
        ax_s.plot(freq_band, p_gapped, color=COLOR_UNIMPUTED, lw=1.4, label='Unimputed', alpha=0.85)
        ax_s.plot(freq_band, p_rtvf, color=COLOR_RTVF, lw=1.4, label='R-TVF', alpha=0.95)
        ax_s.set_xlim(fmin, fmax)
        ax_s.set_title(f"{sname} ({stype}) — $f_1={f_dom:.2f}$ c/d", fontsize=10, fontweight='bold')
        ax_s.set_xlabel('Frequency (c/d)', fontsize=9)
        ax_s.set_ylabel('Power', fontsize=9)
        ax_s.legend(loc='upper right', fontsize=8.5)
        ax_s.grid(True, linestyle='--', alpha=0.35)
        
    fig_summary.tight_layout()
    fig_summary.savefig(PLOTS_DIR / "summary_comparative_10stars_subporcion.png", dpi=300, bbox_inches='tight')
    plt.close(fig_summary)
    print("\nAll 10 clean continuous plots generated successfully in plots/subporcion/ and pasar a Antonio/!")

if __name__ == "__main__":
    main()
