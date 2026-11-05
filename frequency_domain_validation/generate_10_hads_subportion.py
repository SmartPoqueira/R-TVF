#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════════════════
  Top 10 HADS Stars Sub-portion Comparative Validation (All R2 > 0.985)
  Continuous Orbit 1 (Excluding Downlink Gaps)
  Saves to plots/subporcion/hads_10/ and pasar a Antonio/HADS_10/
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
PLOTS_DIR = BASE_DIR / "plots" / "subporcion" / "hads_10"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ANTONIO_HADS_DIR = BASE_DIR.parent / "pasar a Antonio" / "HADS_10"
ANTONIO_HADS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR.parent.parent / "major review (do not touch)" / "estrellas_test" / "hads"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent.parent / "first_submission (do not touch)" / "estrellas_test" / "hads"

sys.path.insert(0, str(BASE_DIR))
import validate_frequency_comparison as vfc

# Top 10 HADS Stars
HADS_STARS = [
    {"file": "AD Cmilc.dat", "name": "AD CMi", "fmin_plot": 3.0, "fmax_plot": 25.0},
    {"file": "HD 146953lc.dat", "name": "HD 146953", "fmin_plot": 3.0, "fmax_plot": 25.0},
    {"file": "BE Lynlc.dat", "name": "BE Lyn", "fmin_plot": 3.0, "fmax_plot": 25.0},
    {"file": "BS Aqrlc.dat", "name": "BS Aqr", "fmin_plot": 3.0, "fmax_plot": 25.0},
    {"file": "DX Cetlc.dat", "name": "DX Cet", "fmin_plot": 3.0, "fmax_plot": 25.0},
    {"file": "DE Laclc.dat", "name": "DE Lac", "fmin_plot": 2.0, "fmax_plot": 20.0},
    {"file": "GW Umalc.dat", "name": "GW UMa", "fmin_plot": 2.0, "fmax_plot": 20.0},
    {"file": "V367 Camlc.dat", "name": "V367 Cam", "fmin_plot": 3.0, "fmax_plot": 25.0},
    {"file": "V 524 Andlc.dat", "name": "V524 And", "fmin_plot": 3.0, "fmax_plot": 25.0},
    {"file": "V1719 Cyglc.dat", "name": "V1719 Cyg", "fmin_plot": 3.0, "fmax_plot": 25.0},
]

COLOR_GROUND_TRUTH = '#000000'
COLOR_UNIMPUTED    = '#f77f00'
COLOR_RTVF         = '#e63946'
COLOR_PEAK_LINE    = '#2a9d8f'
COLOR_GAP_SHADE    = '#ffeaa7'

def main():
    print("=" * 85)
    print("  GENERATING TOP 10 HADS SUBPORTION COMPARATIVE PLOTS (ALL R2 > 0.985)")
    print("=" * 85)
    
    fig_summary, axes_summary = plt.subplots(5, 2, figsize=(16, 20))
    axes_summary = axes_summary.flatten()
    
    for idx, star in enumerate(HADS_STARS):
        sname = star["name"]
        sfile = star["file"]
        sub_len = 4500
        gap_start = 500
        gap_len = 1000
        gap_end = gap_start + gap_len
        fmin_plot = star["fmin_plot"]
        fmax_plot = star["fmax_plot"]
        
        file_path = DATA_DIR / sfile
        if not file_path.exists():
            continue
            
        data = np.loadtxt(file_path)
        t_full, y_full = data[:, 0], data[:, 1]
        
        # 1. Full-series global fit
        y_gapped_full = y_full.copy()
        y_gapped_full[gap_start:gap_end] = np.nan
        y_rec_full, _, y_std_full, _ = vfc.rtvf_impute(
            t_full, y_gapped_full, gap_start, gap_end, K=30, degree=1, l2=0.1, fmin=2.0, fmax=35.0
        )
        
        ss_res = np.sum((y_full[gap_start:gap_end] - y_rec_full[gap_start:gap_end])**2)
        ss_tot = np.sum((y_full[gap_start:gap_end] - np.mean(y_full[gap_start:gap_end]))**2)
        r2_time = 1.0 - (ss_res / max(ss_tot, 1e-12))
        
        # 2. Continuous Orbit 1 subportion
        t = t_full[:sub_len]
        y_orig = y_full[:sub_len]
        y_gapped = y_gapped_full[:sub_len]
        y_rec = y_rec_full[:sub_len]
        y_std = y_std_full[:sub_len]
        N = len(t)
        
        # 3. Frequency spectrum
        freq_band, p_orig = vfc.compute_ls_spectrum(t, y_orig, fmin_plot, fmax_plot, samples_per_peak=15)
        _, p_gapped = vfc.compute_ls_spectrum(t, y_gapped, fmin_plot, fmax_plot, samples_per_peak=15)
        _, p_rtvf = vfc.compute_ls_spectrum(t, y_rec, fmin_plot, fmax_plot, samples_per_peak=15)
        
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
        ax_t.set_title(f"{sname} (HADS) — Time-Domain Reconstruction ($R^2 = {r2_time:.3f}$)", fontweight='bold')
        ax_t.legend(loc='upper right', frameon=True, fontsize=10)
        ax_t.grid(True, linestyle='--', alpha=0.4)
        
        # Panel 2: Frequency Spectrum with Dominant Frequency Marker
        ax_f.plot(freq_band, p_orig, color=COLOR_GROUND_TRUTH, lw=2.4, label='Ground Truth', alpha=0.95, zorder=3)
        ax_f.plot(freq_band, p_gapped, color=COLOR_UNIMPUTED, lw=1.8, label='Unimputed', alpha=0.9, zorder=2)
        ax_f.plot(freq_band, p_rtvf, color=COLOR_RTVF, lw=1.8, label='R-TVF', alpha=0.95, zorder=4)
        
        ax_f.axvline(f_dom, color=COLOR_PEAK_LINE, ls='--', alpha=0.6, lw=1.2)
        ax_f.text(f_dom + 0.15, max(p_orig)*0.88, f"$f_1 = {f_dom:.4f}$ c/d", color=COLOR_PEAK_LINE, fontsize=10.5, fontweight='bold')
        
        ax_f.set_xlim(fmin_plot, fmax_plot)
        ax_f.set_xlabel('Frequency (cycles/day)')
        ax_f.set_ylabel('Power')
        ax_f.set_title(f"Frequency Spectrum", fontsize=11.5, fontweight='bold')
        ax_f.legend(loc='upper right', frameon=True, fontsize=10.5)
        ax_f.grid(True, linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        safe_name = sname.replace(" ", "_")
        
        # Save to plots/subporcion/hads_10/
        out_sub = PLOTS_DIR / f"comparative_HADS_{safe_name}_subporcion.png"
        fig.savefig(out_sub, dpi=300, bbox_inches='tight')
        
        # Save into pasar a Antonio/HADS_10/
        code_str = f"0{idx+1}" if idx < 9 else f"{idx+1}"
        out_antonio = ANTONIO_HADS_DIR / f"{code_str}_{safe_name}_HADS_comparativa.png"
        fig.savefig(out_antonio, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        print(f"[{idx+1}/10] HADS {sname:12s} | R2 = {r2_time:.4f} | f1 = {f_dom:.4f} c/d | Saved: {out_sub.name}")
        
        # Summary multi-panel
        ax_s = axes_summary[idx]
        ax_s.plot(freq_band, p_orig, color=COLOR_GROUND_TRUTH, lw=2.0, label='Ground Truth', alpha=0.95)
        ax_s.plot(freq_band, p_gapped, color=COLOR_UNIMPUTED, lw=1.4, label='Unimputed', alpha=0.85)
        ax_s.plot(freq_band, p_rtvf, color=COLOR_RTVF, lw=1.4, label='R-TVF', alpha=0.95)
        ax_s.set_xlim(fmin_plot, fmax_plot)
        ax_s.set_title(f"{sname} (HADS) — $f_1={f_dom:.2f}$ c/d ($R^2={r2_time:.3f}$)", fontsize=10, fontweight='bold')
        ax_s.set_xlabel('Frequency (c/d)', fontsize=9)
        ax_s.set_ylabel('Power', fontsize=9)
        ax_s.legend(loc='upper right', fontsize=8.5)
        ax_s.grid(True, linestyle='--', alpha=0.35)
        
    fig_summary.tight_layout()
    fig_summary.savefig(PLOTS_DIR / "summary_comparative_10_HADS_subporcion.png", dpi=300, bbox_inches='tight')
    fig_summary.savefig(ANTONIO_HADS_DIR / "00_Resumen_10_HADS_comparativa.png", dpi=300, bbox_inches='tight')
    plt.close(fig_summary)
    print("\nAll top 10 HADS subportion plots generated successfully!")

if __name__ == "__main__":
    main()
