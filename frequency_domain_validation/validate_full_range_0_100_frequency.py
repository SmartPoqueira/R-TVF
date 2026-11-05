#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════════════════
  Full-Range (0 - 100 cycles/day) Asteroseismic Frequency Validation:
  UNIMPUTED vs. R-TVF vs. GROUND TRUTH
  Evaluated on 10 Delta Scuti Stars (5 HADS, 5 LADS)
  Across N=1000 (~33.5 h) and N=2000 (~67.5 h)
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

# Publication styling
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
PLOTS_DIR = BASE_DIR / "plots"

DATA_DIR = BASE_DIR.parent.parent / "major review (do not touch)" / "estrellas_test"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent.parent / "first_submission (do not touch)" / "estrellas_test"

STARS_CONFIG = [
    # 5 HADS Stars
    {"type": "HADS", "file": "HD 146953lc.dat", "name": "HD 146953", "fmin": 3.0, "fmax": 30.0, "start_1000": 500, "start_2000": 500},
    {"type": "HADS", "file": "AD Cmilc.dat", "name": "AD CMi", "fmin": 3.0, "fmax": 30.0, "start_1000": 500, "start_2000": 500},
    {"type": "HADS", "file": "BE Lynlc.dat", "name": "BE Lyn", "fmin": 3.0, "fmax": 30.0, "start_1000": 500, "start_2000": 2050},
    {"type": "HADS", "file": "BS Aqrlc.dat", "name": "BS Aqr", "fmin": 3.0, "fmax": 30.0, "start_1000": 500, "start_2000": 500},
    {"type": "HADS", "file": "VZ Cnclc.dat", "name": "VZ Cnc", "fmin": 3.0, "fmax": 30.0, "start_1000": 500, "start_2000": 12900},
    # 5 LADS Stars
    {"type": "LADS", "file": "TIC 452982723lc.dat", "name": "TIC 452982723", "fmin": 5.0, "fmax": 80.0, "start_1000": 500, "start_2000": 12900},
    {"type": "LADS", "file": "TIC 100531058lc.dat", "name": "TIC 100531058", "fmin": 5.0, "fmax": 80.0, "start_1000": 500, "start_2000": 13100},
    {"type": "LADS", "file": "TIC 121729614lc.dat", "name": "TIC 121729614", "fmin": 5.0, "fmax": 80.0, "start_1000": 500, "start_2000": 500},
    {"type": "LADS", "file": "TIC 242944780lc.dat", "name": "TIC 242944780", "fmin": 5.0, "fmax": 80.0, "start_1000": 500, "start_2000": 500},
    {"type": "LADS", "file": "TIC 246902545lc.dat", "name": "TIC 246902545", "fmin": 5.0, "fmax": 80.0, "start_1000": 500, "start_2000": 500},
]

def ls_top_frequencies(t, y, M=30, fmin=0.5, fmax=100.0, max_samples=60000):
    mask = np.isfinite(t) & np.isfinite(y)
    t_clean = t[mask]
    y_clean = y[mask] - np.mean(y[mask])
    if len(t_clean) > max_samples:
        step = int(np.ceil(len(t_clean) / max_samples))
        t_clean = t_clean[::step]
        y_clean = y_clean[::step]
    baseline = t_clean.max() - t_clean.min()
    ls = LombScargle(t_clean, y_clean)
    freq, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=10)
    min_sep = 1.0 / max(baseline, 1e-9)
    sorted_idx = np.argsort(power)[::-1]
    chosen_freqs = []
    for idx in sorted_idx:
        f = float(freq[idx])
        if all(abs(f - fc) > min_sep for fc in chosen_freqs):
            chosen_freqs.append(f)
        if len(chosen_freqs) >= M:
            break
    return np.array(chosen_freqs, dtype=np.float64)

def build_rtvf_design(t, freqs, degree=1, trend_deg=1):
    t = np.asarray(t, float)
    t0, t1 = float(t.min()), float(t.max())
    tau = (2.0 * (t - t0) / max(t1 - t0, 1e-12)) - 1.0
    cols = []
    if trend_deg >= 0:
        for j in range(trend_deg + 1):
            cols.append(tau**j)
    for f in freqs:
        w = 2.0 * np.pi * f
        s = np.sin(w * t); c = np.cos(w * t)
        for j in range(degree + 1):
            pj = tau**j
            cols.append(pj * s); cols.append(pj * c)
    return np.column_stack(cols).astype(np.float32)

def rtvf_impute(t, y_missing, start, end, K=30, degree=1, l2=0.1, fmin=0.5, fmax=100.0):
    y = np.asarray(y_missing, float).copy()
    mask = np.isfinite(y)
    freqs = ls_top_frequencies(t[mask], y[mask], M=K, fmin=fmin, fmax=fmax)
    X = build_rtvf_design(t, freqs, degree=degree, trend_deg=1)
    X_obs = X[mask, :].astype(np.float32)
    y_obs = y[mask].astype(np.float32)
    XT_X = X_obs.T @ X_obs
    lamI = l2 * np.eye(XT_X.shape[0], dtype=np.float32)
    Cov_inv = XT_X + lamI
    try:
        beta = np.linalg.solve(Cov_inv, X_obs.T @ y_obs)
        Cov_beta = np.linalg.inv(Cov_inv)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X_obs, y_obs, rcond=None)[0]
        Cov_beta = np.linalg.pinv(Cov_inv)
    y_hat = (X @ beta).astype(np.float64)
    residuals = y_obs - y_hat[mask]
    sigma2_noise = np.var(residuals) if len(residuals) > 0 else 1.0
    var_pred = sigma2_noise + np.sum((X @ Cov_beta) * X, axis=1)
    y_std = np.sqrt(np.clip(var_pred, 0, None))
    y_rec = y.copy()
    y_rec[start:end] = y_hat[start:end]
    return y_rec, y_hat, y_std, freqs

def compute_ls_spectrum(t, y, fmin=0.1, fmax=100.0, samples_per_peak=5):
    mask = np.isfinite(t) & np.isfinite(y)
    t_clean = t[mask]
    y_clean = y[mask] - np.mean(y[mask])
    ls = LombScargle(t_clean, y_clean)
    freq, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)
    return freq, power

# Colors
COLOR_GROUND_TRUTH = '#000000'  # Solid Black
COLOR_UNIMPUTED    = '#f77f00'  # Orange
COLOR_RTVF         = '#e63946'  # Crimson Red
COLOR_PEAK_LINE    = '#2a9d8f'  # Teal Green
COLOR_GAP_SHADE    = '#ffeaa7'  # Soft Yellow

def run_full_range_validation(gap_size=1000):
    out_dir = PLOTS_DIR / f"full_range_0_100_gap_{gap_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_dir_3p = PLOTS_DIR / f"3panel_full_and_zoom_gap_{gap_size}"
    out_dir_3p.mkdir(parents=True, exist_ok=True)
    
    print("=" * 90)
    print(f"  RUNNING FULL RANGE (0 - 100 c/d) VALIDATION: N={gap_size} POINTS")
    print("=" * 90)
    
    for idx, star in enumerate(STARS_CONFIG):
        star_type = star["type"]
        star_name = star["name"]
        star_file = star["file"]
        fmin = star["fmin"]
        fmax = star["fmax"]
        start = star[f"start_{gap_size}"]
        end = start + gap_size
        
        file_path = DATA_DIR / star_type.lower() / star_file
        if not file_path.exists():
            continue
            
        data = np.loadtxt(file_path)
        t, y_orig = data[:, 0], data[:, 1]
        N = len(t)
        gap_duration_hours = (t[end] - t[start]) * 24.0
        
        # Unimputed (Gapped) Series
        y_gapped = y_orig.copy()
        y_gapped[start:end] = np.nan
        
        # R-TVF Imputed Series
        y_rtvf, y_hat, y_std, used_freqs = rtvf_impute(
            t, y_gapped, start, end, K=30, degree=1, l2=0.1, fmin=fmin, fmax=fmax
        )
        
        ss_res = np.sum((y_orig[start:end] - y_rtvf[start:end])**2)
        ss_tot = np.sum((y_orig[start:end] - np.mean(y_orig[start:end]))**2)
        r2_time = 1.0 - (ss_res / max(ss_tot, 1e-12))
        
        # 1. Full Spectrum 0 to 100 c/d
        freq_100, p_orig_100 = compute_ls_spectrum(t, y_orig, fmin=0.1, fmax=100.0, samples_per_peak=5)
        _, p_gapped_100 = compute_ls_spectrum(t, y_gapped, fmin=0.1, fmax=100.0, samples_per_peak=5)
        _, p_rtvf_100 = compute_ls_spectrum(t, y_rtvf, fmin=0.1, fmax=100.0, samples_per_peak=5)
        
        # 2. Focused Zoom around Dominant Mode
        f_dom = float(freq_100[np.argmax(p_orig_100)])
        zoom_span = 0.5
        f_zoom_min = max(0.1, f_dom - zoom_span)
        f_zoom_max = f_dom + zoom_span
        freq_zoom, p_orig_zoom = compute_ls_spectrum(t, y_orig, f_zoom_min, f_zoom_max, samples_per_peak=35)
        _, p_gapped_zoom = compute_ls_spectrum(t, y_gapped, f_zoom_min, f_zoom_max, samples_per_peak=35)
        _, p_rtvf_zoom = compute_ls_spectrum(t, y_rtvf, f_zoom_min, f_zoom_max, samples_per_peak=35)
        
        safe_name = star_name.replace(" ", "_")
        
        # ---------------------------------------------------------------------
        # FORMAT A: 2-Panel Full Range (0 - 100 c/d)
        # ---------------------------------------------------------------------
        fig2, (ax_t, ax_f100) = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={'height_ratios': [1, 1.25]})
        
        pad = int(gap_size * 0.6)
        t_left = max(0, start - pad)
        t_right = min(N, end + pad)
        ax_t.plot(t[t_left:t_right], y_orig[t_left:t_right], color='#6c757d', alpha=0.45, lw=1.0, label='Ground Truth')
        ax_t.axvspan(t[start], t[end-1], color=COLOR_GAP_SHADE, alpha=0.5, label='Gap')
        ax_t.plot(t[start:end], y_rtvf[start:end], color=COLOR_RTVF, lw=1.6, label='R-TVF')
        ax_t.fill_between(
            t[start:end],
            y_rtvf[start:end] - 3.0 * y_std[start:end],
            y_rtvf[start:end] + 3.0 * y_std[start:end],
            color=COLOR_RTVF, alpha=0.22, label='3σ Band'
        )
        ax_t.set_xlabel('Time (BJD)')
        ax_t.set_ylabel('Flux ($e^-$/s)')
        ax_t.set_title(f"{star_name} ({star_type}) — Time-Domain Reconstruction ($N={gap_size}$, $R^2 = {r2_time:.3f}$)", fontweight='bold')
        ax_t.legend(loc='upper right', frameon=True, fontsize=10)
        ax_t.grid(True, linestyle='--', alpha=0.4)
        
        ax_f100.plot(freq_100, p_orig_100, color=COLOR_GROUND_TRUTH, lw=1.8, label='Ground Truth', alpha=0.95, zorder=3)
        ax_f100.plot(freq_100, p_gapped_100, color=COLOR_UNIMPUTED, lw=1.2, label='Unimputed', alpha=0.85, zorder=2)
        ax_f100.plot(freq_100, p_rtvf_100, color=COLOR_RTVF, lw=1.2, label='R-TVF', alpha=0.9, zorder=4)
        ax_f100.set_xlim(0.0, 100.0)
        ax_f100.set_xlabel('Frequency (cycles/day)')
        ax_f100.set_ylabel('Power')
        ax_f100.set_title(f"Full Asteroseismic Frequency Spectrum ($0 - 100$ cycles/day)", fontsize=11.5, fontweight='bold')
        ax_f100.legend(loc='upper right', frameon=True, fontsize=10)
        ax_f100.grid(True, linestyle='--', alpha=0.4)
        
        fig2.tight_layout()
        fig2.savefig(out_dir / f"comparative_{star_type}_{safe_name}_full_0_100_gap_{gap_size}.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # ---------------------------------------------------------------------
        # FORMAT B: 3-Panel (Time Domain + Full 0-100 c/d + Focused Mode Zoom)
        # ---------------------------------------------------------------------
        fig3, (ax3_t, ax3_f100, ax3_fzoom) = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw={'height_ratios': [1, 1, 1.1]})
        
        # Panel 1: Time
        ax3_t.plot(t[t_left:t_right], y_orig[t_left:t_right], color='#6c757d', alpha=0.45, lw=1.0, label='Ground Truth')
        ax3_t.axvspan(t[start], t[end-1], color=COLOR_GAP_SHADE, alpha=0.5, label='Gap')
        ax3_t.plot(t[start:end], y_rtvf[start:end], color=COLOR_RTVF, lw=1.6, label='R-TVF')
        ax3_t.fill_between(
            t[start:end],
            y_rtvf[start:end] - 3.0 * y_std[start:end],
            y_rtvf[start:end] + 3.0 * y_std[start:end],
            color=COLOR_RTVF, alpha=0.22, label='3σ Band'
        )
        ax3_t.set_xlabel('Time (BJD)')
        ax3_t.set_ylabel('Flux ($e^-$/s)')
        ax3_t.set_title(f"{star_name} ({star_type}) — Time-Domain Reconstruction ($N={gap_size}$, $R^2 = {r2_time:.3f}$)", fontweight='bold')
        ax3_t.legend(loc='upper right', frameon=True, fontsize=9.5)
        ax3_t.grid(True, linestyle='--', alpha=0.4)
        
        # Panel 2: Full Spectrum 0 - 100 c/d
        ax3_f100.plot(freq_100, p_orig_100, color=COLOR_GROUND_TRUTH, lw=1.8, label='Ground Truth', alpha=0.95, zorder=3)
        ax3_f100.plot(freq_100, p_gapped_100, color=COLOR_UNIMPUTED, lw=1.2, label='Unimputed', alpha=0.85, zorder=2)
        ax3_f100.plot(freq_100, p_rtvf_100, color=COLOR_RTVF, lw=1.2, label='R-TVF', alpha=0.9, zorder=4)
        ax3_f100.set_xlim(0.0, 100.0)
        ax3_f100.set_xlabel('Frequency (cycles/day)')
        ax3_f100.set_ylabel('Power')
        ax3_f100.set_title(f"Full Spectrum ($0 - 100$ cycles/day)", fontsize=11, fontweight='bold')
        ax3_f100.legend(loc='upper right', frameon=True, fontsize=9.5)
        ax3_f100.grid(True, linestyle='--', alpha=0.4)
        
        # Panel 3: Focused Zoom around f_1
        ax3_fzoom.plot(freq_zoom, p_orig_zoom, color=COLOR_GROUND_TRUTH, lw=2.4, label='Ground Truth', alpha=0.95, zorder=3)
        ax3_fzoom.plot(freq_zoom, p_gapped_zoom, color=COLOR_UNIMPUTED, lw=1.8, label='Unimputed', alpha=0.9, zorder=2)
        ax3_fzoom.plot(freq_zoom, p_rtvf_zoom, color=COLOR_RTVF, lw=1.8, label='R-TVF', alpha=0.95, zorder=4)
        ax3_fzoom.axvline(f_dom, color=COLOR_PEAK_LINE, ls='--', alpha=0.6, lw=1.2)
        ax3_fzoom.text(f_dom + 0.015, max(p_orig_zoom)*0.88, f"$f_1 = {f_dom:.4f}$ c/d", color=COLOR_PEAK_LINE, fontsize=10, fontweight='bold')
        ax3_fzoom.set_xlim(f_zoom_min, f_zoom_max)
        ax3_fzoom.set_xlabel('Frequency (cycles/day)')
        ax3_fzoom.set_ylabel('Power')
        ax3_fzoom.set_title(f"Dominant Mode Zoom ($f_1 = {f_dom:.3f}$ c/d, $\\pm {zoom_span}$ c/d Zoom)", fontsize=11, fontweight='bold')
        ax3_fzoom.legend(loc='upper right', frameon=True, fontsize=9.5)
        ax3_fzoom.grid(True, linestyle='--', alpha=0.4)
        
        fig3.tight_layout()
        fig3.savefig(out_dir_3p / f"comparative_{star_type}_{safe_name}_3panel_gap_{gap_size}.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        print(f"[{idx+1}/10] {star_type} {star_name:14s} | Full 0-100 c/d & 3-Panel plots generated.")

def main():
    run_full_range_validation(gap_size=1000)
    run_full_range_validation(gap_size=2000)
    print("\n" + "=" * 90)
    print("  ALL FULL-RANGE (0 - 100 c/d) PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 90)

if __name__ == "__main__":
    main()
