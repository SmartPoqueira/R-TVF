#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════════════════
  Interactive HTML Periodogram & Time-Domain Visualizer for Antonio
  With Pan, Zoom, Hover tooltips, and Active Band Focus
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.timeseries import LombScargle
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR.parent / "pasar a Antonio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR.parent.parent / "major review (do not touch)" / "estrellas_test"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR.parent.parent / "first_submission (do not touch)" / "estrellas_test"

STARS = [
    {"type": "HADS", "file": "AD Cmilc.dat", "name": "AD CMi", "fmin": 3.0, "fmax": 30.0, "start": 500, "span": 1000},
    {"type": "HADS", "file": "HD 146953lc.dat", "name": "HD 146953", "fmin": 3.0, "fmax": 30.0, "start": 500, "span": 1000},
    {"type": "LADS", "file": "TIC 100531058lc.dat", "name": "TIC 100531058", "fmin": 5.0, "fmax": 80.0, "start": 500, "span": 1000},
    {"type": "LADS", "file": "TIC 246902545lc.dat", "name": "TIC 246902545", "fmin": 5.0, "fmax": 80.0, "start": 500, "span": 1000},
    {"type": "LADS", "file": "TIC 452982723lc.dat", "name": "TIC 452982723", "fmin": 5.0, "fmax": 80.0, "start": 500, "span": 1000},
]

sys.path.insert(0, str(BASE_DIR))
import validate_frequency_comparison as vfc

def create_interactive_plot(star):
    sname = star["name"]
    stype = star["type"]
    sfile = star["file"]
    start = star["start"]
    gap_size = star["span"]
    end = start + gap_size
    
    file_path = DATA_DIR / stype.lower() / sfile
    data = np.loadtxt(file_path)
    t, y_orig = data[:, 0], data[:, 1]
    N = len(t)
    
    y_gapped = y_orig.copy()
    y_gapped[start:end] = np.nan
    
    y_rtvf, y_hat, y_std, freqs = vfc.rtvf_impute(
        t, y_gapped, start, end, K=30, degree=1, l2=0.1, fmin=star["fmin"], fmax=star["fmax"]
    )
    
    # Calculate periodogram 0 to 100 c/d
    freq_100, p_orig_100 = vfc.compute_ls_spectrum(t, y_orig, fmin=0.1, fmax=100.0, samples_per_peak=10)
    _, p_gapped_100 = vfc.compute_ls_spectrum(t, y_gapped, fmin=0.1, fmax=100.0, samples_per_peak=10)
    _, p_rtvf_100 = vfc.compute_ls_spectrum(t, y_rtvf, fmin=0.1, fmax=100.0, samples_per_peak=10)
    
    # Create Subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.4, 0.6],
        vertical_spacing=0.12,
        subplot_titles=(
            f"<b>{sname} ({stype}) — Time Domain Light Curve (Zoom around Gap)</b>",
            f"<b>{sname} ({stype}) — Frequency Spectrum (0 to 100 cycles/day) [Interactive Zoom]</b>"
        )
    )
    
    # Panel 1: Time Domain
    pad = 600
    t_l = max(0, start - pad)
    t_r = min(N, end + pad)
    
    fig.add_trace(go.Scatter(
        x=t[t_l:t_r], y=y_orig[t_l:t_r],
        mode='lines', line=dict(color='#888888', width=1),
        name='Ground Truth (Observed)', legendgroup='time'
    ), row=1, col=1)
    
    # 3-sigma band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t[start:end], t[start:end][::-1]]),
        y=np.concatenate([y_rtvf[start:end] + 3*y_std[start:end], (y_rtvf[start:end] - 3*y_std[start:end])[::-1]]),
        fill='toself', fillcolor='rgba(230, 57, 70, 0.22)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=True, name='3σ Confidence Band', legendgroup='time'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=t[start:end], y=y_rtvf[start:end],
        mode='lines', line=dict(color='#e63946', width=2),
        name='R-TVF Imputation', legendgroup='time'
    ), row=1, col=1)
    
    # Highlight Gap region
    fig.add_vrect(
        x0=t[start], x1=t[end-1],
        fillcolor="#ffeaa7", opacity=0.35, layer="below", line_width=0,
        annotation_text="Gapped Region", annotation_position="top left",
        row=1, col=1
    )
    
    # Panel 2: Frequency Domain (0 - 100 c/d)
    fig.add_trace(go.Scatter(
        x=freq_100, y=p_orig_100,
        mode='lines', line=dict(color='#000000', width=1.6),
        name='Ground Truth Spectrum', legendgroup='freq'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=freq_100, y=p_gapped_100,
        mode='lines', line=dict(color='#f77f00', width=1.2),
        name='Unimputed (Gapped) Spectrum', legendgroup='freq'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=freq_100, y=p_rtvf_100,
        mode='lines', line=dict(color='#e63946', width=1.3),
        name='R-TVF Imputed Spectrum', legendgroup='freq'
    ), row=2, col=1)
    
    fig.update_xaxes(title_text="Time (BJD)", row=1, col=1)
    fig.update_yaxes(title_text="Flux (e-/s)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (cycles/day) — <i>Drag mouse box to zoom into any peak</i>", row=2, col=1)
    fig.update_yaxes(title_text="Lomb-Scargle Power", row=2, col=1)
    
    fig.update_layout(
        height=850, width=1200,
        template='plotly_white',
        title=dict(
            text=f"Interactive Asteroseismic Frequency Validation: {sname} ({stype})",
            font=dict(size=17, color='#1d3557')
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    safe_name = sname.replace(" ", "_")
    html_path = OUT_DIR / f"INTERACTIVO_{safe_name}_{stype}.html"
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    print(f"Generated interactive HTML: {html_path.name}")

def main():
    for star in STARS:
        create_interactive_plot(star)
    print("Done!")

if __name__ == "__main__":
    main()
