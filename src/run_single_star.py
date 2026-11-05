#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-TVF Single Star Imputation Utility
===================================
Applies the Regularized Time-Varying Fourier (R-TVF) method to an individual
photometric time series with missing observations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import modelos_hads_lads as mh
import utils


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute R-TVF imputation on an individual stellar light curve."
    )
    parser.add_argument(
        "--star",
        type=str,
        required=True,
        help="Path to the input photometric file (.dat).",
    )
    parser.add_argument(
        "--gap-start",
        type=int,
        default=None,
        help="Start index for the simulated contiguous gap. If omitted, selected randomly.",
    )
    parser.add_argument(
        "--gap-size",
        type=int,
        default=1000,
        help="Number of contiguous missing observations (default: 1000).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="Number of dominant Fourier frequencies (default: 10).",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=1,
        help="Degree of the polynomial modulation d (default: 1).",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.1,
        help="Tikhonov regularization parameter lambda (default: 0.1).",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=5.0,
        help="Lower frequency boundary in cycles/day (default: 5.0).",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=80.0,
        help="Upper frequency boundary in cycles/day (default: 80.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Pseudorandom generator seed for gap placement (default: 42).",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="single_star_reconstruction.png",
        help="Path to save the output reconstruction figure.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to export the imputed time series to CSV.",
    )
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, start: int, end: int) -> dict[str, float]:
    yt = y_true[start:end]
    yp = y_pred[start:end]
    mae = float(np.mean(np.abs(yt - yp)))
    mse = float(np.mean((yt - yp) ** 2))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(np.abs(yt), 1e-12, None))) * 100)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"MAE": mae, "MSE": mse, "MAPE": mape, "R2": r2}


def main() -> None:
    args = parse_arguments()
    star_path = Path(args.star).resolve()
    if not star_path.exists():
        sys.exit(f"Error: Specified file does not exist: {star_path}")

    # 1. Load photometric data
    t, y_true = utils.load_data(str(star_path))
    n_points = len(t)
    if n_points < args.gap_size:
        sys.exit(f"Error: Time series contains {n_points} points, smaller than gap size {args.gap_size}.")

    # 2. Simulate contiguous missing block
    y_missing, start, end = utils.simulate_block_gap(
        y_true,
        start_index=args.gap_start,
        gap_size=args.gap_size,
        seed=args.seed,
    )

    # 3. Fit R-TVF and reconstruct missing segment
    y_imputed, freqs, y_std = mh.tvar_fourier_impute_block_multi(
        t,
        y_missing,
        start=start,
        end=end,
        M=args.K,
        degree=args.degree,
        trend_deg=1,
        l2=args.l2,
        fmin=args.fmin,
        fmax=args.fmax,
        return_std=True,
    )

    # 4. Quantitative evaluation
    metrics = compute_metrics(y_true, y_imputed, start, end)

    print("=================================================================")
    print(" R-TVF Single Star Imputation Summary")
    print("=================================================================")
    print(f" Target File:         {star_path.name}")
    print(f" Total Observations:  {n_points}")
    print(f" Imputed Gap Range:   [{start}, {end}) (N = {end - start} points)")
    print(f" Time Span:           [{t[start]:.4f}, {t[end-1]:.4f}] days")
    print(f" Parameters:          K={args.K}, d={args.degree}, lambda={args.l2}")
    print(f" Extracted Frequencies ({len(freqs)}):")
    for i, f in enumerate(freqs, 1):
        print(f"   f_{i} = {f:.5f} cycles/day")
    print("-----------------------------------------------------------------")
    print(f" Evaluation Metrics:")
    print(f"   MAE:               {metrics['MAE']:.4f}")
    print(f"   MSE:               {metrics['MSE']:.4f}")
    print(f"   MAPE (%):          {metrics['MAPE']:.2f}%")
    print(f"   R2 Score:          {metrics['R2']:.4f}")
    print("=================================================================")

    # 5. Export reconstructed data if requested
    if args.output_csv:
        out_csv_path = Path(args.output_csv).resolve()
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame({
            "time": t,
            "flux_original": y_true,
            "flux_imputed": y_imputed,
            "pred_std": y_std,
            "is_imputed": [1 if start <= i < end else 0 for i in range(n_points)],
        })
        df_out.to_csv(out_csv_path, index=False)
        print(f" Imputed time series exported to: {out_csv_path}")

    # 6. Visualization
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Overview panel
    ax_full.plot(t, y_true, color="#7f8c8d", alpha=0.5, linewidth=0.7, label="Observed Flux")
    ax_full.axvspan(t[start], t[end - 1], color="#e74c3c", alpha=0.15, label=f"Missing Gap (N={end - start})")
    ax_full.plot(t[start:end], y_imputed[start:end], color="#2980b9", linewidth=1.2, label="R-TVF Reconstruction")
    ax_full.set_ylabel("Flux (e-/s)")
    ax_full.set_title(f"Full Light Curve: {star_path.name}")
    ax_full.legend(loc="upper right", frameon=True)
    ax_full.grid(True, linestyle=":", alpha=0.6)

    # Detailed zoom panel
    padding = max(50, int((end - start) * 0.25))
    i_left = max(0, start - padding)
    i_right = min(n_points, end + padding)

    ax_zoom.scatter(t[i_left:start], y_true[i_left:start], s=6, color="#2c3e50", alpha=0.6, label="Pre-gap Observed")
    ax_zoom.scatter(t[end:i_right], y_true[end:i_right], s=6, color="#2c3e50", alpha=0.6, label="Post-gap Observed")
    ax_zoom.plot(t[start:end], y_true[start:end], color="#7f8c8d", linestyle="--", linewidth=1.0, label="Ground Truth (Unobserved)")
    ax_zoom.plot(t[start:end], y_imputed[start:end], color="#2980b9", linewidth=1.5, label=f"R-TVF Reconstruction (R2={metrics['R2']:.3f})")

    # Analytical confidence band (95%)
    ci_lower = y_imputed[start:end] - 1.96 * y_std[start:end]
    ci_upper = y_imputed[start:end] + 1.96 * y_std[start:end]
    ax_zoom.fill_between(t[start:end], ci_lower, ci_upper, color="#2980b9", alpha=0.2, label=r"Analytical 95% CI")

    ax_zoom.axvspan(t[start], t[end - 1], color="#e74c3c", alpha=0.08)
    ax_zoom.set_xlabel("Time (BJD - 2457000, days)")
    ax_zoom.set_ylabel("Flux (e-/s)")
    ax_zoom.set_title(f"Reconstruction Detail: Gap [{t[start]:.3f}, {t[end-1]:.3f}] days")
    ax_zoom.legend(loc="upper right", frameon=True)
    ax_zoom.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_plot_path = Path(args.output_plot).resolve()
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot_path, dpi=300)
    plt.close()
    print(f" Reconstruction figure exported to: {out_plot_path}")
    print("=================================================================")


if __name__ == "__main__":
    main()
