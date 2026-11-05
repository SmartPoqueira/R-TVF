#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparador 3x2 (auto-selección por estrategia de huecos)

- Si input.json -> missing_strategy == "random_points":
    Ejecuta: Spline, ARMA, KNN, RNN, GRU, LSTM

- Si missing_strategy == "block_runs":
    Ejecuta: ARMA, ARMAX(arma+fourier), MIARMA, BiLSTM, Fourier-TV (multi), SAITS

Parámetros (ventanas, p/q, K, epochs, etc.) se leen de input.json (mismas claves
que ya usas en tus scripts). Los algoritmos de bloque se apoyan en
`modelos_hads_lads.py` si están implementados ahí.

Salida:
  - Figura 3x2: "comparativa_3x2.png"
  - JSON de métricas + tiempos: "metrics_summary.json"

Uso:
    python runner_comparativa_3x2.py input.json
"""
from __future__ import annotations

import json, time, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Warnings ruidosos (no afectan a cálculos)
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message="divide by zero encountered", module="numpy.linalg")
warnings.filterwarnings("ignore", message="overflow encountered", module="numpy.linalg")
warnings.filterwarnings("ignore", message="invalid value encountered", module="numpy.linalg")

# GP imports
try:
    import celerite2
    from celerite2 import terms
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    print("Warning: celerite2 not available. GP-Celerite will be skipped.")

# --- rutas dinámicas para importar los módulos del usuario ---
import sys
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- imports "suaves": se intenta cargar y si no, se degrada con puentes lineales ---
def _try_import_lightcurve():
    try:
        import lightCurve as lc    # funciones random_points y SAITS/ARMA para máscara
        return lc
    except Exception as e:
        print("[WARN] No se pudo importar lightCurve.py ->", e)
        return None

def _try_import_mh():
    try:
        import modelos_hads_lads as mh  # funciones de bloque (ARMAX, MIARMA, BiLSTM, Fourier-TV)
        return mh
    except Exception as e:
        print("[WARN] No se pudo importar modelos_hads_lads.py ->", e)
        return None

# =============== Utilidades básicas ===============
def load_dat(path):
    """Lee .dat (tiempo, flujo) -> arrays float."""
    df = pd.read_csv(path, sep=r"\s+", names=["time", "flux"], comment="#", header=None)
    df = df.dropna().reset_index(drop=True)
    t = df["time"].astype(float).to_numpy()
    y = df["flux"].astype(float).to_numpy()
    return t, y

def simulate_random_points(y, percentage, seed=12):
    """Elimina aleatoriamente % de puntos -> mask_pred (True = imputar)."""
    rng = np.random.default_rng(seed)
    N = len(y); k = max(1, int(round(N * (float(percentage)/100.0))))
    mask = np.zeros(N, dtype=bool)
    idxs = rng.choice(N, size=k, replace=False)
    mask[idxs] = True
    y_m = y.astype(float).copy()
    y_m[mask] = np.nan
    return y_m, mask

def simulate_block_run(y, percentage=None, block_n="full", seed=123):
    """Elimina un bloque contiguo."""
    rng = np.random.default_rng(seed)
    N = len(y)
    if block_n == "full":
        L = max(1, int(round(N * (float(percentage)/100.0))))
    else:
        L = max(1, int(block_n))
    start = int(rng.integers(0, max(1, N - L)))
    end = min(N, start + L)
    y_m = y.astype(float).copy()
    y_m[start:end] = np.nan
    return y_m, start, end

def metrics_from_mask(y_true, y_imp, mask):
    yt = y_true[mask]
    yp = y_imp[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    mse = float(np.mean((yt - yp) ** 2))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(np.abs(yt), 1e-12, None))) * 100)
    
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    
    return dict(MAE=mae, MSE=mse, MAPE=mape, R2=r2)

def metrics_from_block(y_true, y_imp, start, end):
    mask = np.zeros_like(y_true, dtype=bool); mask[start:end] = True
    return metrics_from_mask(y_true, y_imp, mask)

# =============== Plots ===============
def plot_grid_3x2_block(t, y_true, y_missing, methods, start, end, pad=200, title="Comparativa (bloque)"):
    """
    3x2 con sombreado del hueco.
    - Pintamos la serie con NaN **sin filtrar** para que Matplotlib rompa la línea y no trace la diagonal roja.
    - Si hay menos de 6 métodos, rellenamos con 'Sin resultado'.
    """
    left = max(0, start - pad); right = min(len(t), end + pad)
    tx = t[left:right]
    y_true_win = y_true[left:right]
    y_miss_win = y_missing[left:right]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(tx, y_true_win, lw=1.0, color="black", ls="--", label="Original")
        ax.plot(tx, y_miss_win, lw=1.0, color="red", ls="--", label="Con NaN (bloque)")

        if i < len(methods):
            label, y_imp, color = methods[i]
            # y_imp puede venir como list o ndarray; normalizamos
            y_imp = np.asarray(y_imp, dtype=float)
            ax.plot(tx, y_imp[left:right], lw=1.6, color=color, label=label)
        else:
            ax.text(0.5, 0.5, "Sin resultado", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
            ax.set_facecolor("#f7f7f7")

        ax.axvspan(t[start], t[end-1], color="lightgray", alpha=0.5, label=f"Hueco N={end-start}")
        ax.grid(True); ax.set_ylabel("Flux"); ax.legend(loc="upper left", fontsize=9)
    for ax in axes[4:]:
        ax.set_xlabel("HJD")
    fig.suptitle(title, y=0.995, fontsize=15)
    fig.tight_layout(rect=[0,0,1,0.98])
    return fig

def plot_grid_3x2_random(
    t, y_true, y_missing, methods, mask_pred,
    title="Comparativa (puntos aleatorios)", zoom=True, pad_samples=600
):
    """
    Estilo similar a block_runs:
      - Línea negra discontinua: Original
      - Línea roja discontinua: Con NaN (random)  [la línea se corta donde hay NaN]
      - Línea de color: método imputado
      - Puntos:  azul = imputado en NaN,  rojo = ground-truth (valor real) en NaN
    """
    import numpy as np
    import matplotlib.pyplot as plt

    N = len(t)
    idx_nan = np.where(mask_pred)[0]

    # --- Ventana de zoom alrededor de los NaN ---
    if zoom and idx_nan.size > 0:
        span = idx_nan.max() - idx_nan.min()
        if span > 0.6 * N:
            i0 = int(np.median(idx_nan))
            left_i = max(0, i0 - int(pad_samples))
            right_i = min(N, i0 + int(pad_samples))
        else:
            left_i = max(0, int(idx_nan.min()) - int(pad_samples))
            right_i = min(N, int(idx_nan.max()) + int(pad_samples))
        # cota para que la ventana no sea enorme
        max_window = max(200, int(0.4 * N))
        if (right_i - left_i) > max_window:
            i0 = (left_i + right_i) // 2
            half = max_window // 2
            left_i = max(0, i0 - half)
            right_i = min(N, i0 + half)
    else:
        left_i, right_i = 0, N

    # --- Recortes ---
    tx = t[left_i:right_i]
    y_true_win = y_true[left_i:right_i]
    y_miss_win = y_missing[left_i:right_i]
    nan_local = mask_pred[left_i:right_i]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # 1) Original (línea discontinua negra)
        ax.plot(tx, y_true_win, lw=1.0, ls="--", color="black", label="Original")

        # 2) Con NaN (random) como línea roja discontinua SIN filtrar: la línea se corta en NaN
        ax.plot(tx, y_miss_win, lw=1.0, ls="--", color="red", label="Con NaN (random)")

        # 3) Método imputado (línea de color)
        if i < len(methods):
            label, y_imp, color = methods[i]
            y_imp = np.asarray(y_imp, float)
            y_imp_win = y_imp[left_i:right_i]
            # Dibuja la línea imputada, rompiéndola si hay saltos grandes (huecos reales)
            max_gap = np.median(np.diff(tx)) * 5  # umbral: 5x la separación típica
            segments = np.split(np.arange(len(tx)), np.where(np.diff(tx) > max_gap)[0] + 1)
            for seg in segments:
                ax.plot(tx[seg], y_imp_win[seg], lw=1.6, color=color, label=label if seg is segments[0] else "")

            # 4) Puntos en las posiciones perdidas
            if nan_local.any():
                # Imputado (azul) en índices perdidos
                ax.plot(tx[nan_local], y_imp_win[nan_local], "o", ms=3.5, color="blue",
                        label="Imputado (punto)", zorder=5)
                # Ground-truth real (rojo) en los mismos índices
                ax.plot(tx[nan_local], y_true_win[nan_local], "o", ms=3.5, color="red",
                        label="GT (punto)", zorder=6)
        else:
            ax.text(0.5, 0.5, "Sin resultado", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
            ax.set_facecolor("#f7f7f7")

        ax.grid(True)
        ax.set_ylabel("Flux")
        ax.legend(loc="upper left", fontsize=9)

    for ax in axes[4:]:
        ax.set_xlabel("HJD")

    fig.suptitle(title, y=0.995, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig
# =============== Algoritmos ===============
def run_random_points(t, y_true, y_missing, mask_pred, hp, epochs):
    """
    Ejecuta: Spline, ARMA, KNN, RNN, GRU, LSTM (según disponibilidad de deps).
    Devuelve:
      - lista methods_plot: [(label, y_imp, color), ...] máx 6
      - metrics dict por método
    """
    lc = _try_import_lightcurve()
    if lc is None:
        raise RuntimeError("Se requiere lightCurve.py para random_points.")
    df = pd.DataFrame({"tiempo": t, "flujo": y_missing})
    metrics = {}
    results = []

    palette = ["blue", "green", "orange", "purple", "brown", "magenta"]
    idx = 0

    # --- 1) Spline ---
    t0 = time.perf_counter()
    df_spline = lc.rellenar_con_spline(df, int(hp.get("spline_order", 3)), mask_pred)
    y_spline = df_spline["flujo_relleno"].to_numpy(dtype=float)
    elapsed = time.perf_counter() - t0
    metrics["Spline"] = metrics_from_mask(y_true, y_spline, mask_pred) | {"elapsed_seconds": round(elapsed, 3)}
    results.append(("Spline", y_spline, palette[idx])); idx += 1

    # --- 2) ARMA ---
        # --- 2) ARMA (estabilizado) ---
    try:
        t0 = time.perf_counter()
        y_arma = _arma_impute_stable(
            pd.DataFrame({"tiempo": t, "flujo": y_missing}),
            mask_pred,
            int(hp.get("arma_p", 3)),
            int(hp.get("arma_q", 5)),
            lc
        )
        elapsed = time.perf_counter() - t0
        metrics["ARMA"] = metrics_from_mask(y_true, y_arma, mask_pred) | {"elapsed_seconds": round(elapsed, 3)}
        results.append(("ARMA", y_arma, palette[idx])); idx += 1
    except Exception as e:
        print("[WARN] ARMA (random_points) falló ->", e)

    # --- 3) KNN ---
    try:
        t0 = time.perf_counter()
        df_knn = lc.rellenar_con_knn(df, int(hp.get("k", 25)), mask_pred)
        y_knn = df_knn["flujo_relleno"].to_numpy(dtype=float)
        elapsed = time.perf_counter() - t0
        metrics["KNN"] = metrics_from_mask(y_true, y_knn, mask_pred) | {"elapsed_seconds": round(elapsed, 3)}
        results.append(("KNN", y_knn, palette[idx])); idx += 1
    except Exception as e:
        print("[WARN] KNN falló ->", e)

    # --- 4) RNN/GRU/LSTM ---
    def _run_rnn_tipo(tipo, label):
        nonlocal idx
        try:
            t0 = time.perf_counter()
            ret = lc.rellenar_con_rnn(
                df,
                tipo=tipo,
                ventana=int(hp.get("window_size", 100)),
                epocas=int(epochs),
                unidades=int(hp.get("units", 64)),
                mask_pred=mask_pred,
                # entrenamos con el mismo df pero excluyendo los targets simulados
                df_train=df,
                exclude_targets=mask_pred
            )
            df_rnn = ret[0] if isinstance(ret, tuple) else ret
            y_pred = df_rnn["flujo_relleno"].to_numpy(dtype=float)
            elapsed = time.perf_counter() - t0
            metrics[label] = metrics_from_mask(y_true, y_pred, mask_pred) | {"elapsed_seconds": round(elapsed, 3)}
            results.append((label, y_pred, palette[idx])); idx += 1
        except Exception as e:
            print(f"[WARN] {label} falló ->", e)

    _run_rnn_tipo("rnn", "RNN")
    _run_rnn_tipo("gru", "GRU")
    _run_rnn_tipo("lstm", "LSTM")

    # Trunca/ajusta a 6 métodos si sobran/faltan
    results = results[:6]
    return results, metrics


def _arma_impute_stable(df_in: pd.DataFrame, mask_bool: np.ndarray, p: int, q: int, lc) -> np.ndarray:
    """
    Wrapper para ARMA: z-score en observados -> imputar -> desescalar.
    df_in: DataFrame con columnas ['tiempo','flujo'] (como usa lightCurve).
    mask_bool: True en posiciones a imputar.
    """
    y = df_in["flujo"].to_numpy(dtype=float)
    obs = ~mask_bool
    mu = np.nanmean(y[obs])
    sigma = np.nanstd(y[obs])
    if not np.isfinite(sigma) or sigma < 1e-12:
        sigma = 1.0  # evita división por 0 si la serie es casi constante

    df_scaled = df_in.copy()
    df_scaled["flujo"] = (y - mu) / sigma

    df_out = lc.rellenar_con_arma(df_scaled, int(p), int(q), mask_bool)
    y_scaled = df_out["flujo_relleno"].to_numpy(dtype=float)

    return (mu + sigma * y_scaled).astype(float)


def _safe_bilstm_call(mh, t, y_missing, start, end, **kw):
    """
    En algunos entornos, una implementación puede internamente convertir la serie a list y
    luego indexarla con arrays → TypeError. Aseguramos tipos, llamamos y normalizamos salida.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y_missing, dtype=float)
    out = mh.bilstm_impute_block_with_sinusoids_auto(t, y, int(start), int(end), **kw)
    y_imp = out[0] if isinstance(out, tuple) else out
    return np.asarray(y_imp, dtype=float)

def run_gp_imputation(t, y_missing, start, end, params):
    if not GP_AVAILABLE:
        return None
    mask_train = np.isfinite(y_missing)
    t_train = t[mask_train]
    y_train = y_missing[mask_train]
    sigma_est = np.std(y_train)
    try:
        import modelos_hads_lads as mh
        freqs = mh.ls_top_freqs(t_train, y_train, M=1)
        dominant_period = 1.0 / freqs[0]
    except Exception:
        dominant_period = 0.1
    kernel = terms.SHOTerm(
        sigma=sigma_est,
        rho=float(params.get("rho", dominant_period)),
        tau=float(params.get("tau", 10.0))
    )
    mean_val = np.median(y_train)
    gp = celerite2.GaussianProcess(kernel, mean=mean_val)
    try:
        gp.compute(t_train, yerr=float(params.get("noise", 0.001)))
        mu = gp.predict(y_train, t=t, return_var=False)
        return mu
    except Exception as e:
        print(f"[GP Error] {e}")
        return None

def run_block_runs(t, y_true, y_missing, start, end, hp, epochs):
    """
    Ejecuta en orden: ARMAX, GP-Celerite, DHR, FS-BSM, TBATS, R-TVF (ours).
    """
    mh = _try_import_mh()
    if mh is None:
        raise RuntimeError("Se requiere modelos_hads_lads.py para estrategia 'block_runs'.")

    metrics = {}
    results = []
    palette = ["blue", "green", "orange", "purple", "brown", "magenta"]
    idx = 0

    # Frequency band from grid-search hyperparams (set per star-type: HADS 3-30, LADS 5-80 c/d)
    fmin = float(hp.get("fmin", 3.0))
    fmax = float(hp.get("fmax", 80.0))

    # ------- 1) ARMAX (Fourier) -------
    try:
        K = int(hp.get("fourier_K", 3))
        freqs = mh.ls_top_freqs(t[np.isfinite(y_missing)], y_missing[np.isfinite(y_missing)], M=K, fmin=fmin, fmax=fmax)
        t0 = time.perf_counter()
        y_armax, _ = mh.armax_impute_block_multi(t, y_missing, start, end,
                                                 freqs_cpd=freqs, p=int(hp.get("arma_p",3)), q=int(hp.get("arma_q",5)), trend="c")
        elapsed = time.perf_counter() - t0
        y_armax = np.asarray(y_armax, float)
        metrics["ARMAX (Fourier)"] = metrics_from_block(y_true, y_armax, start, end) | {"elapsed_seconds": round(elapsed, 3)}
        results.append(("ARMAX (Fourier)", y_armax, palette[idx])); idx += 1
    except Exception as e:
        print("[WARN] ARMAX falló ->", e)

    # ------- 2) GP-Celerite -------
    try:
        t0 = time.perf_counter()
        gp_params = {
            "tau": float(hp.get("gp_tau", 50.0)),
            "noise": float(hp.get("gp_noise", 1e-3))
        }
        y_gp = run_gp_imputation(t, y_missing, start, end, gp_params)
        elapsed = time.perf_counter() - t0
        if y_gp is not None:
            y_gp = np.asarray(y_gp, float)
            metrics["GP-Celerite"] = metrics_from_block(y_true, y_gp, start, end) | {"elapsed_seconds": round(elapsed, 3)}
            results.append(("GP-Celerite", y_gp, palette[idx])); idx += 1
        else:
            print("[WARN] GP-Celerite no retornó resultado")
    except Exception as e:
        print("[WARN] GP-Celerite falló ->", e)

    # ------- 3) Dynamic Harmonic Regression (DHR) -------
    try:
        K     = int(hp.get("fourier_K", 3))
        p_ar  = int(hp.get("p_ar", 1))
        t0 = time.perf_counter()
        y_dhr = mh.dhr_impute_block(t, y_missing, start, end,
                                    fmin=fmin, fmax=fmax, M=K, p_ar=p_ar, maxiter=20)
        elapsed = time.perf_counter() - t0
        y_dhr = np.asarray(y_dhr, float)
        metrics["DHR"] = metrics_from_block(y_true, y_dhr, start, end) | {"elapsed_seconds": round(elapsed, 3)}
        results.append(("DHR", y_dhr, palette[idx])); idx += 1
    except Exception as e:
        print("[WARN] DHR falló ->", e)

    # ------- 4) Trigonometric Seasonal State-Space (FS-BSM) -------
    try:
        K       = int(hp.get("fourier_K", 3))
        maxiter = int(hp.get("maxiter", 20))
        t0 = time.perf_counter()
        y_trig = mh.trig_ss_impute_block(t, y_missing, start, end,
                                         fmin=fmin, fmax=fmax, M=K, maxiter=maxiter)
        elapsed = time.perf_counter() - t0
        y_trig = np.asarray(y_trig, float)
        metrics["FS-BSM"] = metrics_from_block(y_true, y_trig, start, end) | {"elapsed_seconds": round(elapsed, 3)}
        results.append(("FS-BSM", y_trig, palette[idx])); idx += 1
    except Exception as e:
        print("[WARN] FS-BSM falló ->", e)

    # ------- 5) TBATS (De Livera & Hyndman, 2011) -------
    try:
        K    = int(hp.get("fourier_K", 3))
        p_ar = int(hp.get("p_ar", 1))
        t0 = time.perf_counter()
        y_tbats = mh.tbats_ss_impute_block(t, y_missing, start, end,
                                           fmin=fmin, fmax=fmax, M=K, p_ar=p_ar, maxiter=500)
        elapsed = time.perf_counter() - t0
        y_tbats = np.asarray(y_tbats, float)
        metrics["TBATS"] = metrics_from_block(y_true, y_tbats, start, end) | {"elapsed_seconds": round(elapsed, 3)}
        results.append(("TBATS", y_tbats, palette[idx])); idx += 1
    except Exception as e:
        print("[WARN] TBATS falló ->", e)

    # ------- 6) Fourier-TV (R-TVF) -------
    try:
        t0 = time.perf_counter()
        M = int(hp.get("fourier_K", 5))
        y_ftv, _ = mh.tvar_fourier_impute_block_multi(
            t, y_missing, start, end, M=M, degree=3, trend_deg=1, l2=1e-2,
            fmin=None, fmax=None, max_samples_ls=60000
        )
        elapsed = time.perf_counter() - t0
        y_ftv = np.asarray(y_ftv, float)
        metrics["Fourier-TV (R-TVF)"] = metrics_from_block(y_true, y_ftv, start, end) | {"elapsed_seconds": round(elapsed, 3)}
        results.append(("Fourier-TV (R-TVF)", y_ftv, palette[idx])); idx += 1
    except Exception as e:
        print("[WARN] Fourier-TV (R-TVF) falló ->", e)

    results = results[:6]
    return results, metrics

# =============== Main ===============
def main(config_path: str):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    file_path          = cfg["file_path"]
    simular            = bool(cfg.get("simular_huecos", True))
    missing_strategy   = cfg.get("missing_strategy", "random_points")
    missing_percentage = float(cfg.get("missing_percentage", 10))
    block_run_n        = cfg.get("block_run_n", "full")
    epochs             = int(cfg.get("epochs", 3))
    hp                 = dict(cfg.get("hyperparameters", {}))

    # --- carga datos ---
    t, y_true = load_dat(file_path)

    # --- genera huecos ---
    if simular:
        if missing_strategy == "random_points":
            y_missing, mask_pred = simulate_random_points(y_true, missing_percentage, seed=123)
            start = end = None
        elif missing_strategy == "block_runs":
            y_missing, start, end = simulate_block_run(y_true, missing_percentage, block_n=block_run_n, seed=123)
            mask_pred = None
        else:
            raise ValueError(f"Estrategia desconocida: {missing_strategy}")
    else:
        # Si no se simula, detecta NaN en serie para máscara/bloque
        y_missing = y_true.copy()
        nan_idx = np.where(~np.isfinite(y_missing))[0]
        if len(nan_idx) == 0:
            raise ValueError("No hay NaNs y simular_huecos=false. Nada que imputar.")
        if missing_strategy == "random_points":
            mask_pred = np.zeros_like(y_missing, dtype=bool)
            mask_pred[nan_idx] = True
            start = end = None
        else:
            start, end = nan_idx[0], nan_idx[-1] + 1
            mask_pred = None

    # --- ejecuta métodos según estrategia ---
    t_start = time.perf_counter()
    if missing_strategy == "random_points":
        methods_plot, metrics_dict = run_random_points(t, y_true, y_missing, mask_pred, hp, epochs)
        fig = plot_grid_3x2_random(
            t, y_true, y_missing, methods_plot, mask_pred,
            title="Comparativa 3x2 — (random_points)",
            zoom=True,
            pad_samples=int(hp.get("zoom_pad_samples_random", 600))
        )
    else:
        methods_plot, metrics_dict = run_block_runs(t, y_true, y_missing, start, end, hp, epochs)
        fig = plot_grid_3x2_block(t, y_true, y_missing, methods_plot, start, end, pad=200, title="Comparativa 3x2 — (block_runs)")
    total_elapsed = time.perf_counter() - t_start

    # --- guardar figura ---
    out_png = Path(config_path).with_name("comparativa_3x2.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # --- guardar métricas ---
    out_json = Path(config_path).with_name("metrics_summary.json")
    payload = {
        "file": str(file_path),
        "missing_strategy": missing_strategy,
        "missing_percentage": missing_percentage,
        "block_run_n": block_run_n,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "metrics": metrics_dict,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] Figura: {out_png}")
    print(f"[OK] Métricas: {out_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="input.json", help="Ruta a input.json")
    args = parser.parse_args()
    main(args.config)