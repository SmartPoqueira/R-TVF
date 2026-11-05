#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imputación de UN bloque perdido con 4 métodos:
  - ARMA+Fourier (ARMAX, exógenas seno/coseno)
  - MIARMA (ARMA forward/backward local + blending)
  - BiLSTM con sin/cos AUTOMÁTICO (elige K armónicos o K=0)
  - Fourier-TV (multi): armónicos múltiples con coeficientes que varían suave

Uso:
    python imputa_unificado.py <ruta .dat> [--profile hads|lads] [--start START] [--nblock N]

Salida:
  - 2 figuras (overlay y zoom) a 400 dpi junto al .dat
  - metrics_<base>.txt con las métricas MAE/MSE/MAPE de cada método
"""

import sys, os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Silenciar warnings de convergencia de statsmodels
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")

pd.set_option('display.precision', 8)

# ========================= Utilidades básicas =========================

def load_dat(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", names=["time", "flux"], comment="#", header=None)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values("time").reset_index(drop=True)
    if df.empty or df["time"].isna().any() or df["flux"].isna().any():
        raise ValueError("Archivo vacío o con NaN tras limpieza: se requiere serie completa.")
    return df

def remove_block(df: pd.DataFrame, start: int, n: int):
    y_true = df["flux"].values.astype(float)
    y_missing = y_true.copy()
    Ntot = len(y_true)
    if n <= 0 or start < 0 or start + n > Ntot:
        raise ValueError(f"Bloque fuera de rango: start={start}, n={n}, N={Ntot}")
    end = start + n
    y_missing[start:end] = np.nan
    return y_true, y_missing, start, end

def compute_metrics_block(y_true: np.ndarray, y_imp: np.ndarray, start: int, end: int):
    yt = y_true[start:end]; yp = y_imp[start:end]
    mae = float(np.mean(np.abs(yt - yp)))
    mse = float(np.mean((yt - yp) ** 2))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(np.abs(yt), 1e-12, None))) * 100)
    return mae, mse, mape

def _linear_bridge(y: np.ndarray, a: int, b: int) -> np.ndarray:
    L = b - a
    if a > 0 and b < len(y):
        return np.linspace(y[a-1], y[b], num=L+2)[1:-1]
    x = np.arange(len(y))
    return np.interp(np.arange(a, b), x[~np.isnan(y)], y[~np.isnan(y)])

# ========================= Lomb–Scargle & selección de frecuencias =========================

def _ls_power(t, y, fmin, fmax, nfreq=20000):
    """Lomb–Scargle con astropy/scipy; fallback FFT (emergencia)."""
    try:
        from astropy.timeseries import LombScargle
        freq, power = LombScargle(t, y).autopower(
            minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=10
        )
        return freq, power
    except Exception:
        try:
            from scipy.signal import lombscargle
        except Exception:
            # fallback (uniformizamos y FFT)
            tu = np.linspace(t.min(), t.max(), min(len(t), 60000))
            yu = np.interp(tu, t, y - np.mean(y))
            freqs = np.fft.rfftfreq(len(tu), d=(tu[1]-tu[0]))
            mask = (freqs >= fmin) & (freqs <= fmax)
            spec = np.abs(np.fft.rfft(yu))**2
            return freqs[mask], spec[mask]
        w = np.linspace(2*np.pi*fmin, 2*np.pi*fmax, nfreq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            power = lombscargle(t, y, w, precenter=True, normalize=True)
        return w/(2*np.pi), power

def ls_peak_in_band(t, y, fmin=None, fmax=None, detrend=True, med_win=401, max_samples=60000):
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m]
    if detrend:
        base = pd.Series(y).rolling(window=med_win, center=True, min_periods=1).median().to_numpy()
        y = y - base
    if len(t) > max_samples:
        step = int(np.ceil(len(t) / max_samples))
        t = t[::step]; y = y[::step]
    baseline = t.max() - t.min()
    dt = np.median(np.diff(np.sort(t)))
    fnyq = 0.5 / dt
    if fmax is None: fmax = min(80.0, 0.9*fnyq)
    if fmin is None: fmin = max(1.0/(baseline*5.0), 1e-5)
    freq, power = _ls_power(t, y, fmin, fmax)
    i = int(np.nanargmax(power))
    return float(freq[i])

def ls_top_freqs(t, y, M=7, fmin=None, fmax=None, min_sep=None, max_samples=60000):
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m] - np.mean(y[m])
    if len(t) > max_samples:
        step = int(np.ceil(len(t) / max_samples))
        t = t[::step]; y = y[::step]
    baseline = t.max() - t.min()
    dt = np.median(np.diff(np.sort(t)))
    fnyq = 0.5 / dt
    if fmax is None: fmax = max(min(0.9*fnyq, 200.0), 1e-3)
    if fmin is None: fmin = max(1.0/(baseline*5.0), 1e-5)
    freq, power = _ls_power(t, y, fmin, fmax)
    if min_sep is None:
        min_sep = 1.0 / max(baseline, 1e-9)
    order = np.argsort(power)[::-1]
    chosen = []
    for idx in order:
        f = float(freq[idx])
        if all(abs(f - fc) > min_sep for fc in chosen):
            chosen.append(f)
        if len(chosen) >= M:
            break
    return np.array(chosen, float)

# ========================= ARMAX =========================

def fourier_matrix_from_freqs(t, freqs):
    t = np.asarray(t, float); cols = []
    for f in np.atleast_1d(freqs):
        w = 2*np.pi*f
        cols.append(np.sin(w*t))
        cols.append(np.cos(w*t))
    return np.column_stack(cols) if cols else None

def arma_impute_block(t, y_missing, start, end, p=3, q=5):
    t = np.asarray(t, float)
    y = np.asarray(y_missing, float).copy()
    L = end - start
    i0 = max(0, start - 3 * L)
    i1 = min(len(t), end + 3 * L)
    t_win = t[i0:i1]
    y_win = y[i0:i1].copy()
    start_win = start - i0
    end_win = end - i0
    
    mask_obs = np.isfinite(y_win)
    if mask_obs.sum() < 30:
        y[start:end] = _linear_bridge(y, start, end)
        return y
        
    mu_win = np.nanmean(y_win)
    std_win = np.nanstd(y_win)
    if std_win < 1e-12:
        std_win = 1.0
    y_win_centered = (y_win - mu_win) / std_win
    
    try:
        res = ARIMA(y_win_centered, order=(p, 0, q), trend="c",
                    enforce_stationarity=False, enforce_invertibility=False).fit(
                        method_kwargs={"warn_convergence": False}
                    )
        pred = res.get_prediction().predicted_mean
        y_smoothed = pred * std_win + mu_win
        y[start:end] = y_smoothed[start_win:end_win]
    except Exception as e:
        print(f"[ARMA Fallback due to error] {e}")
        y[start:end] = _linear_bridge(y, start, end)
    return y

def armax_impute_block_multi(time, y_missing, start, end, freqs_cpd, p=1, q=1, trend="c"):
    y = np.asarray(y_missing, float)
    X = fourier_matrix_from_freqs(np.asarray(time, float), freqs_cpd)
    res = ARIMA(y, exog=X, order=(p, 0, q), trend=trend,
                enforce_stationarity=False, enforce_invertibility=False).fit(
                    method_kwargs={"warn_convergence": False}
                )
    pred = res.get_prediction(exog=X).predicted_mean
    y_imp = y.copy(); y_imp[start:end] = pred[start:end]
    return y_imp, pred

# ========================= MIARMA =========================

def _fit_forecast(series: np.ndarray, steps: int, p: int, q: int, trend: str) -> np.ndarray:
    res = ARIMA(series, order=(p, 0, q), trend=trend,
                enforce_stationarity=False, enforce_invertibility=False).fit(
                    method_kwargs={"warn_convergence": False}
                )
    return res.forecast(steps=steps)

def miarma_impute_block(y_missing: np.ndarray, start: int, end: int,
                        p: int, q: int, trend: str,
                        win_left: int, win_right: int) -> np.ndarray:
    y_out = y_missing.copy()
    L = end - start; Ntot = len(y_out)
    i0 = max(0, start - win_left); i1 = min(Ntot, end + win_right)
    left_hist = y_out[i0:start]; right_hist = y_out[end:i1]
    left_hist = left_hist[~np.isnan(left_hist)]
    right_hist = right_hist[~np.isnan(right_hist)]
    min_needed = max(20, 3 * (p + q + 1))
    if len(left_hist) < min_needed or len(right_hist) < min_needed:
        y_out[start:end] = _linear_bridge(y_out, start, end); return y_out
    try:
        y_fwd = _fit_forecast(left_hist, steps=L, p=p, q=q, trend=trend)
        y_bwd = _fit_forecast(right_hist[::-1], steps=L, p=p, q=q, trend=trend)[::-1]
    except Exception:
        y_out[start:end] = _linear_bridge(y_out, start, end); return y_out
    k = np.arange(1, L + 1, dtype=float); w_b = k / (L + 1.0); w_f = 1.0 - w_b
    y_gap = w_f * y_fwd + w_b * y_bwd
    y_out[start:end] = y_gap; return y_out

# ========================= BiLSTM (sin/cos AUTOMÁTICO) =========================

def _quick_best_K(t, y, f0, K_grid=(1,2,3), win=4096, min_gain=0.01):
    """
    Heurística barata para elegir K (nº armónicos) o K=0 si no aporta:
      - calcula SSE base vs. SSE con regresión lineal en sin/cos(k f0)
      - elige K que más reduce SSE; si la mejora relativa < min_gain -> K=0
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m]
    if len(t) == 0:
        return 1
    # usar ventana cerca del centro para evitar tendencias largas
    if len(t) > win:
        c = len(t)//2; h = win//2; t = t[c-h:c+h]; y = y[c-h:c+h]
    sse0 = float(np.sum((y - np.mean(y))**2))
    bestK, best_gain = 0, 0.0
    for K in sorted(set(K_grid)):
        if K <= 0:
            continue
        cols = []
        w = 2*np.pi*f0
        for k in range(1, K+1):
            cols.append(np.sin(k*w*t))
            cols.append(np.cos(k*w*t))
        X = np.column_stack(cols)
        # ridge pequeño para estabilidad
        XT_X = X.T @ X + 1e-6*np.eye(X.shape[1])
        beta = np.linalg.solve(XT_X, X.T @ y)
        yhat = X @ beta
        sse = float(np.sum((y - yhat)**2))
        gain = max(0.0, (sse0 - sse) / max(1e-12, sse0))
        if gain > best_gain:
            best_gain, bestK = gain, K
    return bestK if best_gain >= min_gain else 1

def bilstm_impute_block_with_sinusoids_auto(
    t, y_missing, start, end, fmin, fmax,
    K_grid=(1,2,3), window: int = 512, hidden: int = 64, layers: int = 2,
    epochs: int = 3, batch: int = 256, lr: float = 1e-3,
    stride: int = 4, max_windows: int = 80000, seed: int = 123,
    device: str = None
):
    """
    BiLSTM autoregresivo con FEATURES seno/cos **opcionales**:
      - Estima f0 (LS en banda).
      - Elige automáticamente K∈K_grid mediante _quick_best_K (o K=0 si no aporta).
      - Entrena BiLSTM (ventanas que no crucen NaN). Predice forward + backward y mezcla.
    """
    try:
        import torch
        from torch import nn
        torch.manual_seed(seed); np.random.seed(seed)
    except ImportError:
        print("PyTorch no disponible: BiLSTM -> puente lineal.")
        y = y_missing.copy(); y[start:end] = _linear_bridge(y_missing, start, end)
        return y, {"K": 0, "f0": None, "note": "no torch"}

    # ------- seleccionar f0 y K -------
    f0 = ls_peak_in_band(t, y_missing.copy().astype(float), fmin=fmin, fmax=fmax, detrend=True)
    K_chosen = _quick_best_K(t, y_missing, f0, K_grid=K_grid, win=min(4096, len(t)))
    print(f"[BiLSTM auto] f0≈{f0:.4f} c/d  |  K elegido={K_chosen}")

    # ------- construir ventanas -------
    y = y_missing.astype(np.float32).copy()
    L = end - start
    N = len(y)
    w0 = 2*np.pi*f0

    def make_feats(tt, K):
        cols = []
        for k in range(1, K+1):
            cols.append(np.sin(k*w0*tt))
            cols.append(np.cos(k*w0*tt))
        return np.column_stack(cols) if K > 0 else np.zeros((len(tt), 0), dtype=np.float32)

    Xfeat = make_feats(np.asarray(t, float), K_chosen).astype(np.float32)  # (N, 2K)
    F = Xfeat.shape[1]

    def make_windows(series, feats, window, stride=1, cap=None):
        X, Y = [], []
        limit = len(series) - window
        i = 0
        while i < limit:
            seg_y = series[i:i+window+1]
            if not np.isnan(seg_y).any():
                seg_f = feats[i:i+window, :] if F > 0 else np.zeros((window, 0), np.float32)
                X.append(np.column_stack([seg_y[:-1][:, None], seg_f]))
                Y.append(seg_y[-1])
                if cap is not None and len(X) >= cap:
                    break
            i += stride
        if len(X) == 0:
            return np.empty((0, window, 1+F), np.float32), np.empty((0,), np.float32)
        return np.array(X, np.float32), np.array(Y, np.float32)

    X_all, y_all = make_windows(y, Xfeat, window, stride=stride, cap=max_windows)
    if len(X_all) < max(200, window):
        y[start:end] = _linear_bridge(y, start, end)
        return y, {"K": K_chosen, "f0": f0, "note": "datos insuficientes"}

    # normalización sólo de la columna y (col 0)
    mu = X_all[..., 0].mean(dtype=np.float32)
    sd = X_all[..., 0].std(dtype=np.float32) + 1e-8
    X_all[..., 0] = (X_all[..., 0] - mu) / sd
    yn = ((y_all - mu) / sd).astype(np.float32)
    import torch
    class BiLSTMReg(torch.nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden,
                                      num_layers=layers, batch_first=True, bidirectional=True)
            self.fc = torch.nn.Linear(hidden*2, 1)
        def forward(self, x):
            out, _ = self.lstm(x.float())
            return self.fc(out[:, -1, :])

    # Detectar GPU (usar device explícito si se proporciona)
    if device is None:
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(device)
    print(f"[BiLSTM] Using device: {torch_device}")

    input_size = 1 + F
    model_f = BiLSTMReg(input_size).to(torch_device)
    opt = torch.optim.AdamW(model_f.parameters(), lr=lr, weight_decay=1e-2)
    lossf = torch.nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_all), torch.from_numpy(yn)[:, None]),
        batch_size=batch, shuffle=True, drop_last=False
    )

    # ------- entrenamiento forward -------
    model_f.train(); losses_f=[]
    for e in range(epochs):
        tot, nb = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(torch_device), yb.to(torch_device)
            opt.zero_grad(); pred = model_f(xb); loss = lossf(pred, yb)
            loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        losses_f.append(tot/max(1,nb))
        print(f"BiLSTM(auto) FORWARD epoch {e+1}/{epochs} loss={losses_f[-1]:.6f}")

    @torch.no_grad()
    def forecast_seq(model, seed_y, feats_for_gap, steps):
        seqy = seed_y.copy().astype(np.float32); out = []
        for i in range(steps):
            y_win = ((seqy[-window:] - mu) / sd).astype(np.float32)
            f_win = feats_for_gap[i:i+window, :] if F>0 else np.zeros((window,0), np.float32)
            x_win = np.column_stack([y_win[:, None], f_win])[None, :, :]
            yhat = model(torch.from_numpy(x_win).to(torch_device)).cpu().numpy().ravel()[0]
            yhat = yhat * sd + mu
            seqy = np.append(seqy, np.float32(yhat)); out.append(float(yhat))
        return np.array(out, float)

    left_ctx = y[:start]; left_ctx = left_ctx[~np.isnan(left_ctx)]
    if len(left_ctx) < window:
        y[start:end] = _linear_bridge(y, start, end)
        return y, {"K": K_chosen, "f0": f0, "note": "contexto corto"}
    feats_gap = Xfeat[start-window:end, :] if F>0 else np.zeros((window+(end-start),0), np.float32)
    y_fwd = forecast_seq(model_f, left_ctx, feats_gap, L)

    # ------- backward (serie invertida) -------
    y_rev = y[::-1].copy()
    Xfeat_rev = Xfeat[::-1, :] if F>0 else np.zeros_like(Xfeat)
    Xb_all, yb_all = make_windows(y_rev, Xfeat_rev, window, stride=stride, cap=max_windows//2)
    if len(Xb_all) < max(200, window):
        y_bwd = y_fwd[::-1]
    else:
        mub = Xb_all[..., 0].mean(dtype=np.float32); sdb = Xb_all[..., 0].std(dtype=np.float32) + 1e-8
        Xb_all[..., 0] = (Xb_all[..., 0] - mub) / sdb
        ybn = ((yb_all - mub) / sdb).astype(np.float32)
        model_b = BiLSTMReg(input_size).to(torch_device)
        optb = torch.optim.AdamW(model_b.parameters(), lr=lr, weight_decay=1e-2)
        loader_b = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(Xb_all),
                                           torch.from_numpy(ybn)[:, None]),
            batch_size=batch, shuffle=True, drop_last=False
        )
        model_b.train()
        for e in range(max(1, epochs-1)):
            for xb, yb in loader_b:
                xb, yb = xb.to(torch_device), yb.to(torch_device)
                optb.zero_grad(); pred = model_b(xb); loss = torch.nn.functional.mse_loss(pred, yb)
                loss.backward(); optb.step()

        @torch.no_grad()
        def forecast_seq_rev(model, seed_y, feats_for_gap_rev, steps):
            seqy = seed_y.copy().astype(np.float32); out = []
            for i in range(steps):
                y_win = ((seqy[-window:] - mub) / sdb).astype(np.float32)
                f_win = feats_for_gap_rev[i:i+window, :] if F>0 else np.zeros((window,0), np.float32)
                x_win = np.column_stack([y_win[:, None], f_win])[None, :, :]
                yhat = model(torch.from_numpy(x_win).to(torch_device)).cpu().numpy().ravel()[0]
                yhat = yhat * sdb + mub
                seqy = np.append(seqy, np.float32(yhat)); out.append(float(yhat))
            return np.array(out, float)

        right_ctx = y_rev[:N-end]; right_ctx = right_ctx[~np.isnan(right_ctx)]
        feats_gap_rev = Xfeat_rev[(N-end)-window:(N-start), :] if F>0 else np.zeros((window+(end-start),0), np.float32)
        if len(right_ctx) < window:
            y_bwd = y_fwd[::-1]
        else:
            y_bwd_rev = forecast_seq_rev(model_b, right_ctx, feats_gap_rev, L)
            y_bwd = y_bwd_rev[::-1]

    # mezcla tipo MIARMA
    k = np.arange(1, L + 1, dtype=float); w_b = k / (L + 1.0); w_f = 1.0 - w_b
    y_gap = w_f * y_fwd + w_b * y_bwd
    y[start:end] = y_gap
    return y.astype(np.float64), {"K": K_chosen, "f0": f0, "note": "ok"}

# ========================= Fourier-TV (multi) =========================

def _build_tvar_design(t, freqs, degree=3, trend_deg=1):
    t = np.asarray(t, float); N = len(t)
    t0, t1 = float(t.min()), float(t.max())
    tau = (2.0 * (t - t0) / max(t1 - t0, 1e-12)) - 1.0
    cols = []
    if trend_deg >= 0:
        for j in range(trend_deg + 1): cols.append(tau**j)
    for f in freqs:
        w = 2*np.pi*f; s = np.sin(w*t); c = np.cos(w*t)
        for j in range(degree + 1):
            pj = tau**j; cols.append(pj * s); cols.append(pj * c)
    return np.column_stack(cols).astype(np.float32) if cols else np.zeros((N,0), np.float32)

def tvar_fourier_impute_block_multi(t, y_missing, start, end,
                                    M=7, degree=3, trend_deg=1, l2=1e-2,
                                    fmin=None, fmax=None, max_samples_ls=60000,
                                    return_std=False):
    y = np.asarray(y_missing, float).copy(); t = np.asarray(t, float)
    mask = np.isfinite(y)
    if mask.sum() < 50:
        y[start:end] = _linear_bridge(y, start, end)
        if return_std:
            return y, np.array([]), np.zeros_like(y)
        return y, np.array([])
    freqs = ls_top_freqs(t[mask], y[mask], M=M, fmin=fmin, fmax=fmax, max_samples=max_samples_ls)
    print(f"[Fourier-TV] {len(freqs)} frecuencias (c/d): {np.round(freqs, 5)}")
    X = _build_tvar_design(t, freqs, degree=degree, trend_deg=trend_deg)
    Xo, yo = X[mask, :].astype(np.float32), y[mask].astype(np.float32)
    XT_X = Xo.T @ Xo; lamI = (l2 * np.eye(XT_X.shape[0], dtype=np.float32))
    Cov_inv = XT_X + lamI
    try:
        beta = np.linalg.solve(Cov_inv, Xo.T @ yo)
        Cov_beta = np.linalg.inv(Cov_inv)
    except np.linalg.LinAlgError:
        res = np.linalg.lstsq(Xo.astype(np.float64), yo.astype(np.float64), rcond=None)
        beta = res[0].astype(np.float32)
        Cov_beta = np.linalg.pinv(Cov_inv.astype(np.float64)).astype(np.float32)
    yhat = (X @ beta).astype(np.float64)
    y_out = y.copy(); y_out[start:end] = yhat[start:end]
    
    if return_std:
        residuals = yo - yhat[mask]
        sigma2_noise = np.var(residuals) if len(residuals) > 0 else 1.0
        variance = sigma2_noise + np.sum((X @ Cov_beta) * X, axis=1)
        y_std = np.sqrt(np.clip(variance, 0, None))
        return y_out, freqs, y_std

    return y_out, freqs

# ========================= DHR (Dynamic Harmonic Regression) =========================
def dhr_impute_block(t, y_missing, start, end, fmin=None, fmax=None, M=3, p_ar=2, maxiter=20):
    t = np.asarray(t, float)
    y = np.asarray(y_missing, float).copy()
    L = end - start
    
    # 1. Local windowing for speed and stability
    i0 = max(0, start - 3 * L)
    i1 = min(len(t), end + 3 * L)
    t_win = t[i0:i1]
    y_win = y[i0:i1].copy()
    start_win = start - i0
    end_win = end - i0
    
    mask_obs = np.isfinite(y_win)
    if mask_obs.sum() < 30:
        y[start:end] = _linear_bridge(y, start, end)
        return y
    
    # Center the window
    mu_win = np.nanmean(y_win)
    y_win_centered = y_win - mu_win
    
    # 2. Estimate top M frequencies
    dt = np.median(np.diff(t_win)) if len(t_win) > 1 else 1.0
    freqs = ls_top_freqs(t_win[mask_obs], y_win_centered[mask_obs], M=M, fmin=fmin, fmax=fmax)
    
    freq_seasonal = []
    for f in freqs:
        period = 1.0 / (f * dt)
        if period > 2.0:
            freq_seasonal.append({'period': period, 'harmonics': 1})
            
    if not freq_seasonal:
        y[start:end] = _linear_bridge(y, start, end)
        return y
    
    # 3. Fit UnobservedComponents (DHR)
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        model = UnobservedComponents(
            y_win_centered,
            level='local level',
            stochastic_level=True,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=[True] * len(freq_seasonal),
            autoregressive=p_ar
        )
        res = model.fit(disp=False, maxiter=maxiter)
        pred = res.get_prediction()
        y_smoothed = pred.predicted_mean + mu_win
        y[start:end] = y_smoothed[start_win:end_win]
    except Exception as e:
        print(f"[DHR Fallback due to error] {e}")
        try:
            freqs_armax = freqs[:min(len(freqs), 2)]
            y_armax, _ = armax_impute_block_multi(t_win, y_win, start_win, end_win, freqs_armax, p=1, q=1)
            y[start:end] = y_armax[start_win:end_win]
        except Exception:
            y[start:end] = _linear_bridge(y, start, end)
            
    return y

# ========================= Trigonometric Seasonal Structural Model (FS-BSM) =========================
def _fourier_ols_fit(t_obs, y_obs, freqs):
    """OLS fit of Fourier model. Returns coefficient vector."""
    cols = [np.ones(len(t_obs))]
    for f in freqs:
        cols.append(np.cos(2 * np.pi * f * t_obs))
        cols.append(np.sin(2 * np.pi * f * t_obs))
    X = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(X, y_obs, rcond=None)
    return coeffs

def _fourier_ols_predict(t_pred, freqs, coeffs):
    """Evaluate Fourier OLS model at t_pred."""
    cols = [np.ones(len(t_pred))]
    for f in freqs:
        cols.append(np.cos(2 * np.pi * f * t_pred))
        cols.append(np.sin(2 * np.pi * f * t_pred))
    return np.column_stack(cols) @ coeffs

def trig_ss_impute_block(t, y_missing, start, end, fmin=None, fmax=None, M=3, maxiter=20):
    """FS-BSM with warm-start OLS Fourier blend.

    Replaces the MLE Kalman smoother (which diverges for long gaps) with a
    deterministic bidirectional Fourier projection estimated by OLS from the
    observed context windows on each side of the gap.
    Keeps the same signature as the original for full pipeline compatibility.

    Reference model: Hindrayanto et al. (2010), FS-BSM.
    Warm-start strategy: Proietti (2000) fixed-coefficient projection.
    """
    t = np.asarray(t, float)
    y = np.asarray(y_missing, float).copy()
    L = end - start

    # Context windows (same size logic as original)
    win = max(min(3 * L, start, len(t) - end), 200)
    t_left  = t[max(0, start - win):start]
    y_left  = y[max(0, start - win):start]
    t_right = t[end:min(len(t), end + win)]
    y_right = y[end:min(len(t), end + win)]

    mask_l = np.isfinite(y_left)
    mask_r = np.isfinite(y_right)
    if mask_l.sum() < 20 or mask_r.sum() < 20:
        y[start:end] = _linear_bridge(y, start, end)
        return y

    mu = np.nanmean(np.concatenate([y_left[mask_l], y_right[mask_r]]))
    freqs = ls_top_freqs(t_left[mask_l], y_left[mask_l] - mu, M=M, fmin=fmin, fmax=fmax)
    if len(freqs) == 0:
        y[start:end] = mu
        return y

    coeffs_fwd = _fourier_ols_fit(t_left[mask_l], y_left[mask_l] - mu, freqs)
    coeffs_bwd = _fourier_ols_fit(t_right[mask_r], y_right[mask_r] - mu, freqs)

    t_gap  = t[start:end]
    y_fwd  = _fourier_ols_predict(t_gap, freqs, coeffs_fwd) + mu
    y_bwd  = _fourier_ols_predict(t_gap, freqs, coeffs_bwd) + mu

    w = np.linspace(1.0, 0.0, L)
    y[start:end] = w * y_fwd + (1.0 - w) * y_bwd
    return y

# ========================= TBATS (De Livera, Hyndman & Koehler, 2011, JASA) =========================
def tbats_ss_impute_block(t, y_missing, start, end, fmin=None, fmax=None, M=3, p_ar=1, maxiter=500):
    """TBATS with warm-start: Holt-Winters ES level + Fourier OLS blend.

    Reference:
        De Livera, A. M., Hyndman, R. J., & Koehler, A. B. (2011).
        Forecasting time series with complex seasonal patterns using exponential smoothing.
        Journal of the American Statistical Association, 106(496), 1513-1527.

    Warm-start strategy for long gaps:
      - Estimate level+trend via Holt-Winters (alpha=0.3, beta=0.05) on the left window.
      - Estimate Fourier oscillatory component via OLS from both sides.
      - Blend the ES level anchor (fwd) with Fourier projection (bwd) using linear weights.
      - Eliminates Nelder-Mead divergence for N>=1000 gaps while preserving TBATS semantics.
    """
    t = np.asarray(t, float)
    y = np.asarray(y_missing, float).copy()
    L = end - start

    win = max(min(3 * L, start, len(t) - end), 200)
    t_left  = t[max(0, start - win):start]
    y_left  = y[max(0, start - win):start]
    t_right = t[end:min(len(t), end + win)]
    y_right = y[end:min(len(t), end + win)]

    mask_l = np.isfinite(y_left)
    mask_r = np.isfinite(y_right)
    if mask_l.sum() < 20 or mask_r.sum() < 20:
        y[start:end] = _linear_bridge(y, start, end)
        return y

    mu = np.nanmean(np.concatenate([y_left[mask_l], y_right[mask_r]]))
    dt = np.median(np.diff(t_left)) if len(t_left) > 1 else 1.0
    freqs = ls_top_freqs(t_left[mask_l], y_left[mask_l] - mu, M=M, fmin=fmin, fmax=fmax)
    if len(freqs) == 0:
        y[start:end] = mu
        return y

    # Holt-Winters level+trend (TBATS exponential smoothing anchor)
    alpha_hw, beta_hw = 0.3, 0.05
    y_l = y_left[mask_l]
    lev, trend = y_l[0], 0.0
    for val in y_l[1:]:
        lev_p, trend_p = lev, trend
        lev   = alpha_hw * val + (1 - alpha_hw) * (lev_p + trend_p)
        trend = beta_hw * (lev - lev_p) + (1 - beta_hw) * trend_p
    fwd_anchor = np.array([lev + (i + 1) * trend for i in range(L)])

    # Fourier oscillatory component (OLS from each side)
    coeffs_fwd = _fourier_ols_fit(t_left[mask_l], y_left[mask_l] - mu, freqs)
    coeffs_bwd = _fourier_ols_fit(t_right[mask_r], y_right[mask_r] - mu, freqs)
    t_gap = t[start:end]
    y_fourier_fwd = _fourier_ols_predict(t_gap, freqs, coeffs_fwd) + mu
    y_fourier_bwd = _fourier_ols_predict(t_gap, freqs, coeffs_bwd) + mu

    # Blend: ES level anchor (fwd) + Fourier (bwd) with linear weights
    w = np.linspace(1.0, 0.0, L)
    y_fwd_tbats = w * fwd_anchor + (1.0 - w) * y_fourier_fwd
    y[start:end] = w * y_fwd_tbats + (1.0 - w) * y_fourier_bwd
    return y


# ========================= Plots =========================

def _legend_ul(ax):
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99),
              frameon=True, framealpha=0.9, facecolor='white', edgecolor='0.8',
              borderaxespad=0.4)

def _plot_connect_outside_points_inside(ax, t, y, left, right, start, end, color, label,
                                        faint_inside=False, alpha_inside=0.35, MS=2.0, LW=1.0):
    L1 = max(left, 0); R1 = min(right, len(t))
    L_left, R_left = max(L1, 0), min(R1, start); lbl = label
    if R_left - L_left > 1:
        ax.plot(t[L_left:R_left], y[L_left:R_left], '-', lw=LW, color=color, label=lbl); lbl = None
    L_mid, R_mid = max(L1, start), min(R1, end)
    if R_mid > L_mid:
        ax.plot(t[L_mid:R_mid], y[L_mid:R_mid], '.', ms=MS, color=color, label=lbl); lbl = None
        tm = t[L_mid:R_mid]; ym = y[L_mid:R_mid]
        if faint_inside and np.isfinite(ym).sum() > 1:
            ax.plot(tm[np.isfinite(ym)], ym[np.isfinite(ym)], '-', lw=LW, color=color, alpha=alpha_inside)
    L_right, R_right = max(L1, end), min(R1, len(t))
    if R_right - L_right > 1:
        ax.plot(t[L_right:R_right], y[L_right:R_right], '-', lw=LW, color=color, label=lbl)

def plot_full_overlay(t, y_true, y_missing, series_dict, start, end):
    fig = plt.figure(figsize=(12, 5)); ax = plt.gca()
    _plot_connect_outside_points_inside(ax, t, y_true,   0, len(t), start, end, "black", "Original",         False)
    _plot_connect_outside_points_inside(ax, t, y_missing,0, len(t), start, end, "red",   "Con NaN (bloque)", False)
    for label, (y_imp, color) in series_dict.items():
        _plot_connect_outside_points_inside(ax, t, y_imp, 0, len(t), start, end, color, label, True)
    ax.axvspan(t[start], t[end-1], color="lightgray", alpha=0.6, label=f"Hueco N={end-start}")
    ax.set_xlabel("HJD"); ax.set_ylabel("Flux"); ax.set_title("Vista completa"); ax.grid(True); _legend_ul(ax)
    return fig

def plot_zoom_grid(t, y_true, y_missing, methods, start, end, pad=200):
    left = max(0, start - pad); right = min(len(t), end + pad)
    nrows = len(methods)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.5*nrows), sharex=True)
    if nrows == 1: axes = [axes]
    for ax, (label, y_imp, color) in zip(axes, methods):
        _plot_connect_outside_points_inside(ax, t, y_true,   left, right, start, end, "black", "Original",         False)
        _plot_connect_outside_points_inside(ax, t, y_missing,left, right, start, end, "red",   "Con NaN (bloque)", False)
        _plot_connect_outside_points_inside(ax, t, y_imp,    left, right, start, end, color,   label,              True)
        ax.axvspan(t[start], t[end-1], color="lightgray", alpha=0.6, label=f"Hueco N={end-start}")
        ax.set_ylabel("Flux"); ax.grid(True); _legend_ul(ax); ax.set_title(f"{label} — Zoom (±pad)")
    axes[-1].set_xlabel("HJD")
    fig.suptitle("Zoom", y=0.995, fontsize=14); fig.tight_layout(rect=[0,0,1,0.98])
    return fig

# ========================= Perfiles =========================

def presets(profile: str):
    """Devuelve diccionario con parámetros por perfil."""
    profile = profile.lower()
    if profile not in {"hads", "lads"}:
        profile = "lads"

    if profile == "hads":
        return dict(
            FMIN=3.0, FMAX=30.0,              # banda más estrecha
            N_FREQS_ARMAX=1,                  # frecuencia dominante
            M_FREQS_TV=5, TV_DEG=3, TV_TREND=1, TV_L2=1e-2,
            P_ARMAX=1, Q_ARMAX=1, TREND_X="c",
            PAD=400
        )
    else:  # LADS (δ-Scuti general, multifrecuencia)
        return dict(
            FMIN=5.0, FMAX=80.0,              # banda más ancha
            N_FREQS_ARMAX=2,                  # 1-2 componentes
            M_FREQS_TV=7, TV_DEG=3, TV_TREND=1, TV_L2=1e-2,
            P_ARMAX=1, Q_ARMAX=1, TREND_X="c",
            PAD=600
        )

# ========================= main =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat_path", help="Ruta al archivo .dat (cols: time flux)")
    ap.add_argument("--profile", choices=["hads","lads"], default="lads", help="Presets de parámetros")
    ap.add_argument("--start", type=int, default=None, help="Índice inicial del hueco (0-based). Si omites, se centra.")
    ap.add_argument("--nblock", type=int, default=1000, help="Tamaño del hueco a eliminar")
    args = ap.parse_args()

    df = load_dat(args.dat_path)
    t = df["time"].values; y = df["flux"].values
    N = len(df)

    pr = presets(args.profile)
    N_BLOCK = int(args.nblock)
    if args.start is None:
        START = max(0, (N - N_BLOCK)//4)
    else:
        START = int(args.start)
    end = START + N_BLOCK
    if not (0 <= START < end <= N):
        raise ValueError("Hueco fuera de rango respecto a la serie.")

    # Construir y_missing
    y_true, y_missing, start, end = remove_block(df, START, N_BLOCK)
    print(f"[{args.profile.upper()}] Bloque eliminado: [{start}:{end}) N={N_BLOCK}  "
          f"t≈{t[start]:.6f} → {t[end-1]:.6f}")

    # ----- ARMAX -----
    freqs_for_armax = ls_top_freqs(t, y_true, M=max(1, pr["N_FREQS_ARMAX"]),
                                   fmin=pr["FMIN"], fmax=pr["FMAX"])
    freqs_for_armax = freqs_for_armax[:max(1, pr["N_FREQS_ARMAX"])]
    print(f"[ARMAX] freqs (c/d): {np.round(freqs_for_armax,5)}")
    y_armax, _ = armax_impute_block_multi(
        t, y_missing, start, end, freqs_cpd=freqs_for_armax,
        p=pr["P_ARMAX"], q=pr["Q_ARMAX"], trend=pr["TREND_X"]
    )

    # ----- Fourier-TV (R-TVF) -----
    y_tvar, freqs_used_tv = tvar_fourier_impute_block_multi(
        t, y_missing, start, end, M=pr["M_FREQS_TV"], degree=pr["TV_DEG"],
        trend_deg=pr["TV_TREND"], l2=pr["TV_L2"],
        fmin=pr["FMIN"], fmax=pr["FMAX"], max_samples_ls=60000
    )

    # ----- DHR -----
    y_dhr = dhr_impute_block(
        t, y_missing, start, end, fmin=pr["FMIN"], fmax=pr["FMAX"], M=3, p_ar=2, maxiter=20
    )

    # ----- FS-BSM -----
    y_trig = trig_ss_impute_block(
        t, y_missing, start, end, fmin=pr["FMIN"], fmax=pr["FMAX"], M=3, maxiter=20
    )

    # ----- TBATS -----
    y_tbats = tbats_ss_impute_block(
        t, y_missing, start, end, fmin=pr["FMIN"], fmax=pr["FMAX"], M=3, p_ar=1, maxiter=20
    )

    # ----- Métricas (solo en el bloque) -----
    m_armax = compute_metrics_block(y_true, y_armax, start, end)
    m_tvar  = compute_metrics_block(y_true, y_tvar,   start, end)
    m_dhr   = compute_metrics_block(y_true, y_dhr,    start, end)
    m_trig  = compute_metrics_block(y_true, y_trig,    start, end)
    m_tbats = compute_metrics_block(y_true, y_tbats,   start, end)

    print("\nMétricas en el bloque imputado:")
    print(f"  ARMA+Fourier:              MAE={m_armax[0]:.6g}  MSE={m_armax[1]:.6g}  MAPE={m_armax[2]:.3f}%")
    print(f"  Fourier-TV (R-TVF):        MAE={m_tvar[0]:.6g}  MSE={m_tvar[1]:.6g}  MAPE={m_tvar[2]:.3f}%")
    print(f"  DHR:                       MAE={m_dhr[0]:.6g}   MSE={m_dhr[1]:.6g}   MAPE={m_dhr[2]:.3f}%")
    print(f"  FS-BSM:                   MAE={m_trig[0]:.6g}  MSE={m_trig[1]:.6g}  MAPE={m_trig[2]:.3f}%")
    print(f"  TBATS:                  MAE={m_tbats[0]:.6g} MSE={m_tbats[1]:.6g} MAPE={m_tbats[2]:.3f}%")

    # ----- Figuras -----
    series_for_full = {
        "ARMA+Fourier":       (y_armax, "blue"),
        "Fourier-TV (R-TVF)": (y_tvar,   "purple"),
        "DHR":                (y_dhr,    "green"),
        "FS-BSM":            (y_trig,   "orange"),
        "TBATS":           (y_tbats,  "brown")
    }
    methods_for_zoom = [
        ("ARMA+Fourier",        y_armax,  "blue"),
        ("Fourier-TV (R-TVF)",  y_tvar,   "purple"),
        ("DHR",                 y_dhr,    "green"),
        ("FS-BSM",             y_trig,   "orange"),
        ("TBATS",            y_tbats,  "brown")
    ]
    fig_full = plot_full_overlay(t, y_true, y_missing, series_for_full, start, end)
    fig_zoom = plot_zoom_grid(t, y_true, y_missing, methods_for_zoom, start, end, pad=pr["PAD"])

    base = os.path.splitext(os.path.basename(args.dat_path))[0]
    out_dir = os.path.dirname(args.dat_path) or "."
    path_full = os.path.join(out_dir, f"{base}_{args.profile}_overlay.png")
    path_zoom = os.path.join(out_dir, f"{base}_{args.profile}_zoom.png")
    fig_full.savefig(path_full, dpi=400, bbox_inches="tight")
    fig_zoom.savefig(path_zoom, dpi=400, bbox_inches="tight")
    plt.show()
    print(f"\nFiguras guardadas: {path_full}  |  {path_zoom}")

    # ----- Guardar métricas a .txt -----
    metrics_path = os.path.join(out_dir, f"{base}_{args.profile}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Archivo: {args.dat_path}\nPerfil: {args.profile}\nHueco: start={start} end={end} N={N_BLOCK}\n")
        f.write(f"ARMAX freqs(c/d): {np.round(freqs_for_armax,5).tolist()}\n")
        f.write(f"Fourier-TV freqs(c/d): {np.round(freqs_used_tv,5).tolist()}\n")
        f.write("\nMétricas (solo bloque):\n")
        f.write(f"ARMA+Fourier   MAE={m_armax[0]:.8g}  MSE={m_armax[1]:.8g}  MAPE={m_armax[2]:.5f}%\n")
        f.write(f"Fourier-TV     MAE={m_tvar[0]:.8g}  MSE={m_tvar[1]:.8g}  MAPE={m_tvar[2]:.5f}%\n")
        f.write(f"DHR            MAE={m_dhr[0]:.8g}   MSE={m_dhr[1]:.8g}   MAPE={m_dhr[2]:.5f}%\n")
        f.write(f"FS-BSM        MAE={m_trig[0]:.8g}  MSE={m_trig[1]:.8g}  MAPE={m_trig[2]:.5f}%\n")
        f.write(f"TBATS       MAE={m_tbats[0]:.8g} MSE={m_tbats[1]:.8g} MAPE={m_tbats[2]:.5f}%\n")
    print(f"Métricas guardadas en: {metrics_path}")

if __name__ == "__main__":
    main()

