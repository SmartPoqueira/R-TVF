#!/usr/bin/env python3
"""
Lightcurve Imputation  (train on ground truth + bidirectional block_runs)
- Algoritmos: lstm | gru | rnn | knn | moda | mediana | spline | cnn | arma | armax | bilstm | saits
- RNN y CNN usan ventanas con forward-fill interno para inputs con NaN.
- ENTRENAMIENTO: usa la serie ORIGINAL (ground truth) como fuente, pero EXCLUYE como targets
  los índices simulados (no se entrena directamente sobre lo que luego evaluamos).
- IMPUTACIÓN: se hace sobre la serie con huecos; en block_runs, se usa pase forward+backward y promedio.
- Métricas: MAE, MSE, RMSE, MAPE (%). 'metric' string o lista.
- Simulación de huecos:
    * "random_points": puntos sueltos al azar.
    * "block_runs": elimina p aleatorio y los siguientes n puntos, con n = block_run_n.
- Plot:
    * Fondo atenuado = ground truth completa.
    * Puntos observados usados para entrenar (negro).
    * En índices imputados: real (azul) vs generado (rojo).

Salida:
    - output.json con métricas + metadatos (resumen)
    - out/*_exec.json con info técnica (periodo, hueco, etc.)
    - resultados_ejecuciones/{algorithm}_YYYYMMDD_HHMMSS.json (log por ejecución)
"""

import os, json, math, datetime
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Input, Dense, Bidirectional
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    from tensorflow.keras.optimizers import Adam
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
import time

import plotly.express as px
import plotly.graph_objects as go

# Dependencias opcionales para ARMA y SAITS
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from pypots.imputation.saits import SAITS
    _HAS_PYPOTS = True
except Exception:
    _HAS_PYPOTS = False

from sklearn.preprocessing import StandardScaler

pd.set_option("display.precision", 8)

# cache periodo global (estimado en main)
_GLOBAL_PERIOD_EST = None

# ---------------- Visualización ----------------
def plot_sin_relleno(df):
    fig = px.line(df, x='tiempo', y='flujo', title='Serie temporal con huecos (sin relleno)')
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=True), title='Tiempo (HJD)'),
        yaxis=dict(title='Flujo')
    )
    fig.show()

def plot_comparativa(df_ground_truth, df_con_huecos, df_relleno, algoritmo, mask_pred):
    # usar máscara posicional para evitar desalineados
    mp = np.asarray(mask_pred, dtype=bool)
    mo = ~mp

    fig = go.Figure()
    # Fondo: ground truth completa
    fig.add_trace(go.Scatter(
        x=df_ground_truth['tiempo'], y=df_ground_truth['flujo'],
        mode='lines', name='Ground Truth', line=dict(color='lightgray', width=1)
    ))
    # Observados (negro)
    fig.add_trace(go.Scatter(
        x=df_con_huecos.iloc[mo]['tiempo'], y=df_con_huecos.iloc[mo]['flujo'],
        mode='markers', name='Observados (train)', marker=dict(color='black', size=3)
    ))
    # Predicho (rojo) – lo dibujamos ANTES para que no tape al azul
    fig.add_trace(go.Scatter(
        x=df_con_huecos.iloc[mp]['tiempo'],
        y=df_relleno.iloc[mp]['flujo_relleno'],
        mode='lines+markers', name=f'Generado ({algoritmo.upper()})',
        marker=dict(color='red', size=6),
        line=dict(color='red', width=1.2)
    ))
    # Reales en imputados (azul) – al final, encima de todo
    fig.add_trace(go.Scatter(
        x=df_ground_truth.iloc[mp]['tiempo'],
        y=df_ground_truth.iloc[mp]['flujo'],
        mode='markers', name='Real (imputados)',
        marker=dict(color='blue', size=7, symbol='circle')
    ))

    fig.update_layout(
        title=f'Real vs Generado ({algoritmo.upper()}) en índices imputados',
        xaxis=dict(rangeslider=dict(visible=True), title='Tiempo (HJD)'),
        yaxis=dict(title='Flujo')
    )
    fig.show()


# ---------------- Config / Datos ----------------
def cargar_configuracion(ruta):
    with open(ruta) as f:
        return json.load(f)

def cargar_datos(ruta_archivo):
    return pd.read_csv(
        ruta_archivo, sep=r'\s+', names=['tiempo','flujo'],
        comment='#', header=None
    )

def _to_index_set(arr):
    """Convierte None/list/ndarray/bool-mask en set de índices Python int."""
    if arr is None:
        return set()
    arr = np.asarray(arr)
    # Si viene como máscara booleana, conviértelo a índices
    if arr.dtype == bool:
        arr = np.where(arr)[0]
    return set(int(i) for i in arr.ravel().tolist())


def _left_pad_window_1d(serie, i, ventana):
    """
    Construye una ventana de longitud 'ventana' para predecir el índice i.
    - Usa serie[max(0, i-ventana):i]
    - Hace forward-fill dentro de esa subventana
    - Hace padding a la izquierda con el último observado anterior (o 0.0 si no hay)
    """
    start0 = max(0, i - ventana)
    xw = serie[start0:i].copy()

    # forward-fill dentro de la subventana
    last = 0.0
    has_last = False
    for j in range(len(xw)):
        if np.isnan(xw[j]):
            xw[j] = last if has_last else 0.0
        else:
            last = xw[j]; has_last = True

    # padding a la izquierda si falta longitud
    pad_len = ventana - len(xw)
    if pad_len > 0:
        seed = 0.0
        for k in range(start0 - 1, -1, -1):
            if not np.isnan(serie[k]):
                seed = serie[k]; break
        pad = np.full(pad_len, seed, dtype=float)
        xw = np.concatenate([pad, xw], axis=0)

    return xw  # shape=(ventana,)
# ---------------- Simulación de huecos ----------------
def simular_huecos_random_points(df, porcentaje, random_state=None):
    rng = np.random.default_rng(random_state)
    df2 = df.copy()
    N = len(df2)
    n_missing = int(np.ceil(N * (porcentaje/100.0)))
    idxs = rng.choice(np.arange(N), size=n_missing, replace=False)
    df2.loc[idxs,'flujo'] = np.nan
    return df2, np.sort(idxs)

def simular_huecos_block_runs_fixed(df, porcentaje, n_following, random_state=None):
    """
    Un solo bloque: elige índice p aleatorio y borra los siguientes n_following puntos
    (tamaño del bloque = n_following + 1). Se trunca si llega al final.
    El parámetro 'porcentaje' se IGNORA aquí.
    """
    rng = np.random.default_rng(random_state)
    N = len(df)
    if N <= 0:
        return df.copy(), np.array([], dtype=int)
    start = int(rng.integers(0, N))
    end = min(N, start + int(n_following) + 1)
    idxs = np.arange(start, end, dtype=int)
    df2 = df.copy()
    df2.loc[idxs, 'flujo'] = np.nan
    return df2, idxs

# ---------------- Preprocesado (datasets) ----------------
def construir_dataset_permisivo(flujo_escalado, ventana):
    """(legacy) Crea X,y con next-step y ffill interno; no excluye índices."""
    serie = flujo_escalado[:,0]
    X, y = [], []
    n = len(serie)
    for i in range(ventana, n):
        target = serie[i]
        if np.isnan(target):
            continue
        ventana_seq = serie[i-ventana:i].copy()
        last = 0.0
        for j in range(len(ventana_seq)):
            if np.isnan(ventana_seq[j]):
                ventana_seq[j] = last
            else:
                last = ventana_seq[j]
        X.append(ventana_seq)
        y.append(target)
    if len(X) == 0:
        raise ValueError("No se construyó ningún ejemplo (ventana).")
    X = np.array(X).reshape(-1, ventana, 1)
    return X, np.array(y)

def construir_dataset_permisivo_excluding(flujo_escalado, ventana, exclude_targets=None):
    """Excluye como *targets* los índices en exclude_targets."""
    serie = flujo_escalado[:,0]
    X, y = [], []
    n = len(serie)
    exclude_set = _to_index_set(exclude_targets)
    for i in range(ventana, n):
        if i in exclude_set:
            continue
        target = serie[i]
        if np.isnan(target):
            continue
        ventana_seq = serie[i-ventana:i].copy()
        last = 0.0
        for j in range(len(ventana_seq)):
            if np.isnan(ventana_seq[j]):
                ventana_seq[j] = last
            else:
                last = ventana_seq[j]
        X.append(ventana_seq)
        y.append(target)
    if len(X) == 0:
        raise ValueError("No se construyó ningún ejemplo (ventana/exclusiones).")
    X = np.array(X).reshape(-1, ventana, 1)
    return X, np.array(y)

# -------------- Modelos --------------
def build_rnn(tipo, ventana, unidades):
    capa = {'lstm': LSTM, 'gru': GRU, 'rnn': SimpleRNN}[tipo]
    model = Sequential([
        Input(shape=(ventana,1)),
        capa(unidades),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn(cnn_window, filters1, filters2, kernel1, kernel2, dropout, lr=1e-3):
    schedule = ExponentialDecay(initial_learning_rate=lr, decay_steps=200, decay_rate=0.9)
    opt = Adam(learning_rate=schedule)
    model = Sequential([
        tf.keras.Input(shape=(cnn_window, 1)),
        layers.GaussianNoise(0.1),
        layers.Conv1D(filters=filters1, kernel_size=kernel1, activation='elu', padding='same'),
        layers.MaxPool1D(pool_size=1),
        layers.Conv1D(filters=filters2, kernel_size=kernel2, activation='elu', padding='same'),
        layers.MaxPool1D(pool_size=1),
        layers.Conv1D(filters=5, kernel_size=5, activation='elu', padding='same'),
        layers.MaxPool1D(pool_size=1),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(128, activation='elu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=opt, loss='mse')
    return model

# -------------- Imputación auxiliar (autoregresivo / bloque) --------------
def imputar_autoregresivo(model, flujo_scaled, mask_pred, ventana):
    serie = flujo_scaled[:, 0].copy()
    for i in np.where(mask_pred)[0]:
        if i <= 0:
            xw = np.zeros((ventana,), dtype=float)
        else:
            xw = _left_pad_window_1d(serie, i, ventana)
        y_hat = model.predict(xw.reshape(1, ventana, 1), verbose=0).reshape(-1)[-1]
        serie[i] = float(y_hat)
    flujo_scaled[:, 0] = serie
    return flujo_scaled

def imputar_bloque_bidireccional_scaled(model, flujo_scaled, lost_idxs, ventana):
    N = len(flujo_scaled)
    serie = flujo_scaled[:,0].copy()
    # Forward
    work_f = serie.copy()
    pred_f = {}
    for i in lost_idxs:
        start = i - ventana
        if start < 0: continue
        xw = work_f[start:i]
        if np.isnan(xw).any():
            last = 0.0
            xw = xw.copy()
            for j in range(len(xw)):
                if np.isnan(xw[j]): xw[j] = last
                else: last = xw[j]
        y_hat = model.predict(xw.reshape(1, ventana, 1), verbose=0).reshape(-1)[-1]
        work_f[i] = float(y_hat); pred_f[i] = float(y_hat)

    # Backward
    work_b = serie[::-1].copy()
    idxs_b_rev = (N - 1) - lost_idxs[::-1]
    pred_b_rev = {}
    for i in idxs_b_rev:
        start = i - ventana
        if start < 0: continue
        xw = work_b[start:i]
        if np.isnan(xw).any(): continue
        y_hat = model.predict(xw.reshape(1, ventana, 1), verbose=0).reshape(-1)[-1]
        work_b[i] = float(y_hat); pred_b_rev[i] = float(y_hat)
    pred_b = {(N - 1) - i_rev: y for i_rev, y in pred_b_rev.items()}

    # Combinar
    flujo_pred_scaled = flujo_scaled.copy()
    for i in lost_idxs:
        if   i in pred_f and i in pred_b: flujo_pred_scaled[i, 0] = 0.5*(pred_f[i] + pred_b[i])
        elif i in pred_f:                 flujo_pred_scaled[i, 0] = pred_f[i]
        elif i in pred_b:                 flujo_pred_scaled[i, 0] = pred_b[i]
        else:
            # fallback a vecino observado más cercano
            left, right, val = i - 1, i + 1, np.nan
            while left >= 0 or right < N:
                if left >= 0 and not np.isnan(serie[left]):  val = serie[left];  break
                if right < N and not np.isnan(serie[right]): val = serie[right]; break
                left -= 1; right += 1
            flujo_pred_scaled[i, 0] = 0.0 if np.isnan(val) else val
    return flujo_pred_scaled

# -------------- Imputación (RNN / CNN) --------------
def rellenar_con_rnn(df, tipo, ventana, epocas, unidades, mask_pred, df_train=None, exclude_targets=None):
    """
    Entrena con df_train (ground truth) excluyendo 'exclude_targets' como targets.
    Imputa en df (con huecos).
    """
    df2 = df.copy().reset_index(drop=True)
    scaler = StandardScaler()
    flujo = df2['flujo'].astype(float).to_numpy().reshape(-1,1)
    flujo_inf_scaled = scaler.fit_transform(flujo)

    if df_train is None:
        df_train = df2.copy()
    flujo_train = df_train['flujo'].astype(float).to_numpy().reshape(-1,1)
    flujo_train_scaled = scaler.transform(flujo_train)

    # Dataset excluyendo targets simulados
    X, y = construir_dataset_permisivo_excluding(flujo_train_scaled, ventana, exclude_targets)
    model = build_rnn(tipo, ventana, unidades)
    model.fit(X, y, epochs=epocas, verbose=0)

    # Imputación
    lost_idxs = np.where(mask_pred)[0]
    es_bloque = (lost_idxs.size > 0) and np.all(np.diff(lost_idxs) == 1)
    if es_bloque:
        flujo_pred_scaled = imputar_bloque_bidireccional_scaled(model, flujo_inf_scaled, lost_idxs, ventana)
    else:
        flujo_pred_scaled = imputar_autoregresivo(model, flujo_inf_scaled, mask_pred, ventana)

    df2['flujo_relleno'] = scaler.inverse_transform(flujo_pred_scaled)
    return df2, int(len(y))

def rellenar_con_cnn(df, hp, mask_pred, epochs, df_train=None, exclude_targets=None):
    df2 = df.copy().reset_index(drop=True)
    scaler = StandardScaler()
    flujo = df2['flujo'].astype(float).to_numpy().reshape(-1,1)
    flujo_inf_scaled = scaler.fit_transform(flujo)

    if df_train is None:
        df_train = df2.copy()
    flujo_train = df_train['flujo'].astype(float).to_numpy().reshape(-1,1)
    flujo_train_scaled = scaler.transform(flujo_train)

    # Hiperparámetros
    cnn_window = int(hp['cnn_window'])
    filters1   = int(hp.get('cnn_filters1', 32))
    filters2   = int(hp.get('cnn_filters2', 32))
    kernel1    = int(hp.get('cnn_kernel1', 20))
    kernel2    = int(hp.get('cnn_kernel2', 10))
    dropout    = float(hp.get('cnn_dropout', 0.5))

    X, y = construir_dataset_permisivo_excluding(flujo_train_scaled, cnn_window, exclude_targets)
    model = build_cnn(cnn_window, filters1, filters2, kernel1, kernel2, dropout)
    model.fit(X, y, epochs=epochs, verbose=0)

    lost_idxs = np.where(mask_pred)[0]
    es_bloque = (lost_idxs.size > 0) and np.all(np.diff(lost_idxs) == 1)
    if es_bloque:
        flujo_pred_scaled = imputar_bloque_bidireccional_scaled(model, flujo_inf_scaled, lost_idxs, cnn_window)
    else:
        flujo_pred_scaled = imputar_autoregresivo(model, flujo_inf_scaled, mask_pred, cnn_window)

    df2['flujo_relleno'] = scaler.inverse_transform(flujo_pred_scaled)
    return df2, int(len(y))

def rellenar_con_arma(df, p, q, mask_pred):
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels no disponible. Instala: pip install statsmodels")
    df2 = df.copy().reset_index(drop=True)
    y = df2['flujo'].astype(float)
    model = SARIMAX(y, order=(int(p),0,int(q)), trend='c',
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred_mean = res.get_prediction().predicted_mean
    df2['flujo_relleno'] = df2['flujo']
    df2.loc[mask_pred, 'flujo_relleno'] = pred_mean.values[mask_pred]
    return df2

# -------------- Utilidades de periodo/gap y ventana adaptativa --------------
def estimate_period_ls(t, y, nfreq=20000):
    """Lomb–Scargle rápido (sin dependencias fuertes). Devuelve (P_dias, f_peak, dt_median)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float) - np.nanmean(y)
    dt_med = float(np.nanmedian(np.diff(t))) if len(t) > 1 else np.nan
    try:
        from astropy.timeseries import LombScargle
        freq, power = LombScargle(t, y).autopower(minimum_frequency=1e-6, maximum_frequency=10.0)
    except Exception:
        from scipy.signal import lombscargle
        w = np.linspace(1e-6*2*np.pi, 10.0*2*np.pi, nfreq)
        power = lombscargle(t, y, w, precenter=True, normalize=True)
        freq = w/(2*np.pi)
    if len(freq) == 0 or np.all(~np.isfinite(power)):
        raise RuntimeError("No se pudo estimar el periodo (LS).")
    i = int(np.nanargmax(power))
    f_peak = float(freq[i])
    P = 1.0 / f_peak if f_peak > 0 else np.nan
    return P, f_peak, dt_med

def fourier_matrix(t, period, K=3):
    t = np.asarray(t, float)
    cols = []
    for k in range(1, K+1):
        w = 2*np.pi*k/period
        cols.append(np.sin(w*t))
        cols.append(np.cos(w*t))
    return np.column_stack(cols) if cols else None

def compute_largest_gap_samples(t):
    """Devuelve (gap_samples_est, gap_duration_days). Basado en el mayor salto de tiempo entre puntos."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return 0, 0.0
    diffs = np.diff(t)
    max_gap = float(np.nanmax(diffs))
    dt_med = float(np.nanmedian(diffs))
    if not np.isfinite(dt_med) or dt_med <= 0:
        return 0, max_gap
    est = int(max(0, round(max_gap / dt_med) - 1))
    return est, max_gap

def pick_window(samples_per_period, gap_samples, w_max=2048, base=64):
    s = float(samples_per_period)
    target = max(5*s, gap_samples + 3*s)
    w = int(math.ceil(target / base) * base)
    return min(w, int(w_max))

# -------- ARMAX y BiLSTM con sin/cos --------
def rellenar_con_armax(df, p, q, period_days, K, mask_pred):
    """ARMA(p,q) con senos/cosenos exógenos (ARMAX). Rellena solo en mask_pred."""
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels no disponible. Instala: pip install statsmodels")
    df2 = df.copy().reset_index(drop=True)
    y = df2['flujo'].astype(float).to_numpy()
    t = df2['tiempo'].astype(float).to_numpy()
    X = fourier_matrix(t, period_days, K=K)
    model = SARIMAX(y, order=(int(p),0,int(q)), trend='c', exog=X,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred_mean = res.get_prediction(exog=X).predicted_mean
    df2['flujo_relleno'] = df2['flujo']
    df2.loc[mask_pred, 'flujo_relleno'] = pred_mean[mask_pred]
    return df2

def _make_sincos_feats(t, period_days, K):
    t = np.asarray(t, float)
    feats = []
    for k in range(1, K+1):
        w = 2*np.pi*k/period_days
        feats.append(np.sin(w*t))
        feats.append(np.cos(w*t))
    return np.vstack(feats).T  # (N, 2K)

def construir_dataset_permisivo_exog_excluding(flujo_scaled, tiempos, period_days, ventana, K, exclude_targets=None):
    """Construye X (ventana, 1+2K) y y, excluyendo targets en exclude_targets."""
    serie = flujo_scaled[:,0]
    tiempos = np.asarray(tiempos, float)
    exclude_set = _to_index_set(exclude_targets)
    X, y = [], []
    n = len(serie)
    sc_feats = _make_sincos_feats(tiempos, period_days, K)  # (N, 2K)
    for i in range(ventana, n):
        if i in exclude_set:
            continue
        target = serie[i]
        if np.isnan(target):
            continue
        ventana_seq = serie[i-ventana:i].copy()
        # forward fill interno
        last = 0.0
        for j in range(len(ventana_seq)):
            if np.isnan(ventana_seq[j]):
                ventana_seq[j] = last
            else:
                last = ventana_seq[j]
        # features sin/cos de la ventana
        fwin = sc_feats[i-ventana:i, :]  # (ventana, 2K)
        X.append(np.concatenate([ventana_seq[:,None], fwin], axis=1))
        y.append(target)
    if len(X) == 0:
        raise ValueError("No se construyó ningún ejemplo (ventana/exclusiones) para exógenas.")
    X = np.array(X).reshape(-1, ventana, 1 + 2*K)
    return X, np.array(y)

def build_bilstm_exog(ventana, unidades, K):
    model = Sequential([
        Input(shape=(ventana, 1 + 2*K)),
        Bidirectional(LSTM(unidades)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def imputar_bloque_bidireccional_scaled_exog(model, flujo_scaled, tiempos, lost_idxs, ventana, period_days, K):
    """Versión multi-canal (flujo + sin/cos)."""
    N = len(flujo_scaled)
    serie = flujo_scaled[:,0].copy()
    work_f = serie.copy()
    pred_f, pred_b = {}, {}

    feats = _make_sincos_feats(np.asarray(tiempos,float), period_days, K)

    # Forward
    for i in lost_idxs:
        start = i - ventana
        if start < 0: continue
        xw = work_f[start:i]
        if np.isnan(xw).any():
            # ffill de ventana
            last = 0.0
            xw = xw.copy()
            for j in range(len(xw)):
                if np.isnan(xw[j]): xw[j] = last
                else: last = xw[j]
        fwin = feats[start:i, :]
        Xin = np.concatenate([xw[:,None], fwin], axis=1).reshape(1, ventana, 1+2*K)
        y_hat = model.predict(Xin, verbose=0).reshape(-1)[-1]
        work_f[i] = float(y_hat); pred_f[i] = float(y_hat)

    # Backward
    work_b = serie[::-1].copy()
    tiempos_rev = np.asarray(tiempos)[::-1]
    feats_rev = _make_sincos_feats(tiempos_rev, period_days, K)
    idxs_b_rev = (N - 1) - lost_idxs[::-1]
    for i in idxs_b_rev:
        start = i - ventana
        if start < 0: continue
        xw = work_b[start:i]
        if np.isnan(xw).any():
            last = 0.0
            xw = xw.copy()
            for j in range(len(xw)):
                if np.isnan(xw[j]): xw[j] = last
                else: last = xw[j]
        fwin = feats_rev[start:i, :]
        Xin = np.concatenate([xw[:,None], fwin], axis=1).reshape(1, ventana, 1+2*K)
        y_hat = model.predict(Xin, verbose=0).reshape(-1)[-1]
        work_b[i] = float(y_hat); pred_b[i] = float(y_hat)
    pred_b_map = {(N - 1) - i_rev: y for i_rev, y in pred_b.items()}

    flujo_pred_scaled = flujo_scaled.copy()
    for i in lost_idxs:
        if   i in pred_f and i in pred_b_map: flujo_pred_scaled[i, 0] = 0.5*(pred_f[i] + pred_b_map[i])
        elif i in pred_f:                     flujo_pred_scaled[i, 0] = pred_f[i]
        elif i in pred_b_map:                 flujo_pred_scaled[i, 0] = pred_b_map[i]
    return flujo_pred_scaled


def rellenar_con_bilstm_sincos(df, hp, epocas, mask_pred, df_train=None, exclude_targets=None):
    """
    BiLSTM con características sinusoidales (PyTorch), adaptado de `modelos_hads.py`.
    - Requiere PyTorch (`pip install torch`).
    - Diseñado para imputar UN ÚNICO BLOQUE ('block_runs' strategy).
    - Entrena un modelo forward y otro backward sobre los datos observados.
    - Usa ambos para predecir autorregresivamente sobre el hueco y combina los resultados.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch no disponible. Instala: pip install torch")

    # --- Verificación de compatibilidad de la estrategia de huecos ---
    lost_idxs = np.where(np.asarray(mask_pred, dtype=bool))[0]
    es_bloque = (lost_idxs.size > 0) and np.all(np.diff(lost_idxs) == 1)
    if not es_bloque:
        raise NotImplementedError(
            "La implementación BiLSTM (PyTorch) actual solo soporta la imputación "
            "de un único bloque contiguo (estrategia 'block_runs')."
        )
    start, end = int(lost_idxs[0]), int(lost_idxs[-1] + 1)
    L = end - start # Longitud del hueco
    
    # --- Extracción de hiperparámetros ---
    window     = int(hp['window_size'])
    hidden     = int(hp['units'])
    K          = int(hp['fourier_K'])
    layers     = int(hp.get('bilstm_layers', 2))
    batch_size = int(hp.get('bilstm_batch_size', 128))
    lr         = float(hp.get('bilstm_lr', 1e-3))

    # --- Preparación de datos (numpy arrays) ---
    df2 = df.copy()
    t = df2['tiempo'].to_numpy(dtype=float)
    y_missing = df2['flujo'].to_numpy(dtype=float)
    N = len(y_missing)
    
    # Fuente de entrenamiento: ground truth, con los targets a imputar marcados como NaN
    # para que la función de ventanas los ignore.
    if df_train is None:
        df_train = df2.copy()
    y_train_base = df_train['flujo'].to_numpy(dtype=float)
    if exclude_targets is not None:
        y_train_base[np.asarray(exclude_targets, dtype=int)] = np.nan

    # --- Estimación de periodo (si es necesario) ---
    period_days = _GLOBAL_PERIOD_EST
    if period_days is None or not np.isfinite(period_days):
        try:
            period_days = estimate_period_ls(t, y_train_base)[0]
        except Exception:
            period_days = 1.0 # fallback

    # --- Helpers y definición del modelo (anidados para encapsular) ---
    def _linear_bridge(y, a, b):
        x = np.arange(len(y))
        return np.interp(np.arange(a, b), x[~np.isnan(y)], y[~np.isnan(y)])

    def make_features(tt, period_days, K):
        w = 2 * np.pi / period_days
        cols = []
        for k in range(1, K + 1):
            cols.append(np.sin(k * w * tt))
            cols.append(np.cos(k * w * tt))
        return np.column_stack(cols)

    def make_windows(series, feats, window_size):
        X, Y = [], []
        for i in range(len(series) - window_size):
            seg_y = series[i : i + window_size + 1]
            if np.isnan(seg_y).any():
                continue
            seg_f = feats[i : i + window_size, :]
            X.append(np.column_stack([seg_y[:-1][:, None], seg_f]))
            Y.append(seg_y[-1])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    class BiLSTMReg(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, bidirectional=True
            )
            self.fc = nn.Linear(hidden_size * 2, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    # --- Construcción de características y ventanas ---
    Xfeat = make_features(t, period_days, K)
    input_size = 1 + Xfeat.shape[1]
    
    X_all, y_all = make_windows(y_train_base, Xfeat, window)
    if len(X_all) < max(100, 3*window):
        y_imputed = y_missing.copy()
        y_imputed[start:end] = _linear_bridge(y_imputed, start, end)
        df2['flujo_relleno'] = y_imputed
        return df2, 0

    # Normalización manual (solo columna de flujo)
    mu, sd = X_all[..., 0].mean(), X_all[..., 0].std() + 1e-8
    X_all[..., 0] = (X_all[..., 0] - mu) / sd
    yn = (y_all - mu) / sd
    
    Xten = torch.from_numpy(X_all)
    yten = torch.from_numpy(yn)[:, None]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xten, yten), batch_size=batch_size, shuffle=True
    )

    # --- Entrenamiento Forward ---
    print(f"[INFO] BiLSTM-PyTorch: Entrenando modelo FORWARD ({epocas} épocas)...")
    model_f = BiLSTMReg(input_size, hidden, layers)
    opt = torch.optim.Adam(model_f.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    model_f.train()
    for e in range(epocas):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model_f(xb)
            loss = lossf(pred, yb)
            loss.backward(); opt.step()
    
    # --- Predicción Forward ---
    @torch.no_grad()
    def forecast_seq(model, seed_y, feats_for_gap, steps, mu_s, sd_s):
        model.eval()
        seqy, out = seed_y.copy(), []
        for i in range(steps):
            y_win = ((seqy[-window:] - mu_s) / sd_s).astype(np.float32)
            f_win = feats_for_gap[i:i+window, :].astype(np.float32)
            x_win = torch.from_numpy(np.column_stack([y_win[:, None], f_win])[None, :, :])
            yhat = model(x_win).cpu().numpy().ravel()[0]
            yhat = yhat * sd_s + mu_s
            seqy = np.append(seqy, yhat); out.append(yhat)
        return np.array(out, float)

    left_ctx = y_missing[:start][~np.isnan(y_missing[:start])]
    if len(left_ctx) < window:
        y_imputed = y_missing.copy()
        y_imputed[start:end] = _linear_bridge(y_imputed, start, end)
        df2['flujo_relleno'] = y_imputed
        return df2, len(X_all)
    
    feats_gap = Xfeat[start-window:end, :]
    y_fwd = forecast_seq(model_f, left_ctx, feats_gap, L, mu, sd)

    # --- Entrenamiento y Predicción Backward ---
    print(f"[INFO] BiLSTM-PyTorch: Entrenando modelo BACKWARD ({epocas} épocas)...")
    y_rev = y_train_base[::-1].copy()
    Xfeat_rev = make_features(t[::-1], period_days, K)
    
    Xb_all, yb_all = make_windows(y_rev, Xfeat_rev, window)
    if len(Xb_all) < max(100, 3*window):
        y_bwd = y_fwd[::-1] # Fallback si no hay suficientes datos para el modelo backward
    else:
        mub, sdb = Xb_all[..., 0].mean(), Xb_all[..., 0].std() + 1e-8
        Xb_all[..., 0] = (Xb_all[..., 0] - mub) / sdb
        ybn = (yb_all - mub) / sdb
        
        loader_b = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(Xb_all), torch.from_numpy(ybn)[:, None]),
            batch_size=batch_size, shuffle=True
        )
        model_b = BiLSTMReg(input_size, hidden, layers)
        optb = torch.optim.Adam(model_b.parameters(), lr=lr)
        model_b.train()
        for e in range(epocas):
            for xb, yb in loader_b:
                optb.zero_grad()
                pred = model_b(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
                loss.backward(); optb.step()

        right_ctx = y_missing[end:][~np.isnan(y_missing[end:])][::-1] # Invertido
        if len(right_ctx) < window:
            y_bwd = y_fwd[::-1]
        else:
            feats_gap_rev = Xfeat_rev[(N-end)-window : (N-start), :]
            y_bwd_rev = forecast_seq(model_b, right_ctx, feats_gap_rev, L, mub, sdb)
            y_bwd = y_bwd_rev[::-1]

    # --- Combinación y resultado ---
    k = np.arange(1, L + 1, dtype=float)
    w_b = k / (L + 1.0)
    w_f = 1.0 - w_b
    y_gap = w_f * y_fwd + w_b * y_bwd
    
    df2['flujo_relleno'] = df2['flujo'].copy()
    df2.loc[mask_pred, 'flujo_relleno'] = y_gap
    
    return df2, int(len(X_all))

# === Utilidad para imputar con el algoritmo seleccionado (reuso interno) ===
def _imputar_con_algoritmo(algorithm, df_con_huecos, hp, epocas, mask, df_train=None, exclude_targets=None):
    algorithm = algorithm.lower()
    if algorithm in ['lstm','gru','rnn']:
        df_r, _ = rellenar_con_rnn(
            df_con_huecos, algorithm,
            ventana=hp['window_size'],
            unidades=hp['units'],
            epocas=epocas,
            mask_pred=mask,
            df_train=df_train,
            exclude_targets=exclude_targets
        )
        return df_r
    elif algorithm == 'cnn':
        df_r, _ = rellenar_con_cnn(df_con_huecos, hp, mask, epocas, df_train=df_train, exclude_targets=exclude_targets)
        return df_r
    elif algorithm == 'arma':
        return rellenar_con_arma(df_con_huecos, hp.get('arma_p',2), hp.get('arma_q',1), mask)
    elif algorithm == 'armax':
        period_days = _GLOBAL_PERIOD_EST
        if period_days is None:
            t = df_con_huecos['tiempo'].to_numpy(); y = df_con_huecos['flujo'].to_numpy()
            period_days, _, _ = estimate_period_ls(t, y)
        K = int(hp.get('fourier_K', 3))
        return rellenar_con_armax(df_con_huecos, hp.get('arma_p',2), hp.get('arma_q',1), period_days, K, mask)
    elif algorithm == 'bilstm':
        K = int(hp.get('fourier_K', 3))
        df_r, _ = rellenar_con_bilstm_sincos(
            df_con_huecos, ventana=hp['window_size'], epocas=epocas, unidades=hp['units'], K=K,
            mask_pred=mask, df_train=df_train, exclude_targets=exclude_targets
        )
        return df_r
    elif algorithm == 'saits':
        return rellenar_con_saits(df_con_huecos, hp, epocas, mask)
    elif algorithm == 'knn':
        return rellenar_con_knn(df_con_huecos, hp['k'], mask)
    elif algorithm == 'moda':
        return rellenar_con_moda(df_con_huecos, mask)
    elif algorithm == 'mediana':
        return rellenar_con_mediana(df_con_huecos, mask)
    elif algorithm == 'spline':
        return rellenar_con_spline(df_con_huecos, hp['spline_order'], mask)
    else:
        raise ValueError("Algoritmo no reconocido.")

# -------------- Métricas y duty --------------
def calc_metric(y_true, y_pred, name):
    n = name.upper()
    if n == 'MAE':
        return float(np.mean(np.abs(y_true - y_pred)))
    if n == 'MSE':
        return float(np.mean((y_true - y_pred)**2))
    if n == 'RMSE':
        return float(np.sqrt(np.mean((y_true - y_pred)**2)))
    if n == 'MAPE':
        mask = y_true != 0
        if not np.any(mask):
            raise ValueError("MAPE indefinido (todos y_true = 0).")
        return float(np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask])) * 100.0)
    raise ValueError(f"Métrica no soportada: {name}")

def calcular_metricas(df_original, df_relleno, mask_pred, metrics):
    # --- ÚNICO BLOQUE MODIFICADO: robusto a NaN/Inf y a columnas objeto/arrays ---
    y_true = df_original.loc[mask_pred, 'flujo'].to_numpy(dtype=float)
    y_pred_raw = df_relleno.loc[mask_pred, 'flujo_relleno'].to_numpy()

    # Aplana si vienen arrays/objetos en cada celda
    if y_pred_raw.dtype == object:
        y_pred = np.array([
            (v.item() if hasattr(v, "shape") and v.size == 1 else (v[0] if isinstance(v, (list, np.ndarray)) else v))
            for v in y_pred_raw
        ], dtype=float)
    else:
        y_pred = y_pred_raw.astype(float)

    # Filtro pareado de valores finitos
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]

    if y_true.size == 0:
        return {m.upper(): float('nan') for m in metrics}

    return {m.upper(): calc_metric(y_true, y_pred, m) for m in metrics}

def calcular_duty_cycle(df):
    total = len(df)
    no_nan = df['flujo'].notna().sum()
    return float(no_nan) / float(total) if total > 0 else 0.0

# -------------- Otros algoritmos "simples" --------------
def rellenar_con_knn(df, k, mask_pred):
    # placeholder simple: KNN 1D (no distancia temporal)
    df2 = df.copy()
    serie = df2['flujo'].to_numpy()
    for i in np.where(mask_pred)[0]:
        left, right = i-1, i+1
        vals = []
        while left >= 0 or right < len(serie):
            if left >= 0 and not np.isnan(serie[left]):
                vals.append(serie[left])
                if len(vals) >= k: break
            if right < len(serie) and not np.isnan(serie[right]):
                vals.append(serie[right])
                if len(vals) >= k: break
            left -= 1; right += 1
        df2.at[i,'flujo_relleno'] = float(np.mean(vals)) if vals else np.nan
    # Rellenamos observados con su propio valor
    df2['flujo_relleno'] = df2.get('flujo_relleno', df2['flujo'])
    df2.loc[~mask_pred,'flujo_relleno'] = df2.loc[~mask_pred,'flujo']
    return df2

def rellenar_con_moda(df, mask_pred):
    df2 = df.copy()
    vals = df2.loc[~mask_pred, 'flujo'].dropna()
    if vals.empty:
        df2['flujo_relleno'] = df2['flujo']
        return df2
    mode = float(vals.mode().iloc[0]) if not vals.mode().empty else float(vals.median())
    df2['flujo_relleno'] = df2['flujo']
    df2.loc[mask_pred, 'flujo_relleno'] = mode
    return df2

def rellenar_con_mediana(df, mask_pred):
    df2 = df.copy()
    med = float(df2.loc[~mask_pred, 'flujo'].median())
    df2['flujo_relleno'] = df2['flujo']
    df2.loc[mask_pred, 'flujo_relleno'] = med
    return df2

def rellenar_con_spline(df, order, mask_pred):
    df2 = df.copy()
    df2['__interp__'] = df2['flujo'].interpolate(method='spline', order=order)
    df2['flujo_relleno'] = df2['flujo']
    df2.loc[mask_pred,'flujo_relleno'] = df2.loc[mask_pred,'__interp__']
    df2.drop(columns=['__interp__'], inplace=True)
    return df2

def rellenar_con_saits(df, hp, epocas, mask_pred, df_train=None, device=None):
    """
    SAITS con ventanas, manteniendo NaN y pasando máscara.
    Hiperparámetros opcionales en hp:
      - saits_max_steps (int): longitud de ventana (def 1024)
      - saits_stride    (int): salto entre ventanas (def max_steps//2)
      - saits_patience  (int): se ajusta automáticamente a < epochs si hace falta
    Args:
      device: GPU device string, e.g. 'cuda:0', 'cuda:1', or None for auto-detect
    """
    if not _HAS_PYPOTS:
        raise ImportError("pypots/SAITS no disponible. Instala: pip install pypots torch")

    import numpy as np

    # ---------- min-max con NaN (sin sklearn) ----------
    def _fit_minmax_nan(x):
        v = x[np.isfinite(x)]
        if v.size == 0:
            return 0.0, 1.0
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        scale = max(vmax - vmin, 1e-12)
        return vmin, scale

    def _transform_minmax_nan(x, vmin, scale):
        y = x.astype(np.float32).copy()
        m = np.isfinite(y)
        y[~m] = np.nan
        y[m]  = (y[m] - vmin) / scale
        return y

    def _inverse_minmax_nan(y, vmin, scale):
        z = y.astype(np.float32).copy()
        m = np.isfinite(z)
        z[m] = z[m] * scale + vmin
        return z

    # ---------- helpers de ventanas ----------
    def _windowify_with_mask(arr_L1, L, stride):
        """arr_L1: (N,1) con posibles NaN -> Xw: (B,L,1), Mw: (B,L,1), starts: (B,)"""
        N = arr_L1.shape[0]
        starts = list(range(0, max(0, N - L + 1), stride))
        if not starts or starts[-1] + L < N:
            starts.append(max(0, N - L))
        Xw = np.stack([arr_L1[s:s+L, :] for s in starts], axis=0).astype(np.float32)
        Mw = (~np.isnan(Xw)).astype(np.float32)
        return Xw, Mw, np.asarray(starts, dtype=int)

    def _aggregate_missing_only(y_sum, y_cnt, piece, start, missing_mask_global):
        L = piece.shape[0]
        sl = slice(start, start + L)
        miss = missing_mask_global[sl]
        y_sum[sl][miss] += piece[miss]
        y_cnt[sl][miss] += 1

    # ---------- datos ----------
    df_inf = df.copy().reset_index(drop=True)                         # serie con huecos
    df_src = (df_train if df_train is not None else df_inf).copy()    # fuente para fit del escalado

    y_train = df_src['flujo'].to_numpy().reshape(-1, 1).astype(np.float32)  # puede no tener NaN
    y_inf   = df_inf['flujo'].to_numpy().reshape(-1, 1).astype(np.float32)  # con NaN

    vmin, scale = _fit_minmax_nan(y_train.reshape(-1))
    x_train_scaled = _transform_minmax_nan(y_train, vmin, scale)  # (N,1)
    x_inf_scaled   = _transform_minmax_nan(y_inf,   vmin, scale)  # (N,1) conserva NaN

    N = x_train_scaled.shape[0]
    max_steps = int(hp.get('saits_max_steps', 1024))
    max_steps = max(16, min(max_steps, N))
    stride = int(hp.get('saits_stride', max_steps // 2 if max_steps > 1 else 1))
    stride = max(1, stride)

    # Ventanas para TRAIN (sobre ground truth; SAITS hace random-masking interno)
    Xw_train, Mw_train, starts_train = _windowify_with_mask(x_train_scaled, max_steps, stride)
    # SAITS solo necesita 'X' para entrenar
    train_set = {'X': Xw_train}

    # ---------- hiperparámetros SAITS ----------
    epochs_int = int(epocas)
    patience_cfg = hp.get('saits_patience', None)
    patience = None
    if patience_cfg is not None:
        try:
            p = int(patience_cfg)
            patience = max(0, min(p, max(0, epochs_int - 1)))
            if p != patience:
                print(f"[INFO] Ajustando SAITS patience de {p} a {patience} para epochs={epochs_int}")
        except Exception:
            patience = None

    # Detectar GPU para SAITS (usar device explícito si se proporciona)
    import torch
    if device is None:
        saits_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        saits_device = device
    print(f"[SAITS] Using device: {saits_device}")
    
    model_kwargs = dict(
        n_steps=max_steps,
        n_features=1,
        n_layers=int(hp.get('saits_n_layers', 2)),
        d_model=int(hp.get('saits_d_model', 64)),
        n_heads=int(hp.get('saits_n_heads', 4)),
        d_k=int(hp.get('saits_d_k', 16)),
        d_v=int(hp.get('saits_d_v', 16)),
        d_ffn=int(hp.get('saits_d_ffn', 128)),
        dropout=float(hp.get('saits_dropout', 0.1)),
        epochs=epochs_int,
        batch_size=int(hp.get('batch_size', 8)),
        device=saits_device,
        verbose=False,
    )
    if patience is not None:
        model_kwargs["patience"] = patience

    print(f"[INFO] SAITS(windowed): max_steps={max_steps}, stride={stride}, "
          f"epochs={epochs_int}, patience={model_kwargs.get('patience','None')}, "
          f"batches_train={Xw_train.shape[0]}")

    model = SAITS(**model_kwargs)
    model.fit(train_set)

    # ---------- imputación por ventanas sobre la serie con huecos ----------
    Xw_inf, Mw_inf, starts_inf = _windowify_with_mask(x_inf_scaled, max_steps, stride)

    y_hat_sum = np.zeros(N, dtype=np.float64)
    y_hat_cnt = np.zeros(N, dtype=np.int32)
    missing_global = np.isnan(y_inf.reshape(-1))  # qué puntos faltan realmente

    for xb, mb, s0 in zip(Xw_inf, Mw_inf, starts_inf):
        xb = xb[None, ...]  # (1,L,1)
        mb = mb[None, ...]  # (1,L,1)
        out = model.impute({'X': xb, 'X_m': mb})
        # claves según versión de PyPOTS
        if isinstance(out, dict):
            X_hat_win = out.get("imputed_X", out.get("X_hat", out.get("X")))
            if X_hat_win is None:
                raise RuntimeError(f"No se encontró la salida imputada de SAITS. Claves: {list(out.keys())}")
        else:
            X_hat_win = out
        win_scaled = np.asarray(X_hat_win).reshape(-1, 1)         # (L,1)
        win_unscaled = _inverse_minmax_nan(win_scaled, vmin, scale).reshape(-1)  # (L,)
        _aggregate_missing_only(y_hat_sum, y_hat_cnt, win_unscaled, s0, missing_global)

    # Promedio donde haya cobertura; el resto se mantiene
    y_hat = y_inf.reshape(-1).copy()
    covered = y_hat_cnt > 0
    y_hat[covered] = (y_hat_sum[covered] / np.maximum(y_hat_cnt[covered], 1))

    # Salida: solo sustituimos en los índices imputados simulados
    df_out = df_inf.copy()
    df_out['flujo_relleno'] = df_out['flujo']
    df_out.loc[mask_pred, 'flujo_relleno'] = y_hat[mask_pred]
    return df_out

# -------------- MAIN --------------
def main(ruta_config):
    start_time = time.time()
    elapsed = 0.0

    config = cargar_configuracion(ruta_config)

    file_path          = config['file_path']
    simular            = config['simular_huecos']
    missing_strategy   = config.get('missing_strategy', 'random_points')
    missing_percentage = config['missing_percentage']
    block_run_n_raw    = config.get('block_run_n', 5)  # puede ser int o 'full'/'half'

    algorithm          = config['algorithm'].lower()
    metric_field       = config['metric']
    hp                 = config['hyperparameters']
    epocas             = config['epochs']

    # Normalizar metric
    metrics = [metric_field] if isinstance(metric_field, str) else list(metric_field)

    # Cargar datos base
    df_original_full = cargar_datos(file_path)
    df_base = df_original_full.copy()       # ground truth (para train y duty)
    df_ground_truth = df_base.copy()

    # === Estimación de periodo y hueco grande natural ===
    global _GLOBAL_PERIOD_EST
    t_vals = df_ground_truth['tiempo'].to_numpy()
    y_vals = df_ground_truth['flujo'].to_numpy()
    try:
        P_days, f_peak, dt_med = estimate_period_ls(t_vals, y_vals)
        _GLOBAL_PERIOD_EST = P_days
    except Exception as e:
        print(f"[WARN] No se pudo estimar el periodo via Lomb–Scargle: {e}")
        P_days, dt_med = (np.nan, np.nan)

    gap_samples_est, gap_duration_days = compute_largest_gap_samples(t_vals)
    print(f"Calculando hueco... hueco encontrado = {gap_samples_est} muestras (~{gap_duration_days:.6f} días)")

    # Resolver n del bloque según política (solo para block_runs)
    def _norm(s): return str(s).strip().lower()
    ms = _norm(missing_strategy)
    block_run_n = None  # ignorado salvo en block_runs
    brn_raw = config.get('block_run_n', 5)

    if ms == 'block_runs':
        if isinstance(brn_raw, (str, bytes)):
            pol = _norm(brn_raw)
            if pol in ('full','completo','entero'):
                block_run_n = int(max(1, gap_samples_est))
            elif pol in ('half','mitad'):
                block_run_n = int(max(1, gap_samples_est // 2))
            else:
                try:
                    block_run_n = int(float(pol))  # permite "128" o "128.0"
                except Exception:
                    print(f"[WARN] block_run_n='{brn_raw}' no reconocido; uso n=5.")
                    block_run_n = 5
        else:
            block_run_n = int(brn_raw)
        print(f"Política de bloque: '{brn_raw}' -> n = {block_run_n}")
    else:
        print("Estrategia de huecos: random_points — 'block_run_n' se ignora.")

    # Ventana adaptativa (solo si hay block_runs)
    if missing_strategy == 'block_runs':
        if np.isfinite(P_days) and np.isfinite(dt_med) and dt_med > 0:
            s = P_days / dt_med  # muestras por periodo
            w_adapt = pick_window(s, block_run_n)
            # Aplicar a hyperparameters si corresponde
            if 'window_size' in hp:
                print(f"Ventana adaptativa (window_size): {hp['window_size']} -> {w_adapt} (s≈{s:.1f})")
                hp['window_size'] = int(w_adapt)
            if algorithm == 'cnn' and 'cnn_window' in hp:
                print(f"Ventana adaptativa (cnn_window): {hp['cnn_window']} -> {w_adapt} (s≈{s:.1f})")
                hp['cnn_window'] = int(w_adapt)
        else:
            print("[WARN] No se pudo calcular ventana adaptativa: periodo o dt inválidos.")

    # Simulación de huecos (para imputar/evaluar)
    if simular:
        print('Empezando a perder datos (simulación de huecos)...')
        if missing_strategy == 'block_runs':
            df_con_huecos, idxs_missing = simular_huecos_block_runs_fixed(
                df_base, missing_percentage, block_run_n, random_state=42
            )
        else:
            df_con_huecos, idxs_missing = simular_huecos_random_points(
                df_base, missing_percentage, random_state=42
            )
    else:
        df_con_huecos = df_base.copy()
        idxs_missing = np.array([], dtype=int)

    # Máscaras
    # Máscaras (Serie alineada al índice)
    mask_pred = pd.Series(False, index=df_con_huecos.index)
    if len(idxs_missing):
        mask_pred.iloc[idxs_missing] = True

    # Visual previa
    plot_sin_relleno(df_con_huecos)

    # -------- ENTRENAMIENTO sobre GROUND TRUTH excluyendo targets simulados --------
    print('Empiezo a recuperar (imputación)...')
    df_train_src = df_ground_truth              # entrenamos viendo ambos lados reales
    exclude_targets = idxs_missing              # pero sin usar esos índices como destino

    # Selección de algoritmo (imputación para SIMULADOS → métricas)
    num_train_targets = None
    if algorithm in ['lstm','gru','rnn']:
        df_relleno, num_train_targets = rellenar_con_rnn(
            df_con_huecos, algorithm,
            ventana=hp['window_size'],
            epocas=epocas,
            unidades=hp['units'],
            mask_pred=mask_pred,
            df_train=df_train_src,
            exclude_targets=exclude_targets
        )
    elif algorithm == 'cnn':
        df_relleno, num_train_targets = rellenar_con_cnn(
            df_con_huecos, hp, mask_pred, epochs=epocas,
            df_train=df_train_src, exclude_targets=exclude_targets
        )
    elif algorithm == 'arma':
        df_relleno = rellenar_con_arma(df_con_huecos, hp.get('arma_p',2), hp.get('arma_q',1), mask_pred)
    elif algorithm == 'armax':
        period_days = _GLOBAL_PERIOD_EST if _GLOBAL_PERIOD_EST is not None else estimate_period_ls(
            df_ground_truth['tiempo'].to_numpy(), df_ground_truth['flujo'].to_numpy()
        )[0]
        df_relleno = rellenar_con_armax(
            df_con_huecos, hp.get('arma_p',2), hp.get('arma_q',1),
            period_days, int(hp.get('fourier_K',3)), mask_pred
        )
    elif algorithm == 'bilstm':
        df_relleno, num_train_targets = rellenar_con_bilstm_sincos(
            df_con_huecos, 
            hp=hp, 
            epocas=epocas,
            mask_pred=mask_pred, 
            df_train=df_train_src, 
            exclude_targets=exclude_targets
        )
    elif algorithm == 'saits':
        df_relleno = rellenar_con_saits(df_con_huecos, hp, epocas, mask_pred)
    elif algorithm == 'knn':
        df_relleno = rellenar_con_knn(df_con_huecos, hp['k'], mask_pred)
    elif algorithm == 'moda':
        df_relleno = rellenar_con_moda(df_con_huecos, mask_pred)
    elif algorithm == 'mediana':
        df_relleno = rellenar_con_mediana(df_con_huecos, mask_pred)
    elif algorithm == 'spline':
        df_relleno = rellenar_con_spline(df_con_huecos, hp['spline_order'], mask_pred)
    else:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")

    # --- Gráfica post-imputación ---
    try:
        plot_comparativa(df_ground_truth, df_con_huecos, df_relleno, algorithm, mask_pred)
    except Exception as e:
        print(f"[WARN] No se pudo mostrar la gráfica comparativa: {e}")

    # Métricas sobre índices imputados
    metric_values = calcular_metricas(df_ground_truth, df_relleno, mask_pred, metrics)

    # Guardados
    os.makedirs("out", exist_ok=True)
    basename = os.path.splitext(os.path.basename(file_path))[0]
    out_exec = os.path.join("out", f"{basename}_{algorithm}_exec.json")

    # Duty cycle
    duty_original = calcular_duty_cycle(df_base)
    duty_simulado = calcular_duty_cycle(df_con_huecos)

    # Contadores de entrenamiento (debug/justificación)
    ventana = hp.get('window_size', hp.get('cnn_window', 32))
    total_train_targets = int(num_train_targets) if num_train_targets is not None else None
    pre_block = post_block = None
    if len(idxs_missing) > 0:
        b0, b1 = int(idxs_missing.min()), int(idxs_missing.max())
        # contamos targets *posibles* en ground truth que no se excluyen y no NaN
        targets_validos = []
        flujo_gt = df_ground_truth['flujo'].to_numpy()
        for i in range(ventana, len(flujo_gt)):
            if i in set(exclude_targets): continue
            if not np.isnan(flujo_gt[i]): targets_validos.append(i)
        pre_block  = int(np.sum(np.array(targets_validos) < b0))
        post_block = int(np.sum(np.array(targets_validos) > b1))

    elapsed = time.time() - start_time
    output_info = {
        "algorithm": algorithm,
        "hyperparameters": hp,
        "metrics": metric_values,
        "metrics_requested": [m.upper() for m in metrics],
        "num_points_total": int(len(df_original_full)),
        "num_points_missing_simulated": int(len(idxs_missing)),
        "duty_original": duty_original,
        "duty_after_simulation": duty_simulado,
        "window_used": int(ventana),
        "train_targets_used": total_train_targets,
        "train_targets_pre_block": pre_block,
        "train_targets_post_block": post_block,
        "elapsed_seconds": round(elapsed, 2)
    }

    info_exec = {
        "file": file_path,
        "missing_strategy": missing_strategy,
        "missing_percentage": missing_percentage,
        "block_run_n": int(block_run_n) if block_run_n is not None else None,
        "gap_samples_est": int(gap_samples_est),
        "gap_duration_days": float(gap_duration_days),
        "period_days_est": float(_GLOBAL_PERIOD_EST) if _GLOBAL_PERIOD_EST is not None else None
    }

    # Guardar info técnica (periodo/hueco) en out/
    with open(out_exec, "w") as fexec:
        json.dump(info_exec, fexec, indent=2)

    # Resumen global en output.json
    with open("output.json","w") as f:
        json.dump(output_info, f, indent=2)

    # === Guardar JSON por ejecución (con timestamp y campos extra) ===
    os.makedirs("resultados_ejecuciones_vr", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_exec_ts = f"resultados_ejecuciones_vr/{algorithm}_{ts}.json"
    info_exec_ts = {
        "algorithm": algorithm,
        "hyperparameters": hp,
        "metrics": metric_values,
        "num_points_total": int(len(df_original_full)),
        "num_missing_simulated": int(len(idxs_missing)),
        "num_imputed": int(mask_pred.sum()),
        "missing_percentage": missing_percentage,
        "file_input": file_path,
        "train_targets_total": total_train_targets,
        "train_targets_pre_block": pre_block,
        "train_targets_post_block": post_block,

        # añadidos útiles
        "period_days_est": float(_GLOBAL_PERIOD_EST) if _GLOBAL_PERIOD_EST is not None else None,
        "gap_samples_est": int(gap_samples_est),
        "gap_duration_days": float(gap_duration_days),
        "window_used": int(ventana)
    }
    with open(out_exec_ts, "w") as fexec2:
        json.dump(info_exec_ts, fexec2, indent=2)

    print(f">> out/exec: {out_exec}")
    print(f">> JSON de ejecución: {out_exec_ts}")

    print(f">> Tiempo total de ejecución: {elapsed:.2f} segundos")
    print(">> Proceso completado.")
    print(json.dumps(output_info, indent=2))


if __name__ == "__main__":
    main("input.json")