import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to sys.path to import modules
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    import lightCurve as lc
    import modelos_hads_lads as mh
except ImportError as e:
    print(f"Error importing modules from parent directory: {e}")
    # We don't exit here to allow importing this module even if imports fail (e.g. for testing)
    lc = None
    mh = None

def load_data(file_path):
    """
    Loads a .dat file (time, flux).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path, sep=r"\s+", names=["time", "flux"], comment="#", header=None)
    df = df.dropna().reset_index(drop=True)
    t = df["time"].astype(float).to_numpy()
    y = df["flux"].astype(float).to_numpy()
    return t, y

def simulate_block_gap(y, start_index=None, gap_size=None, percentage=None, seed=42):
    """
    Simulates a block gap in the time series.
    If start_index is None, it is chosen randomly.
    If gap_size is None, it is calculated from percentage.
    """
    N = len(y)
    rng = np.random.default_rng(seed)
    
    if gap_size is None:
        if percentage is None:
            raise ValueError("Must provide either gap_size or percentage")
        gap_size = max(1, int(round(N * (float(percentage)/100.0))))
    else:
        gap_size = int(gap_size)
    
    if start_index is None:
        start_index = int(rng.integers(0, max(1, N - gap_size)))
    else:
        start_index = int(start_index)
        
    end_index = min(N, start_index + gap_size)
    
    y_missing = y.astype(float).copy()
    y_missing[start_index:end_index] = np.nan
    
    return y_missing, start_index, end_index

def get_metrics(y_true, y_imp, start, end):
    """
    Calculates MAE, MSE, MAPE for the imputed block.
    """
    yt = y_true[start:end]
    yp = y_imp[start:end]
    
    mae = float(np.mean(np.abs(yt - yp)))
    mse = float(np.mean((yt - yp) ** 2))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(np.abs(yt), 1e-12, None))) * 100)
    
    return {"MAE": mae, "MSE": mse, "MAPE": mape}
