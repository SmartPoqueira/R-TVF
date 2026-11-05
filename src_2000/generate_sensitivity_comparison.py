
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# PATHS
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
GRID_SEARCH_DIR = PARENT_DIR / "results" / "1_grid_search"
OUTPUT_DIR = PARENT_DIR / "results" / "3_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    dfs = []
    for csv_name in ['grid_search_hads.csv', 'grid_search_lads.csv']:
        csv_path = GRID_SEARCH_DIR / csv_name
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
    
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['Method'] == 'R-TVF'].copy()
    
    def parse_params(params_str):
        try:
            return json.loads(str(params_str).replace("'", '"'))
        except:
            return {}
            
    params = df['Params'].apply(parse_params).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), params], axis=1)
    
    # Filter valid R2
    df = df[(df['R2'] >= -10) & (df['R2'] <= 1.1)] # Allow some negative but filter catastrophic
    return df

def plot_comparison(df, star_type):
    df_type = df[df['StarType'] == star_type]
    if len(df_type) == 0: return

    # Lambdas to plot: 0.0, 0.001, 0.01 (User asked for 0.1 but we have 0.01)
    lambdas = [0.0, 0.001, 0.01]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for i, l2_val in enumerate(lambdas):
        ax = axes[i]
        subset = df_type[df_type['l2'] == l2_val]
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, f'No Data\n$\lambda={l2_val}$', ha='center', va='center')
            continue
            
        pivot = subset.pivot_table(values='R2', index='fourier_K', columns='degree', aggfunc='mean')
        
        # Heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', 
                    vmin=0.0, vmax=1.0, ax=ax, cbar=(i==2))
        
        ax.set_title(f"$\lambda = {l2_val}$", fontsize=16, fontweight='bold')
        ax.set_xlabel("Degree ($d$)", fontsize=14)
        if i == 0:
            ax.set_ylabel("Fourier Harmonics ($K$)", fontsize=14)
        else:
            ax.set_ylabel("")
            
    fig.suptitle(f"R-TVF Sensitivity ({star_type}): $R^2$ vs $K, d$ across $\lambda$", fontsize=18, y=1.05)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"sensitivity_comparison_{star_type.lower()}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

def main():
    df = load_data()
    if df is not None:
        plot_comparison(df, 'HADS')
        plot_comparison(df, 'LADS')

if __name__ == "__main__":
    main()
