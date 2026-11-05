#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   3. SENSITIVITY ANALYSIS - R-TVF Parameter Study                            ║
║   Delta Scuti Stars Missing Values Imputation                                ║
║   Analysis of fourier_K, degree, and l2 regularization parameters            ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script generates:
1. Parameter sensitivity tables (how R² varies with each parameter)
2. Line plots showing R² vs parameter value
3. Heatmaps for parameter interactions (e.g., fourier_K × degree)
4. Optimal parameter identification per star type

Required: Grid search results from step 1 (grid_search_hads.csv, grid_search_lads.csv)

Usage:
    python 3.sensitivity_analysis.py
"""

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

# ============= CONFIGURATION =============
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
GRID_SEARCH_DIR = PARENT_DIR / "results" / "1_grid_search"
OUTPUT_DIR = PARENT_DIR / "results" / "3_sensitivity"

# Create organized subdirectories
DIRS = {
    'tables': OUTPUT_DIR / "tables",
    'lineplots': OUTPUT_DIR / "lineplots",
    'heatmaps': OUTPUT_DIR / "heatmaps",
    'summary': OUTPUT_DIR / "summary"
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ============= STYLE (Q1 JCR Publication Quality) =============
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
FIGSIZE = (12, 7)
DPI = 600

# Publication-quality font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20
})

# R-TVF Parameters with LaTeX math notation
RTVF_PARAMS = {
    'fourier_K': r'Number of Fourier Harmonics ($K$)',
    'degree': r'Polynomial Degree ($d$)',
    'l2': r'Regularization Strength ($\lambda$)'
}

# Short labels for axes (descriptive + math)
PARAM_LABELS = {
    'fourier_K': r'Fourier $K$',
    'degree': r'Degree $d$',
    'l2': r'$\lambda$'
}

# ============= LOAD DATA =============
def load_grid_search_results():
    """Load grid search results for R-TVF only."""
    dfs = []
    
    for csv_name in ['grid_search_results.csv']:
        csv_path = GRID_SEARCH_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"  ✓ Loaded {csv_name}: {len(df)} rows")
    
    if not dfs:
        raise FileNotFoundError(f"No grid search results found in {GRID_SEARCH_DIR}")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter only R-TVF
    df_rtvf = df[df['Method'] == 'R-TVF'].copy()
    print(f"DEBUG: Analyzed R-TVF rows: {len(df_rtvf)}")
    # Parse Params JSON - handle CSV escaped quotes and newlines
    def parse_params(params_str):
        try:
            # Clean the string: remove newlines, normalize quotes
            clean = str(params_str).replace('\n', '').replace('\r', '')
            clean = clean.replace('""', '"').replace("'", '"')
            # Remove outer quotes if present
            if clean.startswith('"') and clean.endswith('"'):
                clean = clean[1:-1]
            return json.loads(clean)
        except Exception as e:
            return {}
    # Remove verbose debug parser and use standard (working) one
    def parse_params(params_str):
        try:
            clean = str(params_str).replace('\n', '').replace('\r', '')
            clean = clean.replace('""', '"').replace("'", '"')
            if clean.startswith('"') and clean.endswith('"'):
                clean = clean[1:-1]
            return json.loads(clean)
        except:
            return {}

    params_df = df_rtvf['Params'].apply(parse_params).apply(pd.Series)
    
    # Clean column names
    params_df.columns = [str(col).strip().replace('\n', '').replace(' ', '') for col in params_df.columns]
    
    # Concat
    df_rtvf = pd.concat([df_rtvf.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)
    
    # FORCE KEEP ALL ROWS (Fix for Lambda=0 dropping)
    # df_rtvf = df_rtvf[df_rtvf['fourier_K'].notna()] 
    
    # Fill R2 NaNs with divergence value
    df_rtvf['R2'] = df_rtvf['R2'].fillna(-1e12)
    
    # Filter only positive side of divergence (allow all negatives)
    df_rtvf = df_rtvf[df_rtvf['R2'] <= 1.1]
        
    print(f"DEBUG: Final R-TVF rows: {len(df_rtvf)}")
    if 'l2' in df_rtvf.columns:
        print(f"DEBUG: Final l2 values: {df_rtvf['l2'].unique()}")
        print("DEBUG: Final l2 counts:\n", df_rtvf['l2'].value_counts())
    else:
        print("DEBUG: 'l2' column missing!")
    
    return df_rtvf

# ============= SENSITIVITY TABLES =============
def generate_sensitivity_tables(df):
    """Generate tables showing R² for each parameter value."""
    print("\n" + "="*60)
    print("  SENSITIVITY TABLES")
    print("="*60)
    
    for param in ['fourier_K', 'degree', 'l2']:
        if param not in df.columns:
            print(f"  ⚠ Parameter '{param}' not found in data")
            continue
        
        # Overall sensitivity
        sensitivity = df.groupby(param).agg({
            'R2': ['median', 'std', 'count'],
            'MAE': ['mean', 'std']
        }).round(4)
        sensitivity.columns = ['R2_mean', 'R2_std', 'count', 'MAE_mean', 'MAE_std']
        sensitivity = sensitivity.sort_index()
        
        print(f"\n  [{param}] Overall Sensitivity:")
        print(sensitivity.to_string())
        sensitivity.to_csv(DIRS['tables'] / f"sensitivity_{param}_overall.csv")
        
        # By star type
        for star_type in df['StarType'].dropna().unique():
            df_type = df[df['StarType'] == star_type]
            sens_type = df_type.groupby(param).agg({
                'R2': ['median', 'std'],
                'MAE': 'mean'
            }).round(4)
            sens_type.columns = ['R2_mean', 'R2_std', 'MAE_mean']
            sens_type.to_csv(DIRS['tables'] / f"sensitivity_{param}_{str(star_type).lower()}.csv")
        
        print(f"  ✓ sensitivity_{param}_*.csv saved")

# ============= LINE PLOTS =============
def plot_sensitivity_lines(df):
    """Line plots showing R² vs each parameter."""
    print("\n" + "="*60)
    print("  SENSITIVITY LINE PLOTS")
    print("="*60)
    
    for param in ['fourier_K', 'degree', 'l2']:
        if param not in df.columns:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, star_type in enumerate(['HADS', 'LADS']):
            ax = axes[idx]
            df_type = df[df['StarType'] == star_type]
            
            # Aggregate by parameter
            agg = df_type.groupby(param).agg({
                'R2': ['median', 'std']
            }).reset_index()
            agg.columns = [param, 'R2_mean', 'R2_std']
            agg = agg.sort_values(param)
            
            # Plot with larger markers
            color = '#2ecc71' if star_type == 'HADS' else '#3498db'
            ax.errorbar(agg[param], agg['R2_mean'], yerr=agg['R2_std'],
                       marker='o', capsize=8, linewidth=3, markersize=12,
                       color=color, capthick=2, elinewidth=2)
            
            # Labels with math mode
            ax.set_xlabel(RTVF_PARAMS.get(param, param), fontsize=16)
            ax.set_ylabel(r'Median $R^2$ ± std', fontsize=16)
            ax.set_title(f'{star_type}: $R^2$ vs {PARAM_LABELS.get(param, param)}', fontsize=18, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=14)
            ax.set_ylim(0, 1.0)  # Fixed Y axis for R²
            
            # Integer tick labels for K and degree
            if param in ['fourier_K', 'degree']:
                ax.set_xticks(agg[param].values)
                ax.set_xticklabels([str(int(v)) for v in agg[param].values], fontsize=14)
            
            # Mark best value
            best_idx = agg['R2_mean'].idxmax()
            best_val = agg.loc[best_idx, param]
            best_r2 = agg.loc[best_idx, 'R2_mean']
            ax.axvline(x=best_val, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            # Format annotation based on parameter type
            if param in ['fourier_K', 'degree']:
                label_val = f'{int(best_val)}'
            else:
                label_val = f'{best_val}'
            ax.annotate(f'Best: {label_val}', xy=(best_val, best_r2),
                       xytext=(10, 10), textcoords='offset points', fontsize=14, color='red',
                       fontweight='bold')
            
            if param == 'l2':
                ax.set_xscale('log')
        
        plt.suptitle(f'R-TVF Sensitivity: {RTVF_PARAMS.get(param, param)}', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(DIRS['lineplots'] / f"sensitivity_{param}.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ sensitivity_{param}.png")

# ============= HEATMAPS (Parameter Interactions with Fixed Lambda) =============
def plot_parameter_heatmaps(df):
    """Heatmap showing K × degree interaction with fixed optimal lambda."""
    print("\n" + "="*60)
    print("  PARAMETER INTERACTION HEATMAP")
    print("="*60)
    
    # Show K × degree heatmap for EACH lambda value
    if 'fourier_K' not in df.columns or 'degree' not in df.columns:
        print("  ⚠ Required columns not found")
        return
    
    # Both lambda values
    l2_values = [0.1, 0.01, 0.001, 0.0]
    aggregators = ['median', 'mean']
    
    for l2_val in l2_values:
        for agg_name in aggregators:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            for idx, star_type in enumerate(['HADS', 'LADS']):
                ax = axes[idx]
                df_type = df[(df['StarType'] == star_type) & (df['l2'] == l2_val)]
                
                # Pivot table: K vs degree
                # aggfunc based on current loop
                pivot = df_type.pivot_table(values='R2', index='fourier_K', columns='degree', aggfunc=agg_name)
                
                # Format tick labels as integers
                xticklabels = [str(int(v)) for v in pivot.columns]
                yticklabels = [str(int(v)) for v in pivot.index]
                
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                           vmin=0, vmax=1, ax=ax, linewidths=1.5,
                           xticklabels=xticklabels, yticklabels=yticklabels,
                           annot_kws={'size': 16, 'weight': 'bold'})
                
                title_agg = "Mean" if agg_name == 'mean' else "Median"
                ax.set_title(f'{star_type}: $R^2$', fontsize=18, fontweight='bold')
                ax.set_xlabel(PARAM_LABELS.get('degree', 'degree'), fontsize=16)
                ax.set_ylabel(PARAM_LABELS.get('fourier_K', 'fourier_K'), fontsize=16)
                ax.tick_params(axis='both', labelsize=14)
            
            # Format lambda for filename and title
            l2_str = str(l2_val).replace('.', '_')
            plt.suptitle(rf'R-TVF: Fourier $K$ × Degree $d$ (with $\lambda = {l2_val}$)', fontsize=20, fontweight='bold')
            plt.tight_layout()
            plt.savefig(DIRS['heatmaps'] / f"heatmap_K_vs_degree_lambda{l2_str}_{agg_name}.png", dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"  ✓ heatmap_K_vs_degree_lambda{l2_str}_{agg_name}.png")

# ============= OPTIMAL PARAMETERS =============
def find_optimal_parameters(df):
    """Find and report optimal parameters for R-TVF."""
    print("\n" + "="*60)
    print("  OPTIMAL PARAMETERS")
    print("="*60)
    
    results = []
    
    for star_type in ['HADS', 'LADS']:  # Explicit types instead of unique()
        df_type = df[df['StarType'] == star_type]
        
        if len(df_type) == 0:
            print(f"\n  {star_type}: No data")
            continue
        
        # Find best configuration
        best_idx = df_type['R2'].idxmax()
        best_row = df_type.loc[best_idx]
        
        result = {
            'StarType': star_type,
            'Best_R2': best_row['R2'],
            'Best_MAE': best_row['MAE'],
            'fourier_K': best_row.get('fourier_K', 'N/A'),
            'degree': best_row.get('degree', 'N/A'),
            'l2': best_row.get('l2', 'N/A'),
            'StarFile': best_row['StarFile']
        }
        results.append(result)
        
        print(f"\n  {star_type}:")
        print(f"    Best R² = {result['Best_R2']:.4f}")
        print(f"    Parameters: K={result['fourier_K']}, degree={result['degree']}, l2={result['l2']}")
    
    # Overall aggregated optimal
    df_opt = pd.DataFrame(results)
    df_opt.to_csv(DIRS['summary'] / "optimal_parameters.csv", index=False)
    print(f"\n  ✓ optimal_parameters.csv saved")
    
    # Statistical summary of best parameters across all stars
    print("\n  [Most Frequent Optimal Values]")
    for param in ['fourier_K', 'degree', 'l2']:
        if param in df.columns:
            # For each star, find optimal param value
            opt_per_star = df.loc[df.groupby('StarFile')['R2'].idxmax()][param]
            mode = opt_per_star.mode().values[0] if len(opt_per_star.mode()) > 0 else 'N/A'
            print(f"    {param}: mode = {mode}")
    
    return df_opt

# ============= ROBUSTNESS ANALYSIS =============
def analyze_robustness(df):
    """Analyze how robust R² is across different parameter settings."""
    print("\n" + "="*60)
    print("  ROBUSTNESS ANALYSIS (2×2 Grid)")
    print("="*60)
    
    l2_values = [0.1, 0.01, 0.001, 0.0]
    degree_values = [1, 3]
    
    # Generate one 2x2 grid per star type
    # Generate one 4x2 grid per star type (4 lambda values)
    for star_type in ['HADS', 'LADS']:
        fig, axes = plt.subplots(4, 2, figsize=(14, 24))
        
        for i, l2_val in enumerate(l2_values):
            for j, deg_val in enumerate(degree_values):
                ax = axes[i, j]
                df_type = df[(df['StarType'] == star_type) & 
                            (df['l2'] == l2_val) & 
                            (df['degree'] == deg_val)]
                
                param = 'fourier_K'
                if param in df_type.columns and len(df_type) > 0:
                    param_values = sorted(df_type[param].unique())
                    data = [df_type[df_type[param] == v]['R2'].values for v in param_values]
                    
                    if any(len(d) > 0 for d in data):
                        bp = ax.boxplot(data, labels=[str(int(v)) for v in param_values], patch_artist=True)
                        
                        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(param_values)))
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                
                ax.set_xlabel(r'Fourier $K$', fontsize=14)
                ax.set_ylabel(r'$R^2$', fontsize=14)
                ax.set_title(rf'$\lambda = {l2_val}$, $d = {int(deg_val)}$', fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.01)
                ax.tick_params(axis='both', labelsize=12)
        
        plt.suptitle(f'R-TVF Robustness ({star_type}): $R^2$ vs Fourier $K$', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(DIRS['summary'] / f"robustness_grid_{star_type.lower()}.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ robustness_grid_{star_type.lower()}.png")

# ============= LATEX TABLE =============
def generate_latex_sensitivity_table(df):
    """Generate LaTeX table for paper."""
    print("\n" + "="*60)
    print("  LATEX TABLE")
    print("="*60)
    
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{R-TVF Sensitivity Analysis: Mean R² by Parameter Value}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Parameter & Value & HADS R² & LADS R² \\",
        r"\midrule"
    ]
    
    for param in ['fourier_K', 'degree']:
        if param not in df.columns:
            continue
        
        for val in sorted(df[param].unique()):
            hads_r2 = df[(df['StarType'] == 'HADS') & (df[param] == val)]['R2'].mean()
            lads_r2 = df[(df['StarType'] == 'LADS') & (df[param] == val)]['R2'].mean()
            latex_lines.append(f"{param} & {int(val)} & {hads_r2:.4f} & {lads_r2:.4f} \\\\")
        
        latex_lines.append(r"\midrule")
    
    latex_lines[-1] = r"\bottomrule"
    latex_lines.extend([
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    latex_path = DIRS['tables'] / "sensitivity_table.tex"
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"  ✓ sensitivity_table.tex saved")

# ============= MAIN =============
def main():
    print("\n" + "="*70)
    print("  3. SENSITIVITY ANALYSIS - R-TVF Parameters")
    print("="*70 + "\n")
    
    # Load data
    print("Loading grid search results...")
    df = load_grid_search_results()
    
    # Generate all outputs
    generate_sensitivity_tables(df)
    plot_sensitivity_lines(df)
    plot_parameter_heatmaps(df)
    find_optimal_parameters(df)
    analyze_robustness(df)
    generate_latex_sensitivity_table(df)
    
    print("\n" + "="*70)
    print(f"  ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
    # List generated files
    print("Generated files:")
    for f in sorted(OUTPUT_DIR.rglob('*')):
        if f.is_file():
            print(f"  - {f.relative_to(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
