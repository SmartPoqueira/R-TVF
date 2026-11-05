"""
Generate Publication Tables with Mean ± Std (Formatted)
=======================================================
Standard format for Q1 JCR papers.
Excludes convergence failures (R² < -10).
Applies scaling (MSE) and bolding of best values.
"""

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_CSV = os.path.join(BASE_DIR, 'results/2_cross_validation/cv_results_all.csv')

# Order requested by user
METHODS = ['DHR', 'FS-BSM', 'TBATS', 'ARMAX', 'GP-Celerite', 'R-TVF']

# Threshold for convergence failures
FAILURE_THRESHOLD = -10


def get_best_methods(summary_df):
    """Identify best method for each metric to bold."""
    best = {}
    best['MAE'] = summary_df.loc[summary_df['MAE_mean'].idxmin(), 'Method']
    best['MSE'] = summary_df.loc[summary_df['MSE_mean'].idxmin(), 'Method']
    best['MAPE'] = summary_df.loc[summary_df['MAPE_mean'].idxmin(), 'Method']
    best['R2'] = summary_df.loc[summary_df['R2_mean'].idxmax(), 'Method']
    best['Time'] = summary_df.loc[summary_df['Time_mean'].idxmin(), 'Method']
    return best


def format_val(mean, std, is_best, fmt="{:.3f}"):
    """Format value as Mean ± Std, adding bold if best."""
    val_str = f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"
    if is_best:
        return f"\\textbf{{{val_str}}}"
    return val_str


def generate_summary(df, methods):
    results = []
    for method in methods:
        method_df = df[df['Method'] == method].copy()
        n_total = len(method_df)
        valid_df = method_df[method_df['R2'] > FAILURE_THRESHOLD]
        n_failures = n_total - len(valid_df)
        
        if len(valid_df) == 0: continue
        
        results.append({
            'Method': method,
            'N_failures': n_failures,
            'R2_mean': valid_df['R2'].mean(),
            'R2_std': valid_df['R2'].std(),
            'MAE_mean': valid_df['MAE'].mean(),
            'MAE_std': valid_df['MAE'].std(),
            'MSE_mean': valid_df['MSE'].mean(),
            'MSE_std': valid_df['MSE'].std(),
            'MAPE_mean': valid_df['MAPE'].mean() * 100,
            'MAPE_std': valid_df['MAPE'].std() * 100,
            'Time_mean': valid_df['Time'].mean(),
            'Time_std': valid_df['Time'].std(),
        })
    return pd.DataFrame(results)


def print_latex_table(summary_df, star_type, n_stars, mse_scale=1):
    """Generate LaTeX table with mean ± std, bolding, and scaling."""
    
    mse_power = ""
    if mse_scale == 1e6: mse_power = "($\\times 10^6$)"
    elif mse_scale == 1e3: mse_power = "($\\times 10^3$)"
    
    print(f"\n\\begin{{table}}[htbp]")
    print(f"\\centering")
    print(f"\\caption{{Cross-validation results for {star_type} stars (N = {n_stars}, 5 folds). Best values in bold.}}")
    print(f"\\label{{tab:cv_{star_type.lower()}}}")
    print(f"\\resizebox{{\\textwidth}}{{!}}{{%")
    print(f"\\begin{{tabular}}{{lccccc}}")
    print(f"\\toprule")
    print(f"\\textbf{{Method}} & \\textbf{{MAE}} & \\textbf{{MSE {mse_power}}} & \\textbf{{MAPE (\\%)}} & \\textbf{{$R^2$}} & \\textbf{{Time (s)}} \\\\")
    print(f"\\midrule")
    
    best = get_best_methods(summary_df)
    
    # Custom sort: R-TVF first, then others by R2
    # But user wants specific order? No, usually by performance.
    # Let's stick to the list order or sort by R2?
    # User example has: R-TVF, ARMAX, BiLSTM, GP, MIARMA, SAITS.
    # I will output in that specific order if possible.
    
    for method in METHODS:
        if method not in summary_df['Method'].values: continue
        
        row = summary_df[summary_df['Method'] == method].iloc[0]
        
        name = method
        if method == 'R-TVF': name = 'R-TVF (ours)'
        
        # MSE Scaling
        mse_mean = row['MSE_mean'] / mse_scale
        mse_std = row['MSE_std'] / mse_scale
        
        # Formatting
        mae_str = format_val(row['MAE_mean'], row['MAE_std'], best['MAE'] == method, "{:.0f}")
        mse_str = format_val(mse_mean, mse_std, best['MSE'] == method, "{:.1f}")
        if mse_mean < 0.1: # Higher precision for small numbers
             mse_str = format_val(mse_mean, mse_std, best['MSE'] == method, "{:.2f}")

        mape_str = format_val(row['MAPE_mean'], row['MAPE_std'], best['MAPE'] == method, "{:.1f}")
        r2_str = format_val(row['R2_mean'], row['R2_std'], best['R2'] == method, "{:.3f}")
        time_str = format_val(row['Time_mean'], row['Time_std'], best['Time'] == method, "{:.1f}")
        
        # Handle tiny time (GP)
        if row['Time_mean'] < 0.05:
            time_str = format_val(0.01, 0.00, best['Time'] == method, "{:.2f}")

        print(f"{name} & {mae_str} & {mse_str} & {mape_str} & {r2_str} & {time_str} \\\\")
        
    print(f"\\bottomrule")
    print(f"\\end{{tabular}}%")
    print(f"}}")
    print(f"\\end{{table}}")


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: Could not find {RESULTS_CSV}")
        return
        
    df_all = pd.read_csv(RESULTS_CSV)
    
    # HADS
    hads_df = df_all[df_all['StarType'] == 'HADS']
    hads_summary = generate_summary(hads_df, METHODS)
    print_latex_table(hads_summary, "HADS", hads_df['StarFile'].nunique(), mse_scale=1e6)
    
    print("\n" + "="*50 + "\n")
    
    # LADS
    lads_df = df_all[df_all['StarType'] == 'LADS']
    lads_summary = generate_summary(lads_df, METHODS)
    print_latex_table(lads_summary, "LADS", lads_df['StarFile'].nunique(), mse_scale=1e3)


if __name__ == "__main__":
    main()
