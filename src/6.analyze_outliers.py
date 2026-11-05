"""
Outlier Analysis and Robust Statistics for CV Results
=====================================================
Identifies extreme outliers and recalculates summary tables using
robust statistics (median, IQR) suitable for Q1 JCR publications.
"""

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
HADS_CSV = os.path.join(BASE_DIR, 'results/2_cross_validation/cv_results_hads.csv')
LADS_CSV = os.path.join(BASE_DIR, 'results/2_cross_validation/cv_results_lads.csv')

METHODS = ['DHR', 'FS-BSM', 'TBATS', 'ARMAX', 'GP-Celerite', 'R-TVF']

# Outlier threshold: R² values below this are considered failures
R2_OUTLIER_THRESHOLD = -10  # Any R² < -10 is clearly a model failure


def identify_outliers(df, metric='R2', threshold=-10):
    """Identify extreme outliers in the dataset."""
    outliers = df[df[metric] < threshold].copy()
    return outliers


def robust_summary(df, methods):
    """
    Calculate robust summary statistics using median and IQR.
    
    Returns DataFrame with:
    - Median R², MAE, MSE, MAPE, Time
    - IQR for each metric
    - Count of valid folds (excluding outliers)
    """
    results = []
    
    for method in methods:
        method_df = df[df['Method'] == method]
        
        # Filter out extreme outliers for R²
        valid_df = method_df[method_df['R2'] > R2_OUTLIER_THRESHOLD]
        n_outliers = len(method_df) - len(valid_df)
        n_total = len(method_df)
        
        if len(valid_df) == 0:
            print(f"Warning: All data points are outliers for {method}")
            continue
        
        # Calculate robust statistics
        result = {
            'Method': method,
            'N_valid': len(valid_df),
            'N_outliers': n_outliers,
            'N_total': n_total,
            # Median values
            'MAE_median': valid_df['MAE'].median(),
            'MSE_median': valid_df['MSE'].median(),
            'MAPE_median': valid_df['MAPE'].median() * 100,  # Convert to percentage
            'R2_median': valid_df['R2'].median(),
            'Time_median': valid_df['Time'].median(),
            # IQR values
            'MAE_iqr': valid_df['MAE'].quantile(0.75) - valid_df['MAE'].quantile(0.25),
            'MSE_iqr': valid_df['MSE'].quantile(0.75) - valid_df['MSE'].quantile(0.25),
            'MAPE_iqr': (valid_df['MAPE'].quantile(0.75) - valid_df['MAPE'].quantile(0.25)) * 100,
            'R2_iqr': valid_df['R2'].quantile(0.75) - valid_df['R2'].quantile(0.25),
            'Time_iqr': valid_df['Time'].quantile(0.75) - valid_df['Time'].quantile(0.25),
            # Also keep mean for comparison
            'R2_mean': valid_df['R2'].mean(),
            'R2_std': valid_df['R2'].std(),
        }
        results.append(result)
    
    return pd.DataFrame(results)


def compare_mean_vs_median(df, methods):
    """Show difference between mean and median for each method."""
    print("\n" + "="*80)
    print("COMPARISON: Mean vs Median R² (to show outlier impact)")
    print("="*80)
    print(f"{'Method':<15} {'Mean R² (all)':<18} {'Median R² (all)':<18} {'Mean R² (valid)':<18} {'Median R² (valid)':<18}")
    print("-"*80)
    
    for method in methods:
        method_df = df[df['Method'] == method]
        valid_df = method_df[method_df['R2'] > R2_OUTLIER_THRESHOLD]
        
        mean_all = method_df['R2'].mean()
        median_all = method_df['R2'].median()
        mean_valid = valid_df['R2'].mean() if len(valid_df) > 0 else np.nan
        median_valid = valid_df['R2'].median() if len(valid_df) > 0 else np.nan
        
        print(f"{method:<15} {mean_all:>18.4f} {median_all:>18.4f} {mean_valid:>18.4f} {median_valid:>18.4f}")


def generate_latex_table_robust(summary_df, star_type, n_stars):
    """Generate LaTeX table with robust statistics."""
    print(f"\n{'='*80}")
    print(f"LaTeX Table for {star_type} (Robust Statistics - Median ± IQR)")
    print(f"{'='*80}")
    
    # Sort by R2_median descending
    summary_df = summary_df.sort_values('R2_median', ascending=False)
    
    print(f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Cross-validation results for {star_type} stars (N = {n_stars}, 5 folds). 
Robust statistics: median $\\pm$ IQR to mitigate outlier effects. Best values in bold.}}
\\label{{tab:cv_{star_type.lower()}_robust}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{MAE}} & \\textbf{{MSE}} & \\textbf{{MAPE (\\%)}} & \\textbf{{$R^2$}} & \\textbf{{Time (s)}} \\\\
\\midrule""")
    
    for _, row in summary_df.iterrows():
        method = row['Method']
        if method == 'R-TVF':
            method = 'R-TVF (ours)'
        
        # Format with median ± IQR
        mae = f"{row['MAE_median']:.0f} $\\pm$ {row['MAE_iqr']:.0f}"
        mse = f"{row['MSE_median']:.0f} $\\pm$ {row['MSE_iqr']:.0f}"
        mape = f"{row['MAPE_median']:.1f} $\\pm$ {row['MAPE_iqr']:.1f}"
        r2 = f"{row['R2_median']:.3f} $\\pm$ {row['R2_iqr']:.3f}"
        time = f"{row['Time_median']:.1f} $\\pm$ {row['Time_iqr']:.1f}"
        
        print(f"{method} & {mae} & {mse} & {mape} & {r2} & {time} \\\\")
    
    print("""\\bottomrule
\\end{tabular}%
}
\\end{table}
""")


def main():
    print("="*80)
    print("OUTLIER ANALYSIS FOR CROSS-VALIDATION RESULTS")
    print(f"Outlier threshold: R² < {R2_OUTLIER_THRESHOLD}")
    print("="*80)
    
    # Load data
    hads_df = pd.read_csv(HADS_CSV)
    lads_df = pd.read_csv(LADS_CSV)
    
    # ----- HADS Analysis -----
    print("\n" + "="*80)
    print("HADS STARS ANALYSIS")
    print("="*80)
    
    hads_outliers = identify_outliers(hads_df)
    if len(hads_outliers) > 0:
        print(f"\nFound {len(hads_outliers)} extreme outliers in HADS:")
        print(hads_outliers[['StarFile', 'Method', 'Fold', 'R2', 'MAE']].to_string())
    else:
        print("\nNo extreme outliers found in HADS")
    
    compare_mean_vs_median(hads_df, METHODS)
    
    hads_summary = robust_summary(hads_df, METHODS)
    print("\nRobust Summary Statistics (HADS):")
    print(hads_summary[['Method', 'N_valid', 'N_outliers', 'R2_median', 'R2_iqr', 'R2_mean', 'R2_std']].to_string(index=False))
    
    n_hads_stars = hads_df['StarFile'].nunique()
    generate_latex_table_robust(hads_summary, 'HADS', n_hads_stars)
    
    # ----- LADS Analysis -----
    print("\n" + "="*80)
    print("LADS STARS ANALYSIS")
    print("="*80)
    
    lads_outliers = identify_outliers(lads_df)
    if len(lads_outliers) > 0:
        print(f"\nFound {len(lads_outliers)} extreme outliers in LADS:")
        print(lads_outliers[['StarFile', 'Method', 'Fold', 'R2', 'MAE']].to_string())
    else:
        print("\nNo extreme outliers found in LADS")
    
    compare_mean_vs_median(lads_df, METHODS)
    
    lads_summary = robust_summary(lads_df, METHODS)
    print("\nRobust Summary Statistics (LADS):")
    print(lads_summary[['Method', 'N_valid', 'N_outliers', 'R2_median', 'R2_iqr', 'R2_mean', 'R2_std']].to_string(index=False))
    
    n_lads_stars = lads_df['StarFile'].nunique()
    generate_latex_table_robust(lads_summary, 'LADS', n_lads_stars)
    
    # ----- Summary of all outliers -----
    print("\n" + "="*80)
    print("COMPLETE LIST OF ALL OUTLIERS (for reference)")
    print("="*80)
    
    all_outliers = pd.concat([
        hads_outliers.assign(StarType='HADS'),
        lads_outliers.assign(StarType='LADS')
    ])
    
    if len(all_outliers) > 0:
        print(f"\nTotal: {len(all_outliers)} outliers detected")
        print("\nOutliers sorted by R² (worst first):")
        all_outliers_sorted = all_outliers.sort_values('R2')[['StarType', 'StarFile', 'Method', 'Fold', 'R2', 'MAE']]
        print(all_outliers_sorted.to_string())
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR PAPER:")
    print("="*80)
    print("""
1. Use MEDIAN ± IQR instead of Mean ± Std for all summary tables
2. Mention in methodology: "Robust statistics (median, IQR) are used to 
   mitigate the effect of occasional optimization failures"
3. Optionally report the number of outliers per method in supplementary material
""")


if __name__ == "__main__":
    main()
