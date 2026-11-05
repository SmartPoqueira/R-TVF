
import pandas as pd
import json
import numpy as np

def parse_params(params_str):
    try:
        clean = str(params_str).replace('\n', '').replace('\r', '')
        clean = clean.replace('""', '"').replace("'", '"')
        if clean.startswith('"') and clean.endswith('"'):
            clean = clean[1:-1]
        return json.loads(clean)
    except Exception as e:
        return {}

csv_path = 'results/1_grid_search/grid_search_hads.csv'
df = pd.read_csv(csv_path)

# Filter only R-TVF
df_rtvf = df[df['Method'] == 'R-TVF'].copy()
print(f"Total R-TVF rows: {len(df_rtvf)}")

# Exact logic from sensitivity_analysis.py
params_df = df_rtvf['Params'].apply(parse_params).apply(pd.Series)
print(f"Params_df shape: {params_df.shape}")
print(f"Params_df columns: {params_df.columns.tolist()}")

# Check NaN fourier_K
if 'fourier_K' in params_df:
    nan_k = params_df[params_df['fourier_K'].isna()]
    print(f"Rows with NaN fourier_K: {len(nan_k)}")
    if len(nan_k) > 0:
        print("Sample NaN row index in params_df:", nan_k.index[0])
        # Find corresponding original param string
        # Since indices match (before reset_index in concat)
        original_idx = nan_k.index[0]
        print(f"Original Params string: {df_rtvf.loc[original_idx, 'Params']}")
        print(f"Parsed result: {parse_params(df_rtvf.loc[original_idx, 'Params'])}")

# Concat logic
df_merged = pd.concat([df_rtvf.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)

# Check l2=0 in merged
merged_zeros = df_merged[df_merged['l2'] == 0.0]
print(f"Merged zeros len: {len(merged_zeros)}")
if len(merged_zeros) == 0:
    print("Zeroes lost in merge or type mismatch.")
    print("Unique l2 values:", df_merged['l2'].unique())

