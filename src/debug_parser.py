
import pandas as pd
import json
from pathlib import Path

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
df = df[df['Method'] == 'R-TVF'].copy()

print(f"Total R-TVF: {len(df)}")

# Apply parsing
parsed_series = df['Params'].apply(parse_params)
params_df = parsed_series.apply(pd.Series)

# Check l2=0.0
params_df['l2'] = pd.to_numeric(params_df['l2'], errors='coerce')
zeros = params_df[params_df['l2'] == 0.0]
print(f"Rows with l2=0.0: {len(zeros)}")

if len(zeros) > 0:
    print("Sample l2=0.0 row:")
    print(zeros.iloc[0])
    
    # Check corresponding Params string
    idx = zeros.index[0]
    print(f"Original String: {df.loc[idx, 'Params']}")
else:
    print("No l2=0.0 rows found in parsed data.")
    # Debug raw strings containing 0.0
    raw_zeros = df[df['Params'].str.contains('0.0')]
    print(f"Raw strings with 0.0: {len(raw_zeros)}")
    if len(raw_zeros) > 0:
        s = raw_zeros.iloc[0]['Params']
        print(f"Sample raw string: {s}")
        print(f"Parsed result: {parse_params(s)}")
