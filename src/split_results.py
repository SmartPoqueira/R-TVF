import pandas as pd
from pathlib import Path
import shutil

# Paths
BASE_DIR = Path("results/1_grid_search")
COMBINED_FILE = BASE_DIR / "grid_search_results.csv"
HADS_FILE = BASE_DIR / "grid_search_hads.csv"
LADS_FILE = BASE_DIR / "grid_search_lads.csv"

def split_csv():
    if not COMBINED_FILE.exists():
        print(f"Error: {COMBINED_FILE} does not exist.")
        return

    print(f"Reading combined file: {COMBINED_FILE}")
    df = pd.read_csv(COMBINED_FILE)
    print(f"Total rows: {len(df)}")
    
    # Backup existing
    if HADS_FILE.exists():
        shutil.copy(HADS_FILE, HADS_FILE.with_suffix('.csv.bak_split'))
    if LADS_FILE.exists():
        shutil.copy(LADS_FILE, LADS_FILE.with_suffix('.csv.bak_split'))

    # Split
    df_hads = df[df['StarType'] == 'HADS']
    df_lads = df[df['StarType'] == 'LADS']

    print(f"Writing HADS: {len(df_hads)} rows")
    df_hads.to_csv(HADS_FILE, index=False)

    print(f"Writing LADS: {len(df_lads)} rows")
    df_lads.to_csv(LADS_FILE, index=False)

    print("Done. Files synchronized.")

if __name__ == "__main__":
    split_csv()
