#!/bin/bash
cd "$(dirname "$0")"

echo "========================================="
echo "1. Grid Search (N=1000)"
echo "========================================="
python3 1.grid_search.py

echo "========================================="
echo "2. Cross Validation (N=1000)"
echo "========================================="
python3 2.cross_validation.py

echo "========================================="
echo "3. Sensitivity Analysis (N=1000)"
echo "========================================="
python3 3.sensitivity_analysis.py

echo "========================================="
echo "4. Analysis & Plots (N=1000)"
echo "========================================="
python3 4.analysis_and_plots.py

echo "========================================="
echo "7. Generate Tables (N=1000)"
echo "========================================="
python3 7.generate_tables.py

echo "========================================="
echo "5. Generate Grid Plots (N=1000)"
echo "========================================="
python3 5.generate_grid_plots.py
python3 5.generate_grid_plots_large.py

echo "========================================="
echo "8. Generate R-TVF Grid Plots (N=1000)"
echo "========================================="
python3 8.generate_rtvf_grid.py
python3 8.generate_rtvf_grid_large.py

echo "========================================="
echo "PIPELINE N=1000 COMPLETED"
echo "========================================="
