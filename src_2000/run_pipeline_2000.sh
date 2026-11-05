#!/bin/bash
cd "$(dirname "$0")"

echo "========================================="
echo "1. Grid Search"
echo "========================================="
python3 1.grid_search.py

echo "========================================="
echo "2. Cross Validation"
echo "========================================="
python3 2.cross_validation.py

echo "========================================="
echo "3. Sensitivity Analysis"
echo "========================================="
python3 3.sensitivity_analysis.py

echo "========================================="
echo "4. Analysis & Plots"
echo "========================================="
python3 4.analysis_and_plots.py

echo "========================================="
echo "7. Generate Tables"
echo "========================================="
python3 7.generate_tables.py

echo "========================================="
echo "5. Generate Grid Plots"
echo "========================================="
python3 5.generate_grid_plots.py
python3 5.generate_grid_plots_large.py

echo "========================================="
echo "PIPELINE N=2000 COMPLETED"
echo "========================================="
