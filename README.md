# R-TVF: A Regularized Time-Varying Fourier Method for Missing Data Recovery

Official repository for the paper "R-TVF: A Regularized Time-Varying Fourier method for missing data recovery", published in *Knowledge-Based Systems* (Elsevier, 2026).

## Method Description

R-TVF reconstructs missing intervals in photometric time series by modeling stellar flux as a superposition of harmonic oscillations with time-varying amplitudes:

$$\hat{y}(t) = \mu(t) + \sum_{k=1}^{K} \left[ \alpha_k(t) \cos(2\pi f_k t) + \beta_k(t) \sin(2\pi f_k t) \right]$$

The model extracts dominant frequencies $f_1, \ldots, f_K$ via the Lomb-Scargle periodogram. The terms $\mu(t)$, $\alpha_k(t)$, and $\beta_k(t)$ represent polynomial expansions of degree $d$:

$$\mu(t) = \sum_{j=0}^{d} w_{0,j} t^j, \quad \alpha_k(t) = \sum_{j=0}^{d} w_{k,j}^{(\alpha)} t^j, \quad \beta_k(t) = \sum_{j=0}^{d} w_{k,j}^{(\beta)} t^j$$

Parameter vector $\mathbf{W} \in \mathbb{R}^P$ contains $P = (2K + 1)(d + 1)$ coefficients. The method minimizes a squared error objective with an $\ell_2$ penalty:

$$J(\mathbf{W}) = \sum_{i=1}^{M} \left( y_i - \hat{y}(t_i) \right)^2 + \lambda \|\mathbf{W}\|_2^2$$

The solution is computed in closed form via ridge regression:

$$\mathbf{W}^* = \left( \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I} \right)^{-1} \mathbf{X}^\top \mathbf{y}$$

Analytical predictive variance at timestamp $t$ is calculated as:

$$\sigma_{\text{pred}}^2(t) = \sigma_{\text{noise}}^2 + \mathbf{x}(t)^\top \left( \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I} \right)^{-1} \mathbf{x}(t)$$

## Repository Structure

```text
.
├── db/                           # Photometric light curve datasets (72 Delta Scuti stars)
│   ├── hads/                     # 33 High-Amplitude Delta Scuti stars (.dat files)
│   └── lads/                     # 39 Low-Amplitude Delta Scuti stars (.dat files)
├── estrellas_test                # Symbolic link to db/ for script compatibility
├── src/                          # Execution scripts for contiguous gaps of N = 1000 points
│   ├── 1.grid_search.py          # Hyperparameter grid search
│   ├── 2.cross_validation.py     # 5-fold cross-validation execution
│   ├── 3.sensitivity_analysis.py # Sensitivity evaluation over K, d, and lambda
│   ├── 4.analysis_and_plots.py   # Statistical tests and visualization generation
│   ├── 5.generate_grid_plots.py  # Temporal reconstruction plots
│   ├── 6.analyze_outliers.py     # Outlier and residual analysis
│   ├── 7.generate_tables.py      # LaTeX and CSV summary table compilation
│   ├── 8.generate_rtvf_grid.py   # Consolidated R-TVF grid figures
│   ├── run_single_star.py        # Standalone imputation script for individual stars
│   ├── run_pipeline.sh           # Automated pipeline execution script
│   ├── lightCurve.py             # Data loader and benchmark baselines
│   └── modelos_hads_lads.py      # Implementation of R-TVF and competing methods
├── src_2000/                     # Execution scripts for contiguous gaps of N = 2000 points
│   └── run_pipeline_2000.sh      # Automated pipeline execution script for N = 2000
├── results/                      # Generated tables, metrics, and plots for N = 1000 points
│   ├── 1_grid_search/            # Optimal hyperparameter configurations
│   ├── 2_cross_validation/       # Fold-level prediction outputs and CSV records
│   ├── 3_sensitivity/            # Sensitivity analysis tables and heatmaps
│   └── 4_analysis/               # Summary CSV files, statistical tests, and figures
├── results_2000/                 # Generated tables, metrics, and plots for N = 2000 points
├── frequency_domain_validation/  # Lomb-Scargle periodogram spectral verification
├── paper/                        # Published manuscript sources, BibTeX, and figures
├── requirements.txt              # Python package dependencies
├── LICENSE                       # MIT License
└── README.md                     # Technical documentation
```

## Requirements and Installation

The codebase requires Python 3.10 or higher. Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Reproducibility and Execution

### 1. Single Star Imputation

To execute R-TVF on an individual photometric time series (.dat file):

```bash
cd src

# Run R-TVF on a single light curve with default parameters (K=10, d=1, lambda=0.1)
python run_single_star.py --star "../db/hads/HD 146953lc.dat" --gap-size 1000 --output-plot reconstruction.png

# Optional export of reconstructed series to CSV
python run_single_star.py --star "../db/lads/TIC 452982723lc.dat" --gap-size 1000 --output-plot reconstruction.png --output-csv imputed_series.csv
```

Parameters:
* `--star`: Path to input photometric file.
* `--gap-start`: Start index of the gap (integer). When omitted, index is selected randomly.
* `--gap-size`: Number of contiguous missing observations (default: 1000).
* `--K`: Number of dominant Fourier frequencies (default: 10).
* `--degree`: Polynomial modulation degree d (default: 1).
* `--l2`: Tikhonov regularization parameter lambda (default: 0.1).
* `--output-plot`: Destination filepath for the reconstruction figure.
* `--output-csv`: Destination filepath for the reconstructed numerical time series.

### 2. Gap Scenario N = 1000 points

Navigate to the `src` directory to execute each phase of the evaluation pipeline:

```bash
cd src

# Sequential execution:
python 1.grid_search.py
python 2.cross_validation.py
python 3.sensitivity_analysis.py
python 4.analysis_and_plots.py
python 7.generate_tables.py
python 5.generate_grid_plots.py
python 8.generate_rtvf_grid.py

# Alternatively, execute the full pipeline script:
bash run_pipeline.sh
```

### 3. Gap Scenario N = 2000 points

Navigate to the `src_2000` directory to execute the evaluation for extended gaps:

```bash
cd src_2000

# Sequential execution:
python 1.grid_search.py
python 2.cross_validation.py
python 3.sensitivity_analysis.py
python 4.analysis_and_plots.py
python 7.generate_tables.py
python 5.generate_grid_plots.py
python 8.generate_rtvf_grid.py

# Alternatively, execute the full pipeline script:
bash run_pipeline_2000.sh
```

### 4. Frequency-Domain Validation

Navigate to the `frequency_domain_validation` directory to evaluate spectral recovery:

```bash
cd frequency_domain_validation
python validate_frequency_comparison.py
```

## Empirical Results

### Results for Contiguous Gaps of N = 1000 points

Table 1: 5-fold cross-validation metrics for HADS stars ($N = 1000$). Values report mean $\pm$ standard deviation.

| Method | MAE | MSE ($\times 10^6$) | MAPE (%) | $R^2$ | Total Time (s) | LS / Fit Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ARMAX | 1386 $\pm$ 3467 | 20.9 $\pm$ 100.9 | 151.3 $\pm$ 97.3 | 0.947 $\pm$ 0.045 | 56.2 $\pm$ 33.9 | 0.04 / 56.16 |
| GP-Celerite | 3511 $\pm$ 7870 | 113.4 $\pm$ 475.4 | 485.4 $\pm$ 368.3 | 0.375 $\pm$ 0.617 | 0.52 $\pm$ 0.12 | 0.04 / 0.48 |
| DHR | 4510 $\pm$ 2145 | 145.2 $\pm$ 82.5 | 450.5 $\pm$ 215.3 | 0.206 $\pm$ 0.229 | 15.3 $\pm$ 4.2 | 0.04 / 15.26 |
| TBATS | 3150 $\pm$ 1500 | 95.6 $\pm$ 40.2 | 380.4 $\pm$ 120.5 | 0.452 $\pm$ 0.154 | 18.5 $\pm$ 5.1 | 0.05 / 18.45 |
| FS-BSM | 1693 $\pm$ 1250 | 59.4 $\pm$ 45.2 | 169.9 $\pm$ 60.5 | 0.928 $\pm$ 0.062 | 24.8 $\pm$ 8.5 | 0.05 / 24.75 |
| **R-TVF (ours)** | **1429 $\pm$ 6459** | **60.2 $\pm$ 348.8** | **56.3 $\pm$ 58.2** | **0.987 $\pm$ 0.045** | **10.6 $\pm$ 3.5** | **0.06 / 10.54** |

Table 2: 5-fold cross-validation metrics for LADS stars ($N = 1000$). Values report mean $\pm$ standard deviation.

| Method | MAE | MSE ($\times 10^3$) | MAPE (%) | $R^2$ | Total Time (s) | LS / Fit Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ARMAX | 191 $\pm$ 279 | 173.8 $\pm$ 533.8 | 17.3 $\pm$ 18.6 | 0.791 $\pm$ 0.110 | 77.7 $\pm$ 44.5 | 0.04 / 77.66 |
| GP-Celerite | 647 $\pm$ 942 | 1856.5 $\pm$ 5851.6 | 64.5 $\pm$ 84.3 | -2.873 $\pm$ 6.647 | 0.59 $\pm$ 0.13 | 0.03 / 0.56 |
| DHR | 420 $\pm$ 250 | 950.4 $\pm$ 450.2 | 35.4 $\pm$ 25.1 | 0.158 $\pm$ 0.420 | 17.4 $\pm$ 5.2 | 0.04 / 17.36 |
| TBATS | 435 $\pm$ 260 | 985.6 $\pm$ 480.5 | 38.2 $\pm$ 28.4 | 0.152 $\pm$ 0.198 | 19.1 $\pm$ 6.3 | 0.04 / 19.06 |
| FS-BSM | 220 $\pm$ 150 | 280.5 $\pm$ 180.2 | 20.0 $\pm$ 15.5 | 0.639 $\pm$ 0.190 | 25.2 $\pm$ 9.1 | 0.06 / 25.14 |
| **R-TVF (ours)** | **170 $\pm$ 249** | **139.1 $\pm$ 428.7** | **13.9 $\pm$ 11.2** | **0.816 $\pm$ 0.128** | **9.6 $\pm$ 1.6** | **0.04 / 9.56** |

### Results for Contiguous Gaps of N = 2000 points

Table 3: 5-fold cross-validation metrics for HADS stars ($N = 2000$). Values report mean $\pm$ standard deviation.

| Method | MAE | MSE ($\times 10^6$) | MAPE (%) | $R^2$ | Total Time (s) | LS / Fit Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ARMAX | 1997 $\pm$ 5217 | 45.3 $\pm$ 227.7 | 216.1 $\pm$ 139.5 | 0.891 $\pm$ 0.119 | 47.7 $\pm$ 29.1 | 0.05 / 47.65 |
| GP-Celerite | 11314 $\pm$ 45551 | 2746.3 $\pm$ 16195.9 | 717.7 $\pm$ 296.5 | -0.061 $\pm$ 0.356 | 0.71 $\pm$ 0.16 | 0.03 / 0.68 |
| DHR | 7510 $\pm$ 12145 | 215.2 $\pm$ 682.5 | 550.5 $\pm$ 285.3 | 0.185 $\pm$ 0.254 | 28.3 $\pm$ 6.2 | 0.05 / 28.25 |
| TBATS | 5150 $\pm$ 8500 | 132.6 $\pm$ 410.2 | 410.4 $\pm$ 220.5 | 0.448 $\pm$ 0.162 | 32.5 $\pm$ 8.1 | 0.04 / 32.46 |
| FS-BSM | 2050 $\pm$ 4800 | 68.5 $\pm$ 250.2 | 230.4 $\pm$ 140.5 | 0.885 $\pm$ 0.125 | 45.8 $\pm$ 12.5 | 0.05 / 45.75 |
| **R-TVF (ours)** | **1893 $\pm$ 7713** | **96.9 $\pm$ 559.1** | **98.6 $\pm$ 86.8** | **0.968 $\pm$ 0.051** | **9.2 $\pm$ 3.0** | **0.06 / 9.14** |

Table 4: 5-fold cross-validation metrics for LADS stars ($N = 2000$). Values report mean $\pm$ standard deviation.

| Method | MAE | MSE ($\times 10^3$) | MAPE (%) | $R^2$ | Total Time (s) | LS / Fit Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ARMAX | 243 $\pm$ 451 | 384.6 $\pm$ 1710.3 | 21.0 $\pm$ 25.6 | 0.738 $\pm$ 0.173 | 72.6 $\pm$ 46.9 | 0.05 / 72.55 |
| GP-Celerite | 622 $\pm$ 1032 | 2125.1 $\pm$ 5941.5 | 62.5 $\pm$ 90.3 | -0.163 $\pm$ 0.391 | 0.73 $\pm$ 0.16 | 0.02 / 0.71 |
| DHR | 510 $\pm$ 850 | 1450.4 $\pm$ 4450.2 | 48.4 $\pm$ 45.1 | 0.098 $\pm$ 0.380 | 31.4 $\pm$ 7.2 | 0.04 / 31.36 |
| TBATS | 485 $\pm$ 660 | 1285.6 $\pm$ 3480.5 | 45.2 $\pm$ 35.4 | 0.145 $\pm$ 0.210 | 35.1 $\pm$ 9.3 | 0.05 / 35.05 |
| FS-BSM | 265 $\pm$ 420 | 410.5 $\pm$ 1850.2 | 25.0 $\pm$ 22.5 | 0.725 $\pm$ 0.180 | 48.2 $\pm$ 14.1 | 0.06 / 48.14 |
| **R-TVF (ours)** | **217 $\pm$ 380** | **287.6 $\pm$ 1192.2** | **16.8 $\pm$ 13.4** | **0.761 $\pm$ 0.161** | **9.8 $\pm$ 3.3** | **0.06 / 9.72** |

## Citation

Please cite this publication if you use this code in your research:

```bibtex
@article{duran2026r,
  title={R-TVF: A Regularized Time-Varying Fourier method for missing data recovery},
  author={Dur{\'a}n-L{\'o}pez, Alberto and Bola{\~n}os-Martinez, Daniel and El Ouahabi, Zakaria Sahraoui and Hern{\'a}ndez, Antonio Garc{\'\i}a and Su{\'a}rez, Juan Carlos and Bermudez-Edo, Maria},
  journal={Knowledge-Based Systems},
  volume={352},
  pages={117145},
  year={2026},
  publisher={Elsevier}
}
```

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for conditions.
