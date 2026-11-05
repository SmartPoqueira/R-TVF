# Frequency-Domain Validation: Unimputed vs. R-TVF vs. Ground Truth

This self-contained subfolder contains the frequency-domain asteroseismic validation for **both experimental gap sizes: $N=1000$ (~33.5 h) and $N=2000$ (~67.5 h)**, addressing **Reviewer 3 (Point 3)** and **Reviewer 2 (Weakness 1 & Question 2)** for manuscript `KNOSYS-D-26-06907R1` (*Knowledge-Based Systems*).

## 1. Benchmark Scenario A: $N=1000$ Cadences (~33.5 Hours / 1.4 Days)

| Star | Type | Holdout Span | Gap (Hours) | Time $R^2$ | Spectral Corr [UNIMPUTED] | Spectral Corr [R-TVF] | Spectral MSE [UNIMPUTED] | Spectral MSE [R-TVF] | **Spectral MSE Reduction (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HD 146953** | HADS | [500 : 1500] | 33.8 h | 0.990 | 0.99739 | **0.99996** | 3.64e-06 | **5.21e-08** | **+98.57%** |
| **AD CMi** | HADS | [500 : 1500] | 33.4 h | 0.998 | 0.99870 | **1.00000** | 2.85e-06 | **1.44e-09** | **+99.95%** |
| **BE Lyn** | HADS | [500 : 1500] | 33.8 h | 0.979 | 0.99851 | **1.00000** | 2.34e-06 | **3.89e-09** | **+99.83%** |
| **BS Aqr** | HADS | [500 : 1500] | 33.6 h | 0.999 | 0.99894 | **1.00000** | 1.72e-06 | **1.00e-09** | **+99.94%** |
| **VZ Cnc** | HADS | [500 : 1500] | 33.9 h | 0.931 | 0.99861 | **0.99999** | 1.54e-06 | **1.14e-08** | **+99.26%** |
| **TIC 452982723** | LADS | [500 : 1500] | 33.9 h | 0.817 | 0.99845 | **0.99989** | 2.83e-07 | **2.17e-08** | **+92.35%** |
| **TIC 100531058** | LADS | [500 : 1500] | 33.6 h | 0.954 | 0.99593 | **0.99992** | 7.75e-07 | **1.67e-08** | **+97.84%** |
| **TIC 121729614** | LADS | [500 : 1500] | 33.5 h | 0.655 | 0.99884 | **0.99993** | 1.92e-07 | **2.24e-08** | **+88.32%** |
| **TIC 242944780** | LADS | [500 : 1500] | 33.6 h | 0.769 | 0.99840 | **0.99963** | 1.04e-07 | **1.98e-08** | **+80.89%** |
| **TIC 246902545** | LADS | [500 : 1500] | 33.6 h | 0.923 | 0.99656 | **0.99984** | 3.58e-07 | **1.69e-08** | **+95.27%** |
| **HADS Mean** | HADS | - | 33.6 h | 0.979 | 0.99843 | **0.99999** | - | - | **+99.51%** |
| **LADS Mean** | LADS | - | 33.5 h | 0.824 | 0.99764 | **0.99984** | - | - | **+90.94%** |
| **Overall Mean** | All | - | 33.5 h | **0.902** | **0.99803** | **0.99992** | - | - | **+95.22%** |

## 2. Benchmark Scenario B: $N=2000$ Cadences (~67.5 Hours / 2.8 Days)

| Star | Type | Holdout Span | Gap (Hours) | Time $R^2$ | Spectral Corr [UNIMPUTED] | Spectral Corr [R-TVF] | Spectral MSE [UNIMPUTED] | Spectral MSE [R-TVF] | **Spectral MSE Reduction (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HD 146953** | HADS | [500 : 2500] | 68.4 h | 0.811 | 0.99060 | **0.99833** | 1.36e-05 | **2.13e-06** | **+84.36%** |
| **AD CMi** | HADS | [500 : 2500] | 66.7 h | 0.957 | 0.99513 | **0.99966** | 1.11e-05 | **6.14e-07** | **+94.48%** |
| **BE Lyn** | HADS | [2050 : 4050] | 66.9 h | 0.979 | 0.99511 | **0.99998** | 6.83e-06 | **2.49e-08** | **+99.64%** |
| **BS Aqr** | HADS | [500 : 2500] | 67.6 h | 0.995 | 0.99595 | **0.99998** | 6.86e-06 | **2.06e-08** | **+99.70%** |
| **VZ Cnc** | HADS | [12900 : 14900] | 66.7 h | 0.909 | 0.99556 | **0.99972** | 4.59e-06 | **3.82e-07** | **+91.67%** |
| **TIC 452982723** | LADS | [12900 : 14900] | 66.7 h | 0.790 | 0.99556 | **0.99874** | 7.14e-07 | **1.84e-07** | **+74.22%** |
| **TIC 100531058** | LADS | [13100 : 15100] | 66.7 h | 0.904 | 0.99465 | **0.99852** | 1.10e-06 | **3.34e-07** | **+69.49%** |
| **TIC 121729614** | LADS | [500 : 2500] | 66.9 h | 0.607 | 0.99617 | **0.99935** | 7.10e-07 | **1.27e-07** | **+82.07%** |
| **TIC 242944780** | LADS | [500 : 2500] | 69.3 h | 0.774 | 0.99485 | **0.99937** | 3.48e-07 | **3.82e-08** | **+89.04%** |
| **TIC 246902545** | LADS | [500 : 2500] | 69.1 h | 0.870 | 0.98892 | **0.99698** | 1.23e-06 | **2.92e-07** | **+76.21%** |
| **HADS Mean** | HADS | - | 67.4 h | 0.930 | 0.99447 | **0.99954** | - | - | **+93.97%** |
| **LADS Mean** | LADS | - | 67.2 h | 0.789 | 0.99403 | **0.99859** | - | - | **+78.21%** |
| **Overall Mean** | All | - | 67.3 h | **0.860** | **0.99425** | **0.99907** | - | - | **+86.09%** |

## 3. Comparative Observations

1. **Consistent Frequency Preservation Across Gap Lengths:** When the gap length doubles from $N=1000$ (~1.4 days) to $N=2000$ (~2.8 days), the unimputed light curve suffers stronger window sidelobes and power degradation. R-TVF suppresses these artifacts in both regimes, achieving **+95.23%** mean spectral MSE reduction at $N=1000$ and **+86.09%** at $N=2000$.
2. **Asteroseismic Mode Line Profiles:** The dominant oscillation modes ($f_1$) preserve their exact frequencies and line profiles in both scenarios without spurious aliases.

## 4. Plot Directory Structure

- **Scenario A ($N=1000$) Plots**: [`plots/gap_1000/`](plots/gap_1000/)
- **Scenario B ($N=2000$) Plots**: [`plots/gap_2000/`](plots/gap_2000/)
