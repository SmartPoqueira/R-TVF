# Manual de uso de R-TVF para una serie temporal individual

Este documento describe el procedimiento computacional para ejecutar el método R-TVF sobre una curva de luz estelar aislada.

## 1. Fundamento matemático y formulación del método

El método R-TVF reconstruye segmentos temporales no observados modelando el flujo estelar mediante una superposición armónica con modulación polinómica de amplitud y fase:

$$\hat{y}(t) = \mu(t) + \sum_{k=1}^{K} \left[ \alpha_k(t) \cos(2\pi f_k t) + \beta_k(t) \sin(2\pi f_k t) \right]$$

Las frecuencias $f_1, \ldots, f_K$ se determinan a partir de los máximos locales del periodograma de Lomb-Scargle calculado sobre las observaciones disponibles. La tendencia local $\mu(t)$ y las funciones moduladoras $\alpha_k(t)$ y $\beta_k(t)$ se expresan mediante polinomios de grado $d$:

$$\mu(t) = \sum_{j=0}^{d} w_{0,j} t^j, \quad \alpha_k(t) = \sum_{j=0}^{d} w_{k,j}^{(\alpha)} t^j, \quad \beta_k(t) = \sum_{j=0}^{d} w_{k,j}^{(\beta)} t^j$$

El vector de parámetros $\mathbf{W} \in \mathbb{R}^{(2K+1)(d+1)}$ se estima resolviendo el problema regularizado de Tikhonov con penalización $\ell_2$:

$$\mathbf{W}^* = \left( \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I} \right)^{-1} \mathbf{X}^\top \mathbf{y}$$

La varianza analítica predictiva en el instante $t$ se obtiene mediante la expresión:

$$\sigma_{\text{pred}}^2(t) = \sigma_{\text{ruido}}^2 + \mathbf{x}(t)^\top \left( \mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I} \right)^{-1} \mathbf{x}(t)$$

## 2. Requisitos previos

El método requiere Python 3.10 o superior y las bibliotecas especificadas en requirements.txt:

```bash
pip install -r requirements.txt
```

## 3. Estructura de los datos de entrada

El archivo de entrada debe consistir en un archivo de texto con extensión .dat sin cabeceras o con comentarios precedidos por #, estructurado en dos columnas delimitadas por espacios o tabulaciones:
* Columna 1: Tiempo de observación (días, BJD - 2457000).
* Columna 2: Flujo fotométrico observado (e-/s).

Los archivos de las 72 estrellas del estudio se localizan en los directorios db/hads/ y db/lads/.

## 4. Parámetros operativos del método

| Parámetro | Símbolo | Tipo | Valor por defecto | Descripción funcional |
| :--- | :--- | :--- | :--- | :--- |
| `--star` | - | String | Obligatorio | Ruta al archivo fotométrico .dat de la estrella |
| `--K` | $K$ | Entero | 10 | Número de frecuencias dominantes extraídas del periodograma |
| `--degree` | $d$ | Entero | 1 | Grado del polinomio modulador |
| `--l2` | $\lambda$ | Float | 0.1 | Coeficiente de regularización de Tikhonov |
| `--gap-size` | $N$ | Entero | 1000 | Número de puntos contiguos en el intervalo a imputar |
| `--gap-start` | - | Entero | None | Índice inicial del intervalo; si se omite, se selecciona aleatoriamente |
| `--fmin` | $f_{\min}$ | Float | 5.0 | Límite inferior de frecuencia espectral (ciclos/día) |
| `--fmax` | $f_{\max}$ | Float | 80.0 | Límite superior de frecuencia espectral (ciclos/día) |
| `--output-plot` | - | String | `single_star_reconstruction.png` | Ruta de destino para la figura de reconstrucción |
| `--output-csv` | - | String | None | Ruta opcional para exportar la serie imputada en formato CSV |

## 5. Ejecución mediante interfaz de línea de comandos

Para procesar una estrella de alta amplitud (HADS) con un intervalo ausente de 1000 puntos:

```bash
python src/run_single_star.py \
    --star "db/hads/HD 146953lc.dat" \
    --gap-size 1000 \
    --K 10 \
    --degree 1 \
    --l2 0.1 \
    --output-plot "reconstruccion_HD146953.png"
```

Para procesar una estrella de baja amplitud (LADS) exportando la serie temporal reconstruida a CSV:

```bash
python src/run_single_star.py \
    --star "db/lads/TIC 452982723lc.dat" \
    --gap-size 1000 \
    --K 10 \
    --degree 1 \
    --l2 0.1 \
    --output-plot "reconstruccion_TIC452982723.png" \
    --output-csv "imputacion_TIC452982723.csv"
```

Para fijar un intervalo específico de observación entre los índices 2000 y 3000:

```bash
python src/run_single_star.py \
    --star "db/hads/BL Camlc.dat" \
    --gap-start 2000 \
    --gap-size 1000 \
    --output-plot "reconstruccion_BL_Cam.png"
```

## 6. Integración programática en Python

El método se puede importar directamente en scripts externos sin invocar la interfaz de comandos:

```python
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Inclusión del directorio del método
sys.path.append(str(Path("src").resolve()))
import modelos_hads_lads as mh
import utils

# 1. Carga de datos fotométricos
t, y_observado = utils.load_data("db/hads/HD 146953lc.dat")

# 2. Definición del intervalo a reconstruir
inicio = 1200
fin = 2200
y_incompleto = y_observado.copy()
y_incompleto[inicio:fin] = np.nan

# 3. Ajuste de R-TVF y reconstrucción analítica
y_imputado, frecuencias, desv_est = mh.tvar_fourier_impute_block_multi(
    t=t,
    y_missing=y_incompleto,
    start=inicio,
    end=fin,
    M=10,
    degree=1,
    trend_deg=1,
    l2=0.1,
    fmin=5.0,
    fmax=80.0,
    return_std=True
)

# 4. Cálculo de métricas sobre el intervalo imputado
y_real = y_observado[inicio:fin]
y_pred = y_imputado[inicio:fin]

mae = np.mean(np.abs(y_real - y_pred))
mse = np.mean((y_real - y_pred) ** 2)
mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100.0
r2 = 1.0 - (np.sum((y_real - y_pred) ** 2) / np.sum((y_real - np.mean(y_real)) ** 2))

print(f"Frecuencias dominantes identificadas: {len(frecuencias)}")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
```

## 7. Estructura de la salida generada

1. Consola: Reporte cuantitativo con el tamaño de la serie, rango temporal en días, frecuencias identificadas, error absoluto medio (MAE), error cuadrático medio (MSE), error porcentual absoluto medio (MAPE) y coeficiente de determinación ($R^2$).
2. Gráfico: Imagen en formato PNG que contiene dos paneles sincronizados:
   * Panel superior: Curva de luz completa mostrando las observaciones y la región reconstruida.
   * Panel inferior: Ampliación sobre el intervalo del hueco, con los datos reales de control, la predicción de R-TVF y la banda de confianza analítica al 95% calculada a partir de $\pm 1.96 \sigma_{\text{pred}}(t)$.
3. Archivo CSV (opcional): Tabla con columnas `time`, `flux_original`, `flux_imputed`, `pred_std` e `is_imputed`.
