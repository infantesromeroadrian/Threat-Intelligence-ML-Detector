# Introduccion a Series Temporales

## Que es una Serie Temporal

Una **serie temporal** es una secuencia de observaciones ordenadas en el tiempo, donde el **orden importa**. A diferencia de datos tabulares clasicos, las observaciones no son independientes: cada valor depende potencialmente de valores anteriores.

```
SERIE TEMPORAL
==============

Valor
  │
  │       ●
  │    ●     ●
  │  ●         ●    ●
  │●             ●●   ●
  │                     ●
  └────────────────────────── Tiempo
   t₁  t₂  t₃  t₄  t₅  t₆ ...

Caracteristicas:
- Ordenadas temporalmente (t₁ < t₂ < t₃ ...)
- Equiespaciadas (generalmente): hora, dia, mes
- Dependencia temporal: y_t depende de y_{t-1}, y_{t-2}, ...
```

## Ejemplos en Ciberseguridad

| Dominio | Serie Temporal | Frecuencia Tipica |
|---------|----------------|-------------------|
| Red | Bytes/seg, paquetes/min | Segundos-minutos |
| Logs | Eventos por minuto | Minutos |
| Autenticacion | Logins fallidos/hora | Horas |
| Malware | Detecciones diarias | Dias |
| Vulnerabilidades | CVEs por mes | Meses |
| Incidentes | Alertas SIEM | Segundos-minutos |

---

## Componentes de una Serie Temporal

### Descomposicion Clasica

```
COMPONENTES
===========

Serie Original = Tendencia + Estacionalidad + Residuo

                Y_t = T_t + S_t + R_t  (aditivo)
                Y_t = T_t × S_t × R_t  (multiplicativo)


1. TENDENCIA (Trend) - T_t
   Movimiento a largo plazo

   │    Serie Original          │     Tendencia
   │       ╱╲    ╱╲             │         ___----
   │     ╱    ╲╱    ╲           │    ___--
   │   ╱            ╲          │___--
   │ ╱                         │
   └───────────────────        └───────────────────


2. ESTACIONALIDAD (Seasonality) - S_t
   Patron que se repite en periodos fijos

   │                            │  ╱╲  ╱╲  ╱╲  ╱╲
   │                            │ ╱  ╲╱  ╲╱  ╲╱  ╲
   │                            │╱                ╲
   └───────────────────        └───────────────────
                                periodo  periodo


3. RESIDUO (Residual/Noise) - R_t
   Variacion aleatoria no explicada

   │      ·  ·                  │    ·     ·
   │  ·        ·   ·            │  ·   · ·   ·
   │    ·   ·     ·             │·       ·     ·
   └───────────────────        └───────────────────
```

### Ejemplo Practico

```
TRAFICO DE RED - DESCOMPOSICION
===============================

Serie Original (requests/hora):
│    __    __    __    __
│   /  \  /  \  /  \  /  \
│  /    \/    \/    \/    \
│_/                        \___
└──────────────────────────────
  Lun   Mar   Mie   Jue   Vie


Tendencia: Crecimiento gradual del uso
│                          ___
│                    ___---
│              ___---
│        ___---
│  ___---
└──────────────────────────────


Estacionalidad: Patron diario (pico laboral)
│  __    __    __    __    __
│ /  \  /  \  /  \  /  \  /  \
│/    \/    \/    \/    \/    \
└──────────────────────────────
  9am  5pm  9am  5pm  9am  5pm


Residuo: Variaciones no explicadas
│    ·  ·     ·
│  ·      ·  ·  ·   ·
│·    ·        ·  ·
└──────────────────────────────
```

---

## Estacionariedad

### Concepto Fundamental

```
ESTACIONARIEDAD
===============

Una serie es ESTACIONARIA si sus propiedades estadisticas
NO cambian con el tiempo.

Estacionaria:                    No Estacionaria:
│                                │              ___
│  ~~∿~~∿~~∿~~∿~~               │         ___/
│                                │    ___/
│  Media constante               │___/
│  Varianza constante            │
└─────────────────────           └─────────────────────
                                  (tendencia)

│                                │           ╱╲
│  ~~∿~~∿~~∿~~∿~~               │      ╱╲  ╱  ╲
│                                │  ╱╲╱  ╲╱    ╲
│  Varianza constante            │╱
└─────────────────────           └─────────────────────
                                  (varianza creciente)
```

### Tipos de Estacionariedad

| Tipo | Condicion | Uso |
|------|-----------|-----|
| Estricta | Distribucion completa invariante | Teorico |
| Debil | Media y varianza constantes, covarianza solo depende del lag | Practico |
| Tendencia-estacionaria | Estacionaria tras eliminar tendencia | Comun |
| Diferencia-estacionaria | Estacionaria tras diferenciar | Muy comun |

### Tests de Estacionariedad

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Tuple


def test_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Realiza tests de estacionariedad.

    Tests:
    - ADF (Augmented Dickey-Fuller): H0 = no estacionaria
    - KPSS: H0 = estacionaria

    Args:
        series: Serie temporal a testear
        alpha: Nivel de significancia

    Returns:
        Diccionario con resultados de tests
    """
    results = {}

    # ADF Test
    # H0: Serie tiene raiz unitaria (no estacionaria)
    # Rechazar H0 (p < alpha) -> estacionaria
    adf_result = adfuller(series.dropna(), autolag='AIC')
    results['adf'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < alpha
    }

    # KPSS Test
    # H0: Serie es estacionaria
    # Rechazar H0 (p < alpha) -> NO estacionaria
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    results['kpss'] = {
        'statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'critical_values': kpss_result[3],
        'is_stationary': kpss_result[1] >= alpha
    }

    # Conclusion combinada
    results['conclusion'] = (
        'Estacionaria'
        if results['adf']['is_stationary'] and results['kpss']['is_stationary']
        else 'No estacionaria'
    )

    return results


def make_stationary(
    series: pd.Series,
    max_diff: int = 2
) -> Tuple[pd.Series, int]:
    """
    Transforma serie a estacionaria mediante diferenciacion.

    Args:
        series: Serie original
        max_diff: Maximo numero de diferenciaciones

    Returns:
        Serie estacionaria y orden de diferenciacion
    """
    diff_series = series.copy()
    d = 0

    for i in range(max_diff + 1):
        result = test_stationarity(diff_series)

        if result['conclusion'] == 'Estacionaria':
            print(f"Serie estacionaria con d={d}")
            return diff_series, d

        if i < max_diff:
            diff_series = diff_series.diff().dropna()
            d += 1
            print(f"Diferenciando (d={d})...")

    print(f"Warning: Serie no estacionaria tras {max_diff} diferenciaciones")
    return diff_series, d


# Ejemplo de uso
if __name__ == "__main__":
    # Serie de ejemplo: trafico de red con tendencia
    np.random.seed(42)
    n = 200
    trend = np.linspace(100, 200, n)
    noise = np.random.randn(n) * 10
    series = pd.Series(trend + noise)

    # Test
    print("Serie original:")
    result = test_stationarity(series)
    print(f"  ADF p-value: {result['adf']['p_value']:.4f}")
    print(f"  KPSS p-value: {result['kpss']['p_value']:.4f}")
    print(f"  Conclusion: {result['conclusion']}")

    # Hacer estacionaria
    print("\nTransformando...")
    stationary_series, d = make_stationary(series)
```

---

## Autocorrelacion

### ACF y PACF

```
AUTOCORRELACION (ACF)
=====================

Correlacion de la serie consigo misma en diferentes lags.

ACF(k) = Corr(Y_t, Y_{t-k})

    │ ACF
  1 │█
    │█
0.5 │██
    │███
  0 │████████████████
    │        ████
-0.5│
    └─────────────────
     0  1  2  3  4  5 ...  Lag

- Lag 0: siempre = 1 (correlacion consigo misma)
- Decaimiento gradual: indica AR
- Corte abrupto en lag k: indica MA(k)


AUTOCORRELACION PARCIAL (PACF)
==============================

Correlacion directa entre Y_t e Y_{t-k}, eliminando
efectos de lags intermedios.

PACF(k) = Corr(Y_t, Y_{t-k} | Y_{t-1}, ..., Y_{t-k+1})

    │ PACF
  1 │█
    │█
0.5 │█
    │
  0 │  ████████████████
    │
-0.5│
    └─────────────────
     0  1  2  3  4  5 ...  Lag

- Corte abrupto en lag p: indica AR(p)
- Decaimiento gradual: indica MA
```

### Interpretacion para Identificacion de Modelos

```
REGLAS DE IDENTIFICACION
========================

Patron ACF             Patron PACF           Modelo Sugerido
─────────────────────────────────────────────────────────────
Decae exponencial      Corte en lag p        AR(p)
Corte en lag q         Decae exponencial     MA(q)
Decae exponencial      Decae exponencial     ARMA(p,q)
No decae               -                     No estacionaria
Picos en lags s,2s,3s  -                     Estacionalidad s


Ejemplos visuales:

AR(1):                      MA(1):
ACF        PACF             ACF        PACF
█          █                █          █
██                          ██         ██
███                                    ███
████                                   ████
█████                                  █████
(decae)    (corte)          (corte)    (decae)


AR(2):                      ARMA(1,1):
ACF        PACF             ACF        PACF
█          █                █          █
██         █                ██         ██
███                         ███        ███
████                        ████       ████
(decae)    (corte p=2)      (ambos decaen)
```

### Implementacion

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def analyze_autocorrelation(
    series: pd.Series,
    lags: int = 40,
    alpha: float = 0.05
) -> dict:
    """
    Analiza autocorrelacion de la serie.

    Args:
        series: Serie temporal
        lags: Numero de lags a analizar
        alpha: Nivel de significancia para intervalos

    Returns:
        Diccionario con ACF, PACF y lags significativos
    """
    series_clean = series.dropna()

    # Calcular ACF y PACF
    acf_values, acf_confint = acf(
        series_clean, nlags=lags, alpha=alpha, fft=True
    )
    pacf_values, pacf_confint = pacf(
        series_clean, nlags=lags, alpha=alpha, method='ywm'
    )

    # Intervalo de confianza (aproximado)
    conf_bound = 1.96 / np.sqrt(len(series_clean))

    # Encontrar lags significativos
    acf_significant = np.where(np.abs(acf_values[1:]) > conf_bound)[0] + 1
    pacf_significant = np.where(np.abs(pacf_values[1:]) > conf_bound)[0] + 1

    return {
        'acf': acf_values,
        'pacf': pacf_values,
        'acf_confint': acf_confint,
        'pacf_confint': pacf_confint,
        'conf_bound': conf_bound,
        'acf_significant_lags': acf_significant,
        'pacf_significant_lags': pacf_significant
    }


def plot_acf_pacf(series: pd.Series, lags: int = 40) -> None:
    """Grafica ACF y PACF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)')

    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF)')

    plt.tight_layout()
    plt.show()


def suggest_arima_order(
    series: pd.Series,
    max_p: int = 5,
    max_q: int = 5
) -> dict:
    """
    Sugiere orden ARIMA basado en ACF/PACF.

    Returns:
        Sugerencia de (p, d, q)
    """
    # Primero hacer estacionaria
    stationary, d = make_stationary(series)

    # Analizar autocorrelacion
    analysis = analyze_autocorrelation(stationary)

    # Sugerir p basado en PACF
    pacf_sig = analysis['pacf_significant_lags']
    if len(pacf_sig) > 0:
        # Buscar corte
        p = 0
        for i, lag in enumerate(pacf_sig):
            if i == 0 or lag == pacf_sig[i-1] + 1:
                p = lag
            else:
                break
    else:
        p = 0

    # Sugerir q basado en ACF
    acf_sig = analysis['acf_significant_lags']
    if len(acf_sig) > 0:
        q = 0
        for i, lag in enumerate(acf_sig):
            if i == 0 or lag == acf_sig[i-1] + 1:
                q = lag
            else:
                break
    else:
        q = 0

    # Limitar
    p = min(p, max_p)
    q = min(q, max_q)

    return {
        'suggested_order': (p, d, q),
        'p': p,
        'd': d,
        'q': q,
        'acf_significant': acf_sig.tolist(),
        'pacf_significant': pacf_sig.tolist()
    }
```

---

## Operaciones Basicas

### Diferenciacion

```
DIFERENCIACION
==============

Elimina tendencia para hacer la serie estacionaria.

Primera diferencia (d=1):
    Y'_t = Y_t - Y_{t-1}

Segunda diferencia (d=2):
    Y''_t = Y'_t - Y'_{t-1} = Y_t - 2Y_{t-1} + Y_{t-2}

Diferencia estacional (periodo s):
    Y^s_t = Y_t - Y_{t-s}


Ejemplo (tendencia lineal):

Original:                    Primera diferencia:
│         ___               │
│     ___/                  │  ~~~~~~~~~~~~~~~
│ ___/                      │
│/                          │
└───────────────            └───────────────
10, 12, 14, 16, 18          2, 2, 2, 2
(tendencia)                 (estacionaria!)
```

### Transformaciones

```python
import numpy as np
import pandas as pd


def difference(series: pd.Series, periods: int = 1) -> pd.Series:
    """Diferenciacion simple."""
    return series.diff(periods).dropna()


def seasonal_difference(series: pd.Series, period: int = 12) -> pd.Series:
    """Diferenciacion estacional."""
    return series.diff(period).dropna()


def log_transform(series: pd.Series) -> pd.Series:
    """
    Transformacion logaritmica.
    Estabiliza varianza cuando crece con el nivel.
    """
    # Asegurar valores positivos
    min_val = series.min()
    if min_val <= 0:
        series = series - min_val + 1
    return np.log(series)


def box_cox_transform(
    series: pd.Series,
    lmbda: float | None = None
) -> Tuple[pd.Series, float]:
    """
    Transformacion Box-Cox.
    Encuentra lambda optimo si no se especifica.

    Box-Cox(y, λ):
        (y^λ - 1) / λ  si λ ≠ 0
        log(y)         si λ = 0
    """
    from scipy.stats import boxcox

    # Asegurar positivos
    series_pos = series.copy()
    min_val = series_pos.min()
    if min_val <= 0:
        series_pos = series_pos - min_val + 1

    if lmbda is None:
        transformed, fitted_lambda = boxcox(series_pos)
        return pd.Series(transformed, index=series.index), fitted_lambda
    else:
        if lmbda == 0:
            return np.log(series_pos), 0.0
        else:
            return (series_pos ** lmbda - 1) / lmbda, lmbda


def inverse_box_cox(
    transformed: pd.Series,
    lmbda: float,
    shift: float = 0
) -> pd.Series:
    """Inversa de Box-Cox."""
    if lmbda == 0:
        original = np.exp(transformed)
    else:
        original = (transformed * lmbda + 1) ** (1 / lmbda)
    return original + shift
```

---

## Ventanas y Resampling

### Ventanas Moviles

```
VENTANAS MOVILES (Rolling Windows)
==================================

Media movil suaviza la serie eliminando ruido.

Serie original:
    10, 15, 12, 18, 14, 20, 16, 22, 18, 25

Media movil (ventana=3):
    -, -, 12.3, 15.0, 14.7, 17.3, 16.7, 19.3, 18.7, 21.7

         │ Original        │ Media movil (3)
         │    ●  ●  ●      │
         │ ●  ●     ●  ●   │    ___________
         │    ●        ●   │___/
         │●                │
         └─────────────    └─────────────

Tipos:
- Simple Moving Average (SMA): todos los pesos iguales
- Exponential Moving Average (EMA): pesos decrecientes
- Weighted Moving Average (WMA): pesos personalizados
```

### Implementacion

```python
import pandas as pd
import numpy as np


def rolling_statistics(
    series: pd.Series,
    window: int = 7,
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Calcula estadisticas con ventana movil.

    Args:
        series: Serie temporal
        window: Tamano de ventana
        min_periods: Minimo de observaciones necesarias

    Returns:
        DataFrame con estadisticas rolling
    """
    rolling = series.rolling(window=window, min_periods=min_periods)

    return pd.DataFrame({
        'original': series,
        'rolling_mean': rolling.mean(),
        'rolling_std': rolling.std(),
        'rolling_min': rolling.min(),
        'rolling_max': rolling.max(),
        'rolling_median': rolling.median()
    })


def exponential_moving_average(
    series: pd.Series,
    span: int = 7,
    adjust: bool = True
) -> pd.Series:
    """
    Media movil exponencial (EMA).

    EMA_t = α * Y_t + (1 - α) * EMA_{t-1}
    donde α = 2 / (span + 1)
    """
    return series.ewm(span=span, adjust=adjust).mean()


def resample_series(
    series: pd.Series,
    rule: str = 'H',
    agg_func: str = 'mean'
) -> pd.Series:
    """
    Cambia la frecuencia de la serie.

    Args:
        series: Serie con indice datetime
        rule: Nueva frecuencia ('H'=hora, 'D'=dia, 'W'=semana, 'M'=mes)
        agg_func: Funcion de agregacion ('mean', 'sum', 'max', 'min', 'last')

    Returns:
        Serie resampleada
    """
    resampler = series.resample(rule)

    if agg_func == 'mean':
        return resampler.mean()
    elif agg_func == 'sum':
        return resampler.sum()
    elif agg_func == 'max':
        return resampler.max()
    elif agg_func == 'min':
        return resampler.min()
    elif agg_func == 'last':
        return resampler.last()
    else:
        raise ValueError(f"Funcion no soportada: {agg_func}")


# Ejemplo: Agregar logs por hora
def aggregate_logs_hourly(log_timestamps: pd.Series) -> pd.Series:
    """
    Cuenta eventos de log por hora.

    Args:
        log_timestamps: Serie de timestamps de eventos

    Returns:
        Serie con conteo por hora
    """
    # Crear serie con 1 por cada evento
    events = pd.Series(1, index=pd.to_datetime(log_timestamps))

    # Agregar por hora
    hourly_counts = events.resample('H').sum().fillna(0)

    return hourly_counts
```

---

## Train/Test Split para Series Temporales

### El Problema del Data Leakage

```
SPLIT INCORRECTO (DATA LEAKAGE!)
================================

NUNCA hacer random split en series temporales:

    │●  ●  ●  ●  ●  ●  ●  ●  ●  ●│
    │Train Train Test Train Test │
    │  ●      ●      ●      ●    │  <- Mezcla temporal!
    └────────────────────────────┘

El modelo "ve el futuro" durante entrenamiento!


SPLIT CORRECTO (temporal)
=========================

    │●  ●  ●  ●  ●  ●│●  ●  ●  ●│
    │     TRAIN      │   TEST   │
    └────────────────┴──────────┘
           80%           20%

    Todo el test es POSTERIOR al train.
```

### Implementacion

```python
import pandas as pd
import numpy as np
from typing import Tuple, Generator


def temporal_train_test_split(
    series: pd.Series,
    test_size: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    """
    Split temporal para series temporales.

    Args:
        series: Serie temporal ordenada
        test_size: Proporcion para test (al final)

    Returns:
        (train, test) series
    """
    n = len(series)
    split_idx = int(n * (1 - test_size))

    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    return train, test


def expanding_window_split(
    series: pd.Series,
    initial_train_size: int,
    step: int = 1
) -> Generator[Tuple[pd.Series, pd.Series], None, None]:
    """
    Split con ventana expandible (walk-forward validation).

    ┌──────────────────────────────────┐
    │ Fold 1: [Train    ] [Test]       │
    │ Fold 2: [Train       ] [Test]    │
    │ Fold 3: [Train          ] [Test] │
    └──────────────────────────────────┘

    Args:
        series: Serie temporal
        initial_train_size: Tamano inicial de entrenamiento
        step: Cuantos puntos avanzar por fold

    Yields:
        (train, test) tuples
    """
    n = len(series)

    for end_train in range(initial_train_size, n, step):
        train = series.iloc[:end_train]
        test = series.iloc[end_train:end_train + step]

        if len(test) > 0:
            yield train, test


def sliding_window_split(
    series: pd.Series,
    train_size: int,
    test_size: int = 1,
    step: int = 1
) -> Generator[Tuple[pd.Series, pd.Series], None, None]:
    """
    Split con ventana deslizante (tamano fijo de train).

    ┌──────────────────────────────────┐
    │ Fold 1: [Train    ] [Test]       │
    │ Fold 2:   [Train    ] [Test]     │
    │ Fold 3:     [Train    ] [Test]   │
    └──────────────────────────────────┘

    Args:
        series: Serie temporal
        train_size: Tamano fijo de ventana de entrenamiento
        test_size: Tamano de test
        step: Cuantos puntos deslizar por fold

    Yields:
        (train, test) tuples
    """
    n = len(series)

    for start in range(0, n - train_size - test_size + 1, step):
        train = series.iloc[start:start + train_size]
        test = series.iloc[start + train_size:start + train_size + test_size]

        yield train, test


class TimeSeriesCrossValidator:
    """
    Cross-validation para series temporales.
    Compatible con scikit-learn.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int | None = None,
        gap: int = 0
    ):
        """
        Args:
            n_splits: Numero de folds
            test_size: Tamano de cada test set
            gap: Espacio entre train y test (evita leakage)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Genera indices de train/test."""
        n_samples = len(X)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        # Calcular splits
        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
```

---

## Metricas de Evaluacion

### Metricas Comunes

```
METRICAS PARA SERIES TEMPORALES
===============================

1. MAE (Mean Absolute Error)
   MAE = (1/n) Σ |y_i - ŷ_i|

   Interpretable, mismas unidades que la serie.

2. MSE (Mean Squared Error)
   MSE = (1/n) Σ (y_i - ŷ_i)²

   Penaliza mas los errores grandes.

3. RMSE (Root Mean Squared Error)
   RMSE = √MSE

   Mismas unidades que la serie, penaliza outliers.

4. MAPE (Mean Absolute Percentage Error)
   MAPE = (100/n) Σ |y_i - ŷ_i| / |y_i|

   Porcentual, problematico si y_i ≈ 0.

5. SMAPE (Symmetric MAPE)
   SMAPE = (100/n) Σ |y_i - ŷ_i| / ((|y_i| + |ŷ_i|) / 2)

   Simetrico, mas robusto.

6. MASE (Mean Absolute Scaled Error)
   MASE = MAE / MAE_naive

   < 1: mejor que naive forecast
   > 1: peor que naive forecast
```

### Implementacion

```python
import numpy as np
from typing import Union


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    # Evitar division por cero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Evitar division por cero
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Mean Absolute Scaled Error.

    Compara con naive forecast (seasonal si seasonality > 1).

    Args:
        y_true: Valores reales
        y_pred: Predicciones
        y_train: Serie de entrenamiento (para calcular escala)
        seasonality: Periodo estacional (1 = naive simple)

    Returns:
        MASE score (< 1 es mejor que naive)
    """
    # MAE del modelo
    mae_model = mae(y_true, y_pred)

    # MAE del naive forecast en training
    if seasonality == 1:
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])

    mae_naive = np.mean(naive_errors)

    return mae_model / mae_naive


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 1
) -> dict:
    """
    Evalua forecast con multiples metricas.

    Returns:
        Diccionario con todas las metricas
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred)
    }

    if y_train is not None:
        metrics['MASE'] = mase(y_true, y_pred, y_train, seasonality)

    return metrics
```

---

## Aplicacion: Analisis de Trafico de Red

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def analyze_network_traffic(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    bytes_col: str = 'bytes',
    freq: str = '1H'
) -> dict:
    """
    Analiza serie temporal de trafico de red.

    Args:
        df: DataFrame con datos de trafico
        timestamp_col: Columna de timestamp
        bytes_col: Columna de bytes
        freq: Frecuencia de agregacion

    Returns:
        Diccionario con analisis
    """
    # Preparar serie temporal
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)

    # Agregar por frecuencia
    traffic = df[bytes_col].resample(freq).sum()

    # Analisis basico
    analysis = {
        'series': traffic,
        'statistics': {
            'mean': traffic.mean(),
            'std': traffic.std(),
            'min': traffic.min(),
            'max': traffic.max(),
            'median': traffic.median()
        }
    }

    # Test de estacionariedad
    analysis['stationarity'] = test_stationarity(traffic)

    # Autocorrelacion
    analysis['autocorrelation'] = analyze_autocorrelation(traffic, lags=48)

    # Detectar estacionalidad
    if len(traffic) >= 168:  # Al menos 1 semana de datos horarios
        # Correlacion con lag de 24 (patron diario)
        daily_corr = traffic.autocorr(lag=24)
        # Correlacion con lag de 168 (patron semanal)
        weekly_corr = traffic.autocorr(lag=168) if len(traffic) >= 336 else None

        analysis['seasonality'] = {
            'daily_correlation': daily_corr,
            'weekly_correlation': weekly_corr,
            'has_daily_pattern': abs(daily_corr) > 0.3,
            'has_weekly_pattern': weekly_corr and abs(weekly_corr) > 0.3
        }

    return analysis


def detect_traffic_anomalies_zscore(
    traffic: pd.Series,
    window: int = 24,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detecta anomalias usando Z-score con ventana movil.

    Args:
        traffic: Serie de trafico
        window: Ventana para calcular media/std
        threshold: Umbral de Z-score

    Returns:
        DataFrame con anomalias
    """
    # Rolling statistics
    rolling_mean = traffic.rolling(window=window, center=True).mean()
    rolling_std = traffic.rolling(window=window, center=True).std()

    # Z-score
    z_score = (traffic - rolling_mean) / rolling_std

    # Detectar anomalias
    anomalies = pd.DataFrame({
        'value': traffic,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'z_score': z_score,
        'is_anomaly': abs(z_score) > threshold,
        'anomaly_type': np.where(
            z_score > threshold, 'spike',
            np.where(z_score < -threshold, 'drop', 'normal')
        )
    })

    return anomalies


# Ejemplo de uso
if __name__ == "__main__":
    # Simular datos de trafico
    np.random.seed(42)

    # 30 dias de datos horarios
    dates = pd.date_range('2024-01-01', periods=24*30, freq='H')

    # Patron: base + diario + ruido + anomalias
    base = 1000
    daily_pattern = 500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)  # Ciclo 24h
    noise = np.random.randn(len(dates)) * 100

    traffic = pd.Series(base + daily_pattern + noise, index=dates)

    # Inyectar anomalias
    traffic.iloc[100] = 5000  # Spike
    traffic.iloc[200] = 100   # Drop

    # Analizar
    analysis = analyze_network_traffic(
        pd.DataFrame({'timestamp': dates, 'bytes': traffic.values}),
        freq='1H'
    )

    print("=== Analisis de Trafico ===")
    print(f"Media: {analysis['statistics']['mean']:.2f}")
    print(f"Estacionario: {analysis['stationarity']['conclusion']}")

    # Detectar anomalias
    anomalies = detect_traffic_anomalies_zscore(traffic)
    n_anomalies = anomalies['is_anomaly'].sum()
    print(f"Anomalias detectadas: {n_anomalies}")
```

---

## Resumen

| Concepto | Descripcion | Importancia |
|----------|-------------|-------------|
| Estacionariedad | Propiedades constantes en tiempo | Requisito para muchos modelos |
| ACF/PACF | Autocorrelacion | Identificacion de modelos |
| Diferenciacion | Eliminar tendencia | Hacer estacionaria |
| Descomposicion | T + S + R | Entender componentes |
| Split temporal | Train antes, test despues | Evitar data leakage |
| MASE | Comparacion con naive | Metrica robusta |

### Puntos Clave

1. **Orden temporal** es fundamental - nunca shuffle
2. **Estacionariedad** es requisito para ARIMA
3. **ACF/PACF** ayudan a identificar el modelo
4. **Validacion** siempre hacia adelante en el tiempo
5. **MASE < 1** significa que superamos al baseline naive
6. En ciberseguridad: trafico, logs, metricas son series temporales
