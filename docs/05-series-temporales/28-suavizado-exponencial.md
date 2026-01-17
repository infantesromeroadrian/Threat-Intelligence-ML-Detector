# Suavizado Exponencial (Exponential Smoothing)

## Introduccion

El **suavizado exponencial** es una familia de metodos de forecasting que asignan pesos exponencialmente decrecientes a observaciones pasadas. Son intuitivos, computacionalmente eficientes y sorprendentemente efectivos.

```
IDEA CENTRAL
============

Prediccion = Promedio ponderado donde:
- Observaciones recientes tienen MAS peso
- Observaciones antiguas tienen MENOS peso
- Los pesos decaen exponencialmente

Pesos:
t   : α
t-1 : α(1-α)
t-2 : α(1-α)²
t-3 : α(1-α)³
...

    Peso
      │█
      │██
      │███
      │████
      │█████
      │██████
      └──────────────
       t  t-1 t-2 t-3 ...

α cercano a 1: Mas peso a observaciones recientes (reactivo)
α cercano a 0: Mas peso distribuido (suave)
```

## Metodos de Suavizado Exponencial

### Taxonomia

```
FAMILIA ETS (Error, Trend, Seasonality)
=======================================

                    Tendencia
                    │
         ┌──────────┼──────────┐
         │          │          │
        None      Aditiva    Damped
         │          │          │
         │    ┌─────┴─────┐    │
         │    │           │    │
    ┌────┴────┴───┐   ┌───┴────┴────┐
    │             │   │             │
  Simple        Holt  │           Holt
 Exponential         │          Damped
 Smoothing           │
    │                │
    ▼                ▼
    │    ┌───────────┴───────────┐
    │    │                       │
   Sin   │                       │
Estacion.│                       │
    │    │                       │
    │  Aditiva              Multiplicativa
    │    │                       │
    ▼    ▼                       ▼
   SES  Holt-Winters          Holt-Winters
        Aditivo               Multiplicativo
```

---

## Simple Exponential Smoothing (SES)

### Concepto

```
SES: Para series sin tendencia ni estacionalidad
===================================================

Ecuacion:
    ŷ_{t+1} = αy_t + (1-α)ŷ_t

Equivalentemente:
    ŷ_{t+1} = ŷ_t + α(y_t - ŷ_t)
              ↑          ↑
         prediccion   error
         anterior

α ∈ [0, 1] es el parametro de suavizado


Forma expandida:
    ŷ_{t+1} = α[y_t + (1-α)y_{t-1} + (1-α)²y_{t-2} + ...]

    Suma infinita de pesos que decaen exponencialmente!
```

### Efecto de α

```
EFECTO DEL PARAMETRO α
======================

α = 0.1 (suave):           α = 0.9 (reactivo):
│    ●                     │    ●
│   ●  ●                   │  ● ●
│  ●    ●                  │ ●   ●●
│ ●      ────────          │●      ●──
│●     (linea suave)       │       (sigue datos)
└──────────────            └──────────────

α pequeño:                 α grande:
- Suaviza mas              - Reacciona rapido
- Ignora ruido             - Sigue cambios
- Lento ante cambios       - Sensible a ruido


Regla practica:
- Series ruidosas: α bajo (0.1-0.3)
- Series estables: α medio (0.3-0.5)
- Cambios rapidos: α alto (0.5-0.9)
```

### Implementacion SES

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from typing import Tuple


class SimpleExponentialSmoothing:
    """
    Simple Exponential Smoothing implementado desde cero.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Parametro de suavizado (0-1)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha debe estar entre 0 y 1")
        self.alpha = alpha
        self.fitted_values: np.ndarray | None = None
        self.last_smoothed: float | None = None

    def fit(self, y: np.ndarray) -> 'SimpleExponentialSmoothing':
        """
        Ajusta el modelo.

        Args:
            y: Serie temporal

        Returns:
            self
        """
        n = len(y)
        smoothed = np.zeros(n)

        # Inicializar con primer valor
        smoothed[0] = y[0]

        # Aplicar suavizado
        for t in range(1, n):
            smoothed[t] = self.alpha * y[t] + (1 - self.alpha) * smoothed[t-1]

        self.fitted_values = smoothed
        self.last_smoothed = smoothed[-1]

        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Genera predicciones.
        SES predice el mismo valor para todos los pasos futuros.
        """
        if self.last_smoothed is None:
            raise ValueError("Modelo no ajustado")

        return np.full(steps, self.last_smoothed)

    def fit_predict(
        self,
        y: np.ndarray,
        steps: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ajusta y predice."""
        self.fit(y)
        return self.fitted_values, self.predict(steps)


def optimize_alpha(
    y: np.ndarray,
    metric: str = 'mse'
) -> Tuple[float, float]:
    """
    Encuentra alpha optimo por grid search.

    Args:
        y: Serie temporal
        metric: 'mse' o 'mae'

    Returns:
        (mejor_alpha, mejor_score)
    """
    best_alpha = 0.0
    best_score = np.inf

    for alpha in np.arange(0.01, 1.0, 0.01):
        model = SimpleExponentialSmoothing(alpha=alpha)
        fitted, _ = model.fit_predict(y, steps=1)

        # Error en muestra (one-step-ahead)
        errors = y[1:] - fitted[:-1]

        if metric == 'mse':
            score = np.mean(errors ** 2)
        else:  # mae
            score = np.mean(np.abs(errors))

        if score < best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha, best_score


# Usando statsmodels
def ses_statsmodels(
    series: pd.Series,
    alpha: float | None = None,
    forecast_steps: int = 10
) -> dict:
    """
    SES usando statsmodels.

    Args:
        series: Serie temporal
        alpha: Parametro de suavizado (None = optimizar)
        forecast_steps: Pasos a predecir

    Returns:
        Diccionario con resultados
    """
    if alpha is not None:
        model = SimpleExpSmoothing(series)
        fitted = model.fit(smoothing_level=alpha, optimized=False)
    else:
        model = SimpleExpSmoothing(series)
        fitted = model.fit(optimized=True)

    forecast = fitted.forecast(forecast_steps)

    return {
        'model': fitted,
        'alpha': fitted.params['smoothing_level'],
        'fitted_values': fitted.fittedvalues,
        'forecast': forecast,
        'sse': fitted.sse,
        'aic': fitted.aic
    }
```

---

## Holt's Linear Trend Method

### Concepto

```
HOLT: Para series con TENDENCIA (sin estacionalidad)
====================================================

Dos ecuaciones de suavizado:

1. Nivel (level):
   l_t = αy_t + (1-α)(l_{t-1} + b_{t-1})

2. Tendencia (trend):
   b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}

Prediccion h pasos adelante:
   ŷ_{t+h} = l_t + h·b_t


Parametros:
- α: suavizado del nivel (0-1)
- β: suavizado de la tendencia (0-1)


Visualizacion:
                            ŷ_{t+h} = l_t + h·b_t
                               ╱
                              ╱  (tendencia b_t)
                             ╱
                    ●───────●
                   ╱
                  ╱
    ─────────────●
               l_t
```

### Holt Damped Trend

```
HOLT DAMPED: Tendencia amortiguada
==================================

Problema con Holt lineal: tendencia infinita

    │              ╱
    │            ╱
    │          ╱       ← Holt lineal
    │        ╱            (poco realista)
    │      ╱
    │    ●
    │   ●
    │  ●
    │ ●
    └──────────────

Solucion: Amortiguar la tendencia con φ (phi)

Prediccion:
   ŷ_{t+h} = l_t + (φ + φ² + ... + φʰ)·b_t

Si φ < 1: tendencia se amortigua
Si φ = 1: Holt lineal original

    │           ______
    │         ╱       ← Damped
    │       ╱           (mas realista)
    │     ╱
    │   ●
    │  ●
    │ ●
    └──────────────

φ tipico: 0.8 - 0.98
```

### Implementacion Holt

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt


def holt_linear(
    series: pd.Series,
    alpha: float | None = None,
    beta: float | None = None,
    damped: bool = False,
    phi: float | None = None,
    forecast_steps: int = 10
) -> dict:
    """
    Metodo de Holt (tendencia lineal o amortiguada).

    Args:
        series: Serie temporal
        alpha: Suavizado nivel
        beta: Suavizado tendencia
        damped: Usar tendencia amortiguada
        phi: Factor de amortiguamiento (si damped=True)
        forecast_steps: Pasos a predecir

    Returns:
        Diccionario con resultados
    """
    model = Holt(series, damped_trend=damped)

    # Parametros
    fit_params = {}
    if alpha is not None:
        fit_params['smoothing_level'] = alpha
    if beta is not None:
        fit_params['smoothing_trend'] = beta
    if phi is not None and damped:
        fit_params['damping_trend'] = phi

    if fit_params:
        fitted = model.fit(**fit_params, optimized=False)
    else:
        fitted = model.fit(optimized=True)

    forecast = fitted.forecast(forecast_steps)

    return {
        'model': fitted,
        'alpha': fitted.params.get('smoothing_level'),
        'beta': fitted.params.get('smoothing_trend'),
        'phi': fitted.params.get('damping_trend') if damped else None,
        'fitted_values': fitted.fittedvalues,
        'forecast': forecast,
        'sse': fitted.sse,
        'aic': fitted.aic
    }


def compare_holt_models(
    series: pd.Series,
    test_size: int = 10
) -> pd.DataFrame:
    """
    Compara Holt lineal vs damped.
    """
    train = series[:-test_size]
    test = series[-test_size:]

    results = []

    # Holt lineal
    holt_lin = holt_linear(train, damped=False, forecast_steps=test_size)
    mse_lin = np.mean((test.values - holt_lin['forecast'].values) ** 2)
    results.append({
        'method': 'Holt Linear',
        'alpha': holt_lin['alpha'],
        'beta': holt_lin['beta'],
        'phi': None,
        'mse': mse_lin,
        'aic': holt_lin['aic']
    })

    # Holt damped
    holt_dmp = holt_linear(train, damped=True, forecast_steps=test_size)
    mse_dmp = np.mean((test.values - holt_dmp['forecast'].values) ** 2)
    results.append({
        'method': 'Holt Damped',
        'alpha': holt_dmp['alpha'],
        'beta': holt_dmp['beta'],
        'phi': holt_dmp['phi'],
        'mse': mse_dmp,
        'aic': holt_dmp['aic']
    })

    return pd.DataFrame(results)
```

---

## Holt-Winters (Triple Exponential Smoothing)

### Concepto

```
HOLT-WINTERS: Nivel + Tendencia + Estacionalidad
================================================

Tres componentes:

1. Nivel:
   l_t = α(y_t - s_{t-m}) + (1-α)(l_{t-1} + b_{t-1})    [Aditivo]
   l_t = α(y_t / s_{t-m}) + (1-α)(l_{t-1} + b_{t-1})    [Multiplicativo]

2. Tendencia:
   b_t = β(l_t - l_{t-1}) + (1-β)b_{t-1}

3. Estacionalidad:
   s_t = γ(y_t - l_t) + (1-γ)s_{t-m}                    [Aditivo]
   s_t = γ(y_t / l_t) + (1-γ)s_{t-m}                    [Multiplicativo]


Prediccion h pasos:
   ŷ_{t+h} = l_t + h·b_t + s_{t+h-m}                    [Aditivo]
   ŷ_{t+h} = (l_t + h·b_t) · s_{t+h-m}                  [Multiplicativo]


m = periodo estacional (12 mensual, 4 trimestral, 24 horario)
```

### Aditivo vs Multiplicativo

```
ADITIVO vs MULTIPLICATIVO
=========================

ADITIVO: Estacionalidad CONSTANTE en magnitud

    │       ╱╲       ╱╲
    │      ╱  ╲     ╱  ╲      Amplitud constante
    │     ╱    ╲   ╱    ╲
    │────╱──────╲─╱──────╲────
    │
    └─────────────────────────

    Y = Nivel + Tendencia + Estacionalidad
    Usar cuando variacion estacional es constante


MULTIPLICATIVO: Estacionalidad PROPORCIONAL al nivel

    │                    ╱╲
    │               ╱╲  ╱  ╲
    │          ╱╲  ╱  ╲╱    ╲
    │─────╱╲──╱──╲╱───────────
    │    ╱  ╲╱
    └─────────────────────────

    Y = Nivel × Tendencia × Estacionalidad
    Usar cuando variacion crece con el nivel


REGLA:
- Si el patron estacional tiene altura constante: ADITIVO
- Si el patron estacional crece/decrece: MULTIPLICATIVO
```

### Implementacion Holt-Winters

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Literal


def holt_winters(
    series: pd.Series,
    seasonal_periods: int,
    trend: Literal['add', 'mul', None] = 'add',
    seasonal: Literal['add', 'mul', None] = 'add',
    damped: bool = False,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    phi: float | None = None,
    forecast_steps: int = 10
) -> dict:
    """
    Holt-Winters (Triple Exponential Smoothing).

    Args:
        series: Serie temporal
        seasonal_periods: Periodo estacional (m)
        trend: 'add', 'mul', o None
        seasonal: 'add', 'mul', o None
        damped: Amortiguar tendencia
        alpha, beta, gamma, phi: Parametros (None = optimizar)
        forecast_steps: Pasos a predecir

    Returns:
        Diccionario con resultados
    """
    model = ExponentialSmoothing(
        series,
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped
    )

    # Parametros opcionales
    fit_params = {}
    if alpha is not None:
        fit_params['smoothing_level'] = alpha
    if beta is not None:
        fit_params['smoothing_trend'] = beta
    if gamma is not None:
        fit_params['smoothing_seasonal'] = gamma
    if phi is not None and damped:
        fit_params['damping_trend'] = phi

    if fit_params:
        fitted = model.fit(**fit_params, optimized=False)
    else:
        fitted = model.fit(optimized=True)

    forecast = fitted.forecast(forecast_steps)

    return {
        'model': fitted,
        'params': {
            'alpha': fitted.params.get('smoothing_level'),
            'beta': fitted.params.get('smoothing_trend'),
            'gamma': fitted.params.get('smoothing_seasonal'),
            'phi': fitted.params.get('damping_trend')
        },
        'seasonal_components': fitted.season,
        'fitted_values': fitted.fittedvalues,
        'forecast': forecast,
        'sse': fitted.sse,
        'aic': fitted.aic
    }


def auto_ets(
    series: pd.Series,
    seasonal_periods: int | None = None,
    forecast_steps: int = 10
) -> dict:
    """
    Seleccion automatica del mejor modelo ETS.

    Prueba todas las combinaciones de trend/seasonal.

    Args:
        series: Serie temporal
        seasonal_periods: Periodo estacional (None = detectar)
        forecast_steps: Pasos a predecir

    Returns:
        Mejor modelo encontrado
    """
    # Detectar estacionalidad si no se especifica
    if seasonal_periods is None:
        # Intentar detectar por ACF
        from statsmodels.tsa.stattools import acf
        acf_values = acf(series, nlags=min(len(series)//2, 50))

        # Buscar pico significativo
        peaks = []
        for i in range(2, len(acf_values)):
            if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1] if i+1 < len(acf_values) else True:
                if acf_values[i] > 0.1:  # Umbral minimo
                    peaks.append((i, acf_values[i]))

        if peaks:
            seasonal_periods = peaks[0][0]
            print(f"Periodo estacional detectado: {seasonal_periods}")
        else:
            seasonal_periods = None

    best_aic = np.inf
    best_model = None
    best_config = None

    # Configuraciones a probar
    trends = [None, 'add', 'mul']
    seasonals = [None, 'add', 'mul'] if seasonal_periods else [None]
    dampeds = [False, True]

    for trend in trends:
        for seasonal in seasonals:
            for damped in dampeds:
                # Solo damped si hay tendencia
                if damped and trend is None:
                    continue

                # Multiplicativo necesita valores positivos
                if (trend == 'mul' or seasonal == 'mul') and (series <= 0).any():
                    continue

                try:
                    model = ExponentialSmoothing(
                        series,
                        seasonal_periods=seasonal_periods,
                        trend=trend,
                        seasonal=seasonal,
                        damped_trend=damped
                    )
                    fitted = model.fit(optimized=True)

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_model = fitted
                        best_config = {
                            'trend': trend,
                            'seasonal': seasonal,
                            'damped': damped,
                            'seasonal_periods': seasonal_periods
                        }
                        print(f"  Nuevo mejor: {best_config}, AIC={best_aic:.2f}")

                except Exception:
                    continue

    if best_model is None:
        raise ValueError("No se pudo ajustar ningun modelo ETS")

    return {
        'model': best_model,
        'config': best_config,
        'aic': best_aic,
        'forecast': best_model.forecast(forecast_steps)
    }
```

---

## ETS Framework (Error-Trend-Seasonal)

### Notacion

```
ETS: FRAMEWORK UNIFICADO
========================

Notacion: ETS(E, T, S)

E = Error:      A (aditivo), M (multiplicativo)
T = Tendencia:  N (ninguna), A (aditiva), A_d (aditiva damped)
S = Estacion.:  N (ninguna), A (aditiva), M (multiplicativa)


Ejemplos:
─────────
ETS(A,N,N) = Simple Exponential Smoothing
ETS(A,A,N) = Holt lineal
ETS(A,A_d,N) = Holt damped
ETS(A,A,A) = Holt-Winters aditivo
ETS(A,A,M) = Holt-Winters multiplicativo
ETS(M,A,M) = Modelo con error multiplicativo


Total: 30 combinaciones posibles
(algunas inestables o raramente utiles)
```

### Tabla de Modelos ETS

| Modelo | Error | Trend | Seasonal | Equivalente |
|--------|-------|-------|----------|-------------|
| ETS(A,N,N) | A | N | N | SES |
| ETS(A,A,N) | A | A | N | Holt lineal |
| ETS(A,Ad,N) | A | Ad | N | Holt damped |
| ETS(A,N,A) | A | N | A | Estacional sin tendencia |
| ETS(A,A,A) | A | A | A | Holt-Winters aditivo |
| ETS(A,A,M) | A | A | M | Holt-Winters multiplicativo |
| ETS(M,A,M) | M | A | M | Error multiplicativo |

### Implementacion ETS Completa

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings


class ETSModel:
    """
    Framework ETS completo.
    """

    def __init__(
        self,
        error: str = 'add',
        trend: str | None = None,
        seasonal: str | None = None,
        damped: bool = False,
        seasonal_periods: int | None = None
    ):
        """
        Args:
            error: 'add' o 'mul'
            trend: 'add', 'mul', o None
            seasonal: 'add', 'mul', o None
            damped: Amortiguar tendencia
            seasonal_periods: Periodo estacional
        """
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.damped = damped
        self.seasonal_periods = seasonal_periods
        self.fitted = None
        self.model = None

    def fit(self, y: pd.Series, **kwargs) -> 'ETSModel':
        """Ajusta el modelo."""
        # statsmodels no soporta error multiplicativo directamente
        # en ExponentialSmoothing, usamos la aproximacion

        self.model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            damped_trend=self.damped,
            seasonal_periods=self.seasonal_periods
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted = self.model.fit(**kwargs)

        return self

    def predict(self, steps: int = 1) -> pd.Series:
        """Genera predicciones."""
        if self.fitted is None:
            raise ValueError("Modelo no ajustado")
        return self.fitted.forecast(steps)

    @property
    def aic(self) -> float:
        return self.fitted.aic if self.fitted else np.nan

    @property
    def bic(self) -> float:
        return self.fitted.bic if self.fitted else np.nan

    @property
    def params(self) -> dict:
        if self.fitted is None:
            return {}
        return {
            'alpha': self.fitted.params.get('smoothing_level'),
            'beta': self.fitted.params.get('smoothing_trend'),
            'gamma': self.fitted.params.get('smoothing_seasonal'),
            'phi': self.fitted.params.get('damping_trend')
        }

    def summary(self) -> str:
        """Resumen del modelo."""
        name = f"ETS({self.error[0].upper()},"
        name += f"{'N' if self.trend is None else self.trend[0].upper()}"
        name += f"{'d' if self.damped else ''},"
        name += f"{'N' if self.seasonal is None else self.seasonal[0].upper()})"

        if self.seasonal_periods:
            name += f"[{self.seasonal_periods}]"

        return name


def forecast_with_confidence(
    model,
    steps: int,
    alpha: float = 0.05,
    simulation_n: int = 1000
) -> pd.DataFrame:
    """
    Genera predicciones con intervalos de confianza via simulacion.

    Args:
        model: Modelo ETS ajustado
        steps: Pasos a predecir
        alpha: Nivel de significancia
        simulation_n: Numero de simulaciones

    Returns:
        DataFrame con forecast e intervalos
    """
    # Simulaciones
    simulations = model.fitted.simulate(
        steps,
        repetitions=simulation_n,
        error='mul' if 'mul' in str(model.model) else 'add'
    )

    # Calcular intervalos
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    forecast = model.predict(steps)
    lower = simulations.quantile(lower_q, axis=1)
    upper = simulations.quantile(upper_q, axis=1)

    return pd.DataFrame({
        'forecast': forecast,
        'lower': lower,
        'upper': upper
    })
```

---

## Aplicacion: Monitoreo de Metricas de Seguridad

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SecurityMetricsForecaster:
    """
    Forecaster para metricas de seguridad usando ETS.
    Aplicaciones: trafico, alertas, eventos por hora/dia.
    """

    def __init__(
        self,
        metric_name: str,
        seasonal_period: int = 24,  # Horario
        auto_select: bool = True
    ):
        self.metric_name = metric_name
        self.seasonal_period = seasonal_period
        self.auto_select = auto_select
        self.model = None
        self.best_config = None

    def fit(self, series: pd.Series) -> 'SecurityMetricsForecaster':
        """
        Ajusta modelo a la serie de metricas.
        """
        if self.auto_select:
            result = auto_ets(series, self.seasonal_period)
            self.model = result['model']
            self.best_config = result['config']
        else:
            self.model = ExponentialSmoothing(
                series,
                seasonal_periods=self.seasonal_period,
                trend='add',
                seasonal='add',
                damped_trend=True
            ).fit()
            self.best_config = {
                'trend': 'add',
                'seasonal': 'add',
                'damped': True
            }

        return self

    def forecast(
        self,
        steps: int = 24,
        return_bounds: bool = True
    ) -> pd.DataFrame:
        """
        Genera forecast de metricas.
        """
        if self.model is None:
            raise ValueError("Modelo no ajustado")

        forecast = self.model.forecast(steps)

        if return_bounds:
            # Intervalos basados en residuos
            residuals = self.model.resid
            std = residuals.std()

            return pd.DataFrame({
                'forecast': forecast,
                'lower': forecast - 1.96 * std,
                'upper': forecast + 1.96 * std
            })

        return pd.DataFrame({'forecast': forecast})

    def detect_anomaly(
        self,
        actual: float,
        expected: float,
        threshold_std: float = 2.5
    ) -> dict:
        """
        Detecta si un valor es anomalo comparado con forecast.
        """
        if self.model is None:
            raise ValueError("Modelo no ajustado")

        std = self.model.resid.std()
        z_score = (actual - expected) / std if std > 0 else 0

        return {
            'metric': self.metric_name,
            'actual': actual,
            'expected': expected,
            'deviation': actual - expected,
            'z_score': z_score,
            'is_anomaly': abs(z_score) > threshold_std,
            'severity': (
                'critical' if abs(z_score) > 4 else
                'high' if abs(z_score) > 3 else
                'medium' if abs(z_score) > threshold_std else
                'normal'
            )
        }

    def get_seasonal_pattern(self) -> pd.Series:
        """
        Extrae el patron estacional.
        Util para entender comportamiento normal.
        """
        if self.model is None or self.model.season is None:
            return None

        return pd.Series(
            self.model.season[-self.seasonal_period:],
            name=f'{self.metric_name}_seasonal_pattern'
        )


# Pipeline completo
def security_monitoring_pipeline(
    metrics_df: pd.DataFrame,
    metric_col: str,
    timestamp_col: str = 'timestamp'
) -> dict:
    """
    Pipeline completo de monitoreo con ETS.

    Args:
        metrics_df: DataFrame con metricas
        metric_col: Columna de metrica
        timestamp_col: Columna de timestamp

    Returns:
        Resultados del analisis
    """
    # Preparar serie
    metrics_df[timestamp_col] = pd.to_datetime(metrics_df[timestamp_col])
    series = metrics_df.set_index(timestamp_col)[metric_col].resample('H').mean()

    # Dividir train/test
    train_size = int(len(series) * 0.8)
    train = series[:train_size]
    test = series[train_size:]

    # Ajustar modelo
    forecaster = SecurityMetricsForecaster(
        metric_name=metric_col,
        seasonal_period=24,
        auto_select=True
    )
    forecaster.fit(train)

    # Evaluar en test
    forecast = forecaster.forecast(len(test))
    mse = np.mean((test.values - forecast['forecast'].values) ** 2)
    mae = np.mean(np.abs(test.values - forecast['forecast'].values))

    # Detectar anomalias en test
    anomalies = []
    for i, (actual, expected) in enumerate(zip(test.values, forecast['forecast'].values)):
        result = forecaster.detect_anomaly(actual, expected)
        if result['is_anomaly']:
            result['timestamp'] = test.index[i]
            anomalies.append(result)

    return {
        'model_config': forecaster.best_config,
        'train_size': train_size,
        'test_size': len(test),
        'mse': mse,
        'mae': mae,
        'anomalies': anomalies,
        'n_anomalies': len(anomalies),
        'seasonal_pattern': forecaster.get_seasonal_pattern()
    }


# Ejemplo
if __name__ == "__main__":
    # Simular metricas de red
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', periods=24*14, freq='H')
    hours = np.arange(len(dates)) % 24

    # Patron: bajo en la noche, alto en el dia
    daily_pattern = 100 + 50 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.randn(len(dates)) * 10

    traffic = daily_pattern + trend + noise
    traffic = np.maximum(traffic, 0)  # No negativos

    # Inyectar anomalias
    traffic[100] = 300  # Spike
    traffic[200] = 10   # Drop

    metrics_df = pd.DataFrame({
        'timestamp': dates,
        'bytes_per_second': traffic
    })

    # Ejecutar pipeline
    print("=== Security Monitoring Pipeline ===")
    results = security_monitoring_pipeline(metrics_df, 'bytes_per_second')

    print(f"\nConfiguracion: {results['model_config']}")
    print(f"MSE: {results['mse']:.2f}")
    print(f"MAE: {results['mae']:.2f}")
    print(f"Anomalias detectadas: {results['n_anomalies']}")

    for anomaly in results['anomalies'][:5]:
        print(f"  {anomaly['timestamp']}: {anomaly['severity']} "
              f"(z={anomaly['z_score']:.2f})")
```

---

## Comparativa de Metodos

| Metodo | Tendencia | Estacionalidad | Parametros | Uso |
|--------|-----------|----------------|------------|-----|
| SES | No | No | 1 (α) | Series estables |
| Holt | Si | No | 2 (α, β) | Series con tendencia |
| Holt Damped | Si (amort.) | No | 3 (α, β, φ) | Tendencia que se estabiliza |
| Holt-Winters Add | Si | Si (cte) | 3 (α, β, γ) | Estacionalidad constante |
| Holt-Winters Mul | Si | Si (prop) | 3 (α, β, γ) | Estacionalidad proporcional |

---

## Resumen

```
SELECCION DE METODO
===================

1. ¿Hay tendencia?
   No  → SES
   Si  → Continuar

2. ¿La tendencia es sostenida?
   Si  → Holt lineal
   No  → Holt damped

3. ¿Hay estacionalidad?
   No  → Holt/Holt damped
   Si  → Holt-Winters

4. ¿La estacionalidad es proporcional?
   No (constante)  → Holt-Winters Aditivo
   Si (crece)      → Holt-Winters Multiplicativo


PARAMETROS TIPICOS
==================
α (nivel):       0.1 - 0.3 para series ruidosas
                 0.5 - 0.9 para series estables
β (tendencia):   0.01 - 0.2 (tendencia suave)
γ (estacional):  0.01 - 0.3
φ (damping):     0.8 - 0.98
```

### Puntos Clave

1. **SES** es baseline para series sin patron
2. **Holt** captura tendencia lineal
3. **Holt damped** para tendencias que se estabilizan
4. **Holt-Winters** para patrones estacionales
5. **Aditivo vs multiplicativo**: depende de si la estacionalidad crece
6. En ciberseguridad: ideal para metricas con patrones diarios/semanales
