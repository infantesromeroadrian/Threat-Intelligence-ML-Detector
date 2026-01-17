# ARIMA y SARIMA

## Introduccion

**ARIMA** (AutoRegressive Integrated Moving Average) es la familia de modelos mas utilizada para series temporales. Combina tres componentes:

- **AR** (AutoRegressive): La serie depende de sus valores pasados
- **I** (Integrated): Diferenciacion para estacionariedad
- **MA** (Moving Average): La serie depende de errores pasados

```
ARIMA(p, d, q)
==============

p = orden autoregresivo (cuantos lags de Y usar)
d = orden de diferenciacion (cuantas veces diferenciar)
q = orden de media movil (cuantos lags de errores usar)


Ejemplos comunes:
─────────────────
ARIMA(1,0,0) = AR(1)     - Solo autoregresivo
ARIMA(0,0,1) = MA(1)     - Solo media movil
ARIMA(1,1,1)             - Modelo clasico
ARIMA(0,1,0) = Random Walk
```

---

## Modelo AR (AutoRegressive)

### Concepto

```
AR(p): AutoRegressive de orden p
======================================

Y_t = c + φ₁Y_{t-1} + φ₂Y_{t-2} + ... + φ_pY_{t-p} + ε_t

Donde:
- c: constante
- φ_i: coeficientes autoregresivos
- ε_t: ruido blanco (error aleatorio)


AR(1): Y_t = c + φ₁Y_{t-1} + ε_t

    │Y
    │       ●
    │    ●     ●
    │  ●         ●
    │●             ●
    └─────────────────── t

    Y_t depende de Y_{t-1}
    Si φ₁ > 0: persistencia (valores altos siguen altos)
    Si φ₁ < 0: oscilacion


AR(2): Y_t = c + φ₁Y_{t-1} + φ₂Y_{t-2} + ε_t

    Y_t depende de los 2 valores anteriores
    Puede capturar ciclos
```

### Condiciones de Estacionariedad

```
ESTACIONARIEDAD EN AR
=====================

Para AR(1):
    |φ₁| < 1  → Estacionario

Para AR(2):
    φ₁ + φ₂ < 1
    φ₂ - φ₁ < 1
    |φ₂| < 1

General: Raices del polinomio caracteristico fuera del circulo unitario.


Ejemplo AR(1):

φ₁ = 0.9 (estacionario):     φ₁ = 1.0 (random walk):
│                            │              ___
│  ~~~∿~~~∿~~~              │         ___/
│                            │    ___/
│  Revierte a media          │___/   No revierte
└───────────────             └───────────────
```

### Implementacion AR

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


def fit_ar_model(
    series: pd.Series,
    lags: int = 1,
    trend: str = 'c'
) -> dict:
    """
    Ajusta modelo AR.

    Args:
        series: Serie temporal estacionaria
        lags: Numero de lags (orden p)
        trend: 'c' (constante), 'n' (ninguno), 'ct' (constante + tendencia)

    Returns:
        Diccionario con modelo y resultados
    """
    model = AutoReg(series, lags=lags, trend=trend)
    fitted = model.fit()

    return {
        'model': fitted,
        'params': fitted.params,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'summary': fitted.summary()
    }


def forecast_ar(
    fitted_model,
    steps: int = 10
) -> pd.Series:
    """Genera predicciones con modelo AR."""
    return fitted_model.forecast(steps=steps)


# Ejemplo
if __name__ == "__main__":
    # Generar AR(1)
    np.random.seed(42)
    n = 200
    phi = 0.7

    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 10 + phi * y[t-1] + np.random.randn()

    series = pd.Series(y)

    # Ajustar
    result = fit_ar_model(series, lags=1)
    print(f"Coeficiente estimado: {result['params']['y.L1']:.3f} (real: {phi})")
```

---

## Modelo MA (Moving Average)

### Concepto

```
MA(q): Moving Average de orden q
================================

Y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}

Donde:
- μ: media del proceso
- θ_i: coeficientes de media movil
- ε_t: ruido blanco


MA(1): Y_t = μ + ε_t + θ₁ε_{t-1}

    Y_t depende del error actual y el anterior
    Shocks tienen efecto de 1 periodo


MA(2): Y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2}

    Shocks afectan por 2 periodos


DIFERENCIA AR vs MA:
───────────────────
AR: Depende de VALORES pasados de Y
MA: Depende de ERRORES pasados

AR tiene memoria "infinita" (decae exponencial)
MA tiene memoria "finita" (corte abrupto en q)
```

### Identificacion por ACF/PACF

```
PATRONES ACF/PACF
=================

AR(p):
- ACF: Decae exponencialmente
- PACF: Corta en lag p

MA(q):
- ACF: Corta en lag q
- PACF: Decae exponencialmente


Visualizacion:

AR(2):                          MA(2):
   ACF          PACF               ACF          PACF
   │█           │█                 │█           │█
   │██          │█                 │██          │██
   │███                            │            │███
   │████                                        │████
   │█████                                       │█████
  (decae)    (corte p=2)       (corte q=2)    (decae)
```

---

## Modelo ARMA

### Concepto

```
ARMA(p, q): Combinacion AR + MA
===============================

Y_t = c + φ₁Y_{t-1} + ... + φ_pY_{t-p} + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}

Forma compacta con operadores:
    φ(B)Y_t = θ(B)ε_t

Donde B es el operador de retardo: BY_t = Y_{t-1}


ARMA(1,1):
    Y_t = c + φ₁Y_{t-1} + ε_t + θ₁ε_{t-1}

    - Combina persistencia (AR) con shocks (MA)
    - ACF y PACF ambos decaen
    - Muy flexible


Identificacion ARMA:
───────────────────
Si ACF y PACF ambos decaen gradualmente → ARMA
Determinar p y q requiere criterios de informacion (AIC, BIC)
```

---

## Modelo ARIMA

### Concepto

```
ARIMA(p, d, q): ARMA con Diferenciacion
=======================================

Cuando la serie NO es estacionaria, primero diferenciamos.

d = numero de diferenciaciones necesarias

ARIMA(p, d, q) en la serie original Y_t es equivalente a
ARMA(p, q) en la serie diferenciada d veces.


Ejemplo ARIMA(1, 1, 1):
──────────────────────

Serie original Y_t (no estacionaria, tiene tendencia)
        │
        ▼
Diferenciar: Y'_t = Y_t - Y_{t-1}  (ahora estacionaria)
        │
        ▼
Aplicar ARMA(1,1) a Y'_t:
    Y'_t = c + φ₁Y'_{t-1} + ε_t + θ₁ε_{t-1}


En terminos de Y_t:
    (1 - B)(Y_t) = c + φ₁(1 - B)Y_{t-1} + ε_t + θ₁ε_{t-1}
```

### Seleccion del Orden (p, d, q)

```
PROCESO DE IDENTIFICACION
=========================

1. DETERMINAR d (diferenciacion):
   - Test ADF/KPSS
   - Diferenciar hasta que sea estacionaria
   - Raramente d > 2

2. DETERMINAR p y q:
   a) Metodo ACF/PACF:
      - PACF corta en p → AR(p)
      - ACF corta en q → MA(q)
      - Ambos decaen → probar varios ARMA

   b) Metodo automatico (AIC/BIC):
      - Probar combinaciones de p, q
      - Elegir menor AIC o BIC


CRITERIOS DE INFORMACION
========================

AIC = 2k - 2ln(L)
BIC = k·ln(n) - 2ln(L)

Donde:
- k: numero de parametros
- L: verosimilitud
- n: numero de observaciones

BIC penaliza mas la complejidad (prefiere modelos simples)
```

### Implementacion ARIMA

```python
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product
from typing import Tuple


def determine_d(series: pd.Series, max_d: int = 2) -> int:
    """
    Determina orden de diferenciacion necesario.

    Args:
        series: Serie temporal
        max_d: Maximo d a probar

    Returns:
        Orden d optimo
    """
    for d in range(max_d + 1):
        if d == 0:
            test_series = series
        else:
            test_series = series.diff(d).dropna()

        adf_result = adfuller(test_series, autolag='AIC')
        p_value = adf_result[1]

        if p_value < 0.05:
            print(f"Serie estacionaria con d={d} (p-value={p_value:.4f})")
            return d

    print(f"Warning: Serie no estacionaria con d={max_d}")
    return max_d


def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int]
) -> dict:
    """
    Ajusta modelo ARIMA.

    Args:
        series: Serie temporal
        order: (p, d, q)

    Returns:
        Diccionario con resultados
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()

    return {
        'model': fitted,
        'order': order,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'params': fitted.params,
        'residuals': fitted.resid,
        'summary': fitted.summary()
    }


def auto_arima(
    series: pd.Series,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    criterion: str = 'aic'
) -> dict:
    """
    Seleccion automatica de orden ARIMA.

    Args:
        series: Serie temporal
        max_p: Maximo p
        max_d: Maximo d
        max_q: Maximo q
        criterion: 'aic' o 'bic'

    Returns:
        Mejor modelo encontrado
    """
    # Determinar d
    d = determine_d(series, max_d)

    best_score = np.inf
    best_order = None
    best_model = None

    results_log = []

    # Grid search
    for p, q in product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(series, order=(p, d, q))
                fitted = model.fit()

                score = fitted.aic if criterion == 'aic' else fitted.bic

                results_log.append({
                    'order': (p, d, q),
                    'aic': fitted.aic,
                    'bic': fitted.bic
                })

                if score < best_score:
                    best_score = score
                    best_order = (p, d, q)
                    best_model = fitted

        except Exception:
            continue

    if best_model is None:
        raise ValueError("No se pudo ajustar ningun modelo ARIMA")

    return {
        'model': best_model,
        'order': best_order,
        'aic': best_model.aic,
        'bic': best_model.bic,
        'all_results': pd.DataFrame(results_log).sort_values(criterion)
    }


def forecast_arima(
    fitted_model,
    steps: int = 10,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Genera predicciones con intervalos de confianza.

    Args:
        fitted_model: Modelo ARIMA ajustado
        steps: Numero de pasos a predecir
        alpha: Nivel de significancia para intervalos

    Returns:
        DataFrame con predicciones e intervalos
    """
    forecast = fitted_model.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=alpha)

    return pd.DataFrame({
        'forecast': pred_mean,
        'lower': conf_int.iloc[:, 0],
        'upper': conf_int.iloc[:, 1]
    })


def diagnose_residuals(fitted_model) -> dict:
    """
    Diagnostica residuos del modelo ARIMA.

    Los residuos deben ser:
    - Ruido blanco (sin autocorrelacion)
    - Media cero
    - Varianza constante
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy import stats

    residuals = fitted_model.resid

    # Test Ljung-Box para autocorrelacion
    lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)

    # Test de normalidad
    _, normality_pvalue = stats.normaltest(residuals)

    # Estadisticas basicas
    diagnostics = {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'ljung_box': lb_test,
        'ljung_box_pass': all(lb_test['lb_pvalue'] > 0.05),
        'normality_pvalue': normality_pvalue,
        'normality_pass': normality_pvalue > 0.05
    }

    # Resumen
    diagnostics['model_adequate'] = (
        diagnostics['ljung_box_pass'] and
        abs(diagnostics['mean']) < 0.1 * diagnostics['std']
    )

    return diagnostics


# Ejemplo completo
if __name__ == "__main__":
    # Generar serie ARIMA(1,1,1)
    np.random.seed(42)
    n = 200

    # Random walk + ARMA
    y = np.zeros(n)
    e = np.random.randn(n)

    for t in range(2, n):
        y[t] = y[t-1] + 0.5 * (y[t-1] - y[t-2]) + e[t] + 0.3 * e[t-1]

    series = pd.Series(y, name='y')

    # Auto ARIMA
    print("Buscando mejor modelo...")
    result = auto_arima(series, max_p=3, max_d=2, max_q=3)

    print(f"\nMejor orden: {result['order']}")
    print(f"AIC: {result['aic']:.2f}")
    print(f"BIC: {result['bic']:.2f}")

    # Diagnostico
    print("\nDiagnostico de residuos:")
    diag = diagnose_residuals(result['model'])
    print(f"  Ljung-Box OK: {diag['ljung_box_pass']}")
    print(f"  Modelo adecuado: {diag['model_adequate']}")

    # Forecast
    print("\nPredicciones (5 pasos):")
    pred = forecast_arima(result['model'], steps=5)
    print(pred)
```

---

## Modelo SARIMA

### Concepto

```
SARIMA: ARIMA con Estacionalidad
================================

SARIMA(p, d, q)(P, D, Q)[s]

Componente no-estacional: (p, d, q)
Componente estacional:    (P, D, Q)[s]

s = periodo estacional (12 para mensual, 4 para trimestral, 24 para horario)


Ecuacion:
─────────
φ(B)Φ(B^s)(1-B)^d(1-B^s)^D Y_t = θ(B)Θ(B^s)ε_t

Donde:
- φ(B): polinomio AR no-estacional
- Φ(B^s): polinomio AR estacional
- θ(B): polinomio MA no-estacional
- Θ(B^s): polinomio MA estacional
- (1-B)^d: diferenciacion no-estacional
- (1-B^s)^D: diferenciacion estacional


Ejemplo: SARIMA(1,1,1)(1,1,1)[12] para datos mensuales

- AR(1) no-estacional
- MA(1) no-estacional
- Una diferencia regular
- AR(1) estacional (lag 12)
- MA(1) estacional (lag 12)
- Una diferencia estacional
```

### Visualizacion Estacionalidad

```
PATRONES ESTACIONALES
=====================

Trafico de red con patron diario (s=24 horas):

Hora:  0  4  8  12 16 20 24  4  8  12 16 20 24  4  8  ...
       │  │  │   │  │  │  │  │  │   │  │  │  │  │  │
       │  ▼  │   │  ▼  │  │  ▼  │   │  ▼  │  │  ▼  │
Value: └──┘  └───┴──┘  └──┘  └───┴──┘  └──┘  └───┴──┘

       ◄────── Dia 1 ──────►◄────── Dia 2 ──────►◄─...

El patron se repite cada 24 horas.


ACF con estacionalidad s=24:
    │
    │█  █  █  █  █  █  ...
    │ █  █  █  █  █
    │  █  █  █  █
    │   ███████
    └───────────────────────
     0     24    48    72
           Picos en lags multiplos de s
```

### Implementacion SARIMA

```python
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from typing import Tuple


def fit_sarima(
    series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int]
) -> dict:
    """
    Ajusta modelo SARIMA.

    Args:
        series: Serie temporal
        order: (p, d, q) no-estacional
        seasonal_order: (P, D, Q, s) estacional

    Returns:
        Diccionario con resultados
    """
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)

    return {
        'model': fitted,
        'order': order,
        'seasonal_order': seasonal_order,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'params': fitted.params,
        'summary': fitted.summary()
    }


def auto_sarima(
    series: pd.Series,
    seasonal_period: int,
    max_p: int = 2,
    max_d: int = 1,
    max_q: int = 2,
    max_P: int = 1,
    max_D: int = 1,
    max_Q: int = 1,
    criterion: str = 'aic'
) -> dict:
    """
    Seleccion automatica de SARIMA.

    Args:
        series: Serie temporal
        seasonal_period: Periodo estacional s
        max_*: Maximos ordenes a probar
        criterion: 'aic' o 'bic'

    Returns:
        Mejor modelo encontrado
    """
    best_score = np.inf
    best_result = None
    results_log = []

    # Determinar d y D
    d = determine_d(series, max_d)

    # Diferencia estacional si es necesario
    seasonal_diff = series.diff(seasonal_period).dropna()
    D = 1 if adfuller(seasonal_diff)[1] < adfuller(series)[1] else 0

    print(f"Usando d={d}, D={D}, s={seasonal_period}")

    # Grid search
    total = (max_p + 1) * (max_q + 1) * (max_P + 1) * (max_Q + 1)
    count = 0

    for p, q, P, Q in product(
        range(max_p + 1),
        range(max_q + 1),
        range(max_P + 1),
        range(max_Q + 1)
    ):
        count += 1

        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted = model.fit(disp=False, maxiter=100)

                score = fitted.aic if criterion == 'aic' else fitted.bic

                results_log.append({
                    'order': (p, d, q),
                    'seasonal': (P, D, Q, seasonal_period),
                    'aic': fitted.aic,
                    'bic': fitted.bic
                })

                if score < best_score:
                    best_score = score
                    best_result = {
                        'model': fitted,
                        'order': (p, d, q),
                        'seasonal_order': (P, D, Q, seasonal_period),
                        'aic': fitted.aic,
                        'bic': fitted.bic
                    }

                    print(f"  [{count}/{total}] Nuevo mejor: "
                          f"({p},{d},{q})({P},{D},{Q})[{seasonal_period}] "
                          f"{criterion.upper()}={score:.2f}")

        except Exception:
            continue

    if best_result is None:
        raise ValueError("No se pudo ajustar ningun modelo SARIMA")

    best_result['all_results'] = pd.DataFrame(results_log).sort_values(criterion)

    return best_result


def forecast_sarima(
    fitted_model,
    steps: int,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Genera predicciones SARIMA con intervalos.
    """
    forecast = fitted_model.get_forecast(steps=steps)

    return pd.DataFrame({
        'forecast': forecast.predicted_mean,
        'lower': forecast.conf_int().iloc[:, 0],
        'upper': forecast.conf_int().iloc[:, 1]
    })


# Ejemplo con datos estacionales
def example_sarima():
    """Ejemplo SARIMA con datos de trafico."""

    # Simular trafico con patron diario
    np.random.seed(42)

    # 30 dias de datos horarios
    n = 24 * 30
    hours = np.arange(n) % 24

    # Patron diario: pico en horas laborales
    daily_pattern = (
        50 * np.sin(2 * np.pi * hours / 24 - np.pi/2) +  # Ciclo
        30 * (hours >= 9) * (hours <= 17)                  # Horas laborales
    )

    # Tendencia suave
    trend = np.linspace(100, 120, n)

    # Ruido
    noise = np.random.randn(n) * 10

    # Serie final
    traffic = trend + daily_pattern + noise
    series = pd.Series(
        traffic,
        index=pd.date_range('2024-01-01', periods=n, freq='H'),
        name='traffic'
    )

    print("=== Ajustando SARIMA para trafico de red ===")
    print(f"Periodo estacional: 24 (horario)")

    # Auto SARIMA
    result = auto_sarima(
        series,
        seasonal_period=24,
        max_p=2, max_d=1, max_q=2,
        max_P=1, max_D=1, max_Q=1
    )

    print(f"\nMejor modelo: SARIMA{result['order']}{result['seasonal_order']}")
    print(f"AIC: {result['aic']:.2f}")

    # Prediccion proximas 24 horas
    pred = forecast_sarima(result['model'], steps=24)
    print(f"\nPrediccion proximas 24 horas:")
    print(pred.head(10))

    return result


if __name__ == "__main__":
    example_sarima()
```

---

## Aplicacion: Prediccion de Alertas SIEM

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SIEMAlertForecaster:
    """
    Predictor de alertas SIEM usando SARIMA.
    Util para capacity planning y deteccion de anomalias.
    """

    def __init__(
        self,
        seasonal_period: int = 24,  # Horario
        order: tuple = (1, 1, 1),
        seasonal_order: tuple | None = None
    ):
        self.seasonal_period = seasonal_period
        self.order = order
        self.seasonal_order = seasonal_order or (1, 1, 1, seasonal_period)
        self.model = None
        self.fitted = None

    def fit(self, alert_counts: pd.Series) -> 'SIEMAlertForecaster':
        """
        Entrena modelo con conteos historicos de alertas.

        Args:
            alert_counts: Serie temporal de conteo de alertas
        """
        self.model = SARIMAX(
            alert_counts,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted = self.model.fit(disp=False)
        return self

    def predict(
        self,
        steps: int = 24,
        return_conf_int: bool = True
    ) -> pd.DataFrame:
        """
        Predice alertas futuras.

        Args:
            steps: Horas a predecir
            return_conf_int: Incluir intervalos de confianza

        Returns:
            DataFrame con predicciones
        """
        if self.fitted is None:
            raise ValueError("Modelo no entrenado. Llama fit() primero.")

        forecast = self.fitted.get_forecast(steps=steps)

        result = pd.DataFrame({
            'predicted_alerts': forecast.predicted_mean.round().astype(int)
        })

        if return_conf_int:
            conf = forecast.conf_int()
            result['lower_bound'] = conf.iloc[:, 0].clip(lower=0).round().astype(int)
            result['upper_bound'] = conf.iloc[:, 1].round().astype(int)

        return result

    def detect_anomaly(
        self,
        actual: float,
        predicted: float,
        std: float,
        threshold: float = 2.0
    ) -> dict:
        """
        Detecta si un valor es anomalo.

        Args:
            actual: Valor real observado
            predicted: Valor predicho
            std: Desviacion estandar del modelo
            threshold: Numero de desviaciones para anomalia

        Returns:
            Diccionario con resultado
        """
        z_score = (actual - predicted) / std if std > 0 else 0

        return {
            'actual': actual,
            'predicted': predicted,
            'z_score': z_score,
            'is_anomaly': abs(z_score) > threshold,
            'anomaly_type': (
                'spike' if z_score > threshold else
                'drop' if z_score < -threshold else
                'normal'
            )
        }

    def get_model_diagnostics(self) -> dict:
        """Retorna diagnosticos del modelo."""
        if self.fitted is None:
            raise ValueError("Modelo no entrenado")

        return {
            'aic': self.fitted.aic,
            'bic': self.fitted.bic,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'residual_std': self.fitted.resid.std()
        }


def analyze_siem_alerts(
    alerts_df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    freq: str = 'H'
) -> pd.Series:
    """
    Prepara serie temporal de alertas desde logs SIEM.

    Args:
        alerts_df: DataFrame con alertas
        timestamp_col: Columna de timestamp
        freq: Frecuencia de agregacion

    Returns:
        Serie temporal de conteos
    """
    alerts_df[timestamp_col] = pd.to_datetime(alerts_df[timestamp_col])

    # Contar alertas por periodo
    alert_counts = (
        alerts_df
        .set_index(timestamp_col)
        .resample(freq)
        .size()
        .fillna(0)
    )

    return alert_counts


# Ejemplo de uso
if __name__ == "__main__":
    # Simular alertas SIEM
    np.random.seed(42)

    # 7 dias de alertas horarias
    dates = pd.date_range('2024-01-01', periods=24*7, freq='H')

    # Patron: mas alertas en horas laborales
    hours = dates.hour
    base_rate = 50
    hourly_pattern = 30 * ((hours >= 9) & (hours <= 18)).astype(int)
    noise = np.random.poisson(10, len(dates))

    alert_counts = pd.Series(
        base_rate + hourly_pattern + noise,
        index=dates,
        name='alert_count'
    )

    # Entrenar modelo
    forecaster = SIEMAlertForecaster(
        seasonal_period=24,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24)
    )
    forecaster.fit(alert_counts)

    # Diagnosticos
    print("=== Modelo SIEM Alert Forecaster ===")
    diag = forecaster.get_model_diagnostics()
    print(f"AIC: {diag['aic']:.2f}")
    print(f"Residual STD: {diag['residual_std']:.2f}")

    # Prediccion
    print("\nPrediccion proximas 12 horas:")
    pred = forecaster.predict(steps=12)
    print(pred)

    # Simulacion deteccion anomalia
    print("\nDeteccion de anomalia:")
    anomaly = forecaster.detect_anomaly(
        actual=150,  # Valor muy alto
        predicted=60,
        std=diag['residual_std']
    )
    print(f"  Valor actual: {anomaly['actual']}")
    print(f"  Predicho: {anomaly['predicted']}")
    print(f"  Z-score: {anomaly['z_score']:.2f}")
    print(f"  Es anomalia: {anomaly['is_anomaly']}")
    print(f"  Tipo: {anomaly['anomaly_type']}")
```

---

## Comparativa ARIMA vs SARIMA

| Aspecto | ARIMA | SARIMA |
|---------|-------|--------|
| Estacionalidad | No maneja | Si maneja |
| Parametros | 3 (p, d, q) | 7 (p,d,q,P,D,Q,s) |
| Complejidad | Menor | Mayor |
| Datos necesarios | Menos | Mas (varios ciclos) |
| Uso tipico | Tendencias | Patrones periodicos |

---

## Resumen

```
ARIMA CHEATSHEET
================

1. VERIFICAR ESTACIONARIEDAD
   - Test ADF (p < 0.05 → estacionaria)
   - Diferenciar si es necesario (d)

2. IDENTIFICAR ORDEN
   - PACF corta en p → AR(p)
   - ACF corta en q → MA(q)
   - Ambos decaen → ARMA

3. AJUSTAR MODELO
   - auto_arima() o grid search
   - Minimizar AIC/BIC

4. DIAGNOSTICAR
   - Ljung-Box: residuos sin autocorrelacion
   - Residuos ~ ruido blanco

5. PREDECIR
   - forecast con intervalos
   - Validar con datos retenidos


SARIMA: Anadir (P, D, Q)[s] para estacionalidad
```

### Puntos Clave

1. **ARIMA** requiere serie estacionaria (diferenciar si es necesario)
2. **ACF/PACF** ayudan a identificar ordenes p y q
3. **AIC/BIC** para seleccion automatica de modelo
4. **SARIMA** extiende ARIMA para patrones estacionales
5. **Diagnostico** de residuos es obligatorio
6. En ciberseguridad: prediccion de trafico, alertas, tendencias
