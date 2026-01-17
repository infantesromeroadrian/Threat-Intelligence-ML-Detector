# Prophet y Descomposicion de Series Temporales

## Introduccion

La **descomposicion** separa una serie temporal en sus componentes fundamentales, facilitando el analisis y mejorando los modelos de forecasting. **Prophet** de Meta es una herramienta que automatiza este proceso y maneja casos complejos como festivos y cambios de tendencia.

```
DESCOMPOSICION
==============

Serie Original = f(Tendencia, Estacionalidad, Residuo)

Y_t = T_t + S_t + R_t    (Modelo Aditivo)
Y_t = T_t × S_t × R_t    (Modelo Multiplicativo)


    Serie Original          Tendencia            Estacionalidad         Residuo
         │                     │                      │                   │
    ┌────┴────┐           ┌────┴────┐            ┌────┴────┐         ┌────┴────┐
    │  /\  /\ │           │    ___--│            │ /\/\/\/\│         │ · · · · │
    │ /  \/  \│      =    │___/     │      +     │         │    +    │  ·  · · │
    │/        │           │         │            │         │         │· ·    · │
    └─────────┘           └─────────┘            └─────────┘         └─────────┘
```

---

## Descomposicion Clasica

### Tipos de Descomposicion

```
ADITIVO vs MULTIPLICATIVO
=========================

ADITIVO: Y = T + S + R
Usar cuando: Variacion estacional CONSTANTE

    │     /\    /\                 Amplitud igual
    │    /  \  /  \
    │   /    \/    \
    │__/            \__
    └──────────────────


MULTIPLICATIVO: Y = T × S × R
Usar cuando: Variacion estacional PROPORCIONAL al nivel

    │              /\
    │         /\  /  \
    │    /\  /  \/    \            Amplitud crece
    │___/  \/
    └──────────────────


MIXTO: Y = T × S + R
Cuando solo el residuo es aditivo.
```

### Implementacion Descomposicion Clasica

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import matplotlib.pyplot as plt
from typing import Literal


def classical_decomposition(
    series: pd.Series,
    period: int,
    model: Literal['additive', 'multiplicative'] = 'additive'
) -> dict:
    """
    Descomposicion clasica de series temporales.

    Args:
        series: Serie temporal con indice datetime
        period: Periodo estacional
        model: 'additive' o 'multiplicative'

    Returns:
        Componentes de la descomposicion
    """
    result = seasonal_decompose(
        series,
        model=model,
        period=period,
        extrapolate_trend='freq'
    )

    return {
        'observed': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid,
        'model': model,
        'period': period
    }


def plot_decomposition(decomposition: dict, figsize: tuple = (12, 10)) -> None:
    """Grafica los componentes de la descomposicion."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    axes[0].plot(decomposition['observed'])
    axes[0].set_title('Observed')

    axes[1].plot(decomposition['trend'])
    axes[1].set_title('Trend')

    axes[2].plot(decomposition['seasonal'])
    axes[2].set_title('Seasonal')

    axes[3].plot(decomposition['residual'])
    axes[3].set_title('Residual')

    plt.tight_layout()
    plt.show()


def extract_seasonal_pattern(decomposition: dict) -> pd.Series:
    """
    Extrae un ciclo del patron estacional.

    Returns:
        Serie con un periodo del patron
    """
    seasonal = decomposition['seasonal']
    period = decomposition['period']

    # Tomar el primer ciclo completo (ignorando NaN)
    valid = seasonal.dropna()
    pattern = valid[:period]

    return pattern


# Ejemplo
if __name__ == "__main__":
    # Crear serie con patron conocido
    np.random.seed(42)
    n = 365 * 2  # 2 anos diarios

    t = np.arange(n)
    trend = 100 + 0.05 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 365)  # Ciclo anual
    noise = np.random.randn(n) * 5

    series = pd.Series(
        trend + seasonal + noise,
        index=pd.date_range('2022-01-01', periods=n, freq='D')
    )

    # Descomponer
    decomp = classical_decomposition(series, period=365, model='additive')

    print("Descomposicion completada:")
    print(f"  Trend range: [{decomp['trend'].min():.1f}, {decomp['trend'].max():.1f}]")
    print(f"  Seasonal range: [{decomp['seasonal'].min():.1f}, {decomp['seasonal'].max():.1f}]")
    print(f"  Residual std: {decomp['residual'].std():.2f}")
```

---

## STL Decomposition

### Concepto

```
STL: Seasonal and Trend decomposition using Loess
=================================================

Mejora sobre descomposicion clasica:
- Robusto a outliers
- Estacionalidad puede cambiar en el tiempo
- Usa LOESS (Local Regression) para suavizado


Proceso iterativo:
─────────────────

1. Inicializar tendencia T = 0

2. Loop hasta convergencia:
   a. De-trend: Y - T
   b. Estimar estacionalidad S con LOESS
   c. De-seasonalize: Y - S
   d. Estimar tendencia T con LOESS

3. Calcular residuo: R = Y - T - S


Parametros clave:
- seasonal: ventana para suavizado estacional (debe ser impar)
- trend: ventana para suavizado de tendencia
- robust: usar pesos para reducir efecto de outliers
```

### Implementacion STL

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def stl_decomposition(
    series: pd.Series,
    period: int,
    seasonal: int = 7,
    trend: int | None = None,
    robust: bool = True
) -> dict:
    """
    Descomposicion STL.

    Args:
        series: Serie temporal
        period: Periodo estacional
        seasonal: Ventana para estacionalidad (impar, >= 7)
        trend: Ventana para tendencia (None = automatico)
        robust: Usar pesos para outliers

    Returns:
        Componentes de la descomposicion
    """
    # Asegurar que seasonal es impar
    if seasonal % 2 == 0:
        seasonal += 1

    stl = STL(
        series,
        period=period,
        seasonal=seasonal,
        trend=trend,
        robust=robust
    )
    result = stl.fit()

    return {
        'observed': series,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid,
        'weights': result.weights if robust else None,
        'period': period
    }


def analyze_stl_components(decomposition: dict) -> dict:
    """
    Analiza los componentes STL.

    Returns:
        Estadisticas de cada componente
    """
    trend = decomposition['trend']
    seasonal = decomposition['seasonal']
    residual = decomposition['residual']
    observed = decomposition['observed']

    # Varianza explicada por cada componente
    total_var = observed.var()

    return {
        'trend_strength': max(0, 1 - residual.var() / (trend + residual).var()),
        'seasonal_strength': max(0, 1 - residual.var() / (seasonal + residual).var()),
        'trend_range': (trend.min(), trend.max()),
        'seasonal_amplitude': (seasonal.max() - seasonal.min()) / 2,
        'residual_std': residual.std(),
        'residual_mean': residual.mean(),
        'outliers': (
            decomposition['weights'] < 0.5
        ).sum() if decomposition['weights'] is not None else 0
    }


class STLForecaster:
    """
    Forecasting usando descomposicion STL.

    1. Descompone la serie
    2. Modela tendencia (ARIMA, ETS)
    3. Proyecta estacionalidad
    4. Combina para forecast
    """

    def __init__(
        self,
        period: int,
        trend_model: str = 'arima'
    ):
        self.period = period
        self.trend_model = trend_model
        self.decomposition = None
        self.fitted_trend = None

    def fit(self, series: pd.Series) -> 'STLForecaster':
        """Ajusta modelo."""
        # Descomponer
        self.decomposition = stl_decomposition(series, self.period)

        # Modelar tendencia
        trend = self.decomposition['trend']

        if self.trend_model == 'arima':
            from statsmodels.tsa.arima.model import ARIMA
            self.fitted_trend = ARIMA(trend, order=(1, 1, 1)).fit()
        else:  # ets
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self.fitted_trend = ExponentialSmoothing(
                trend, trend='add', damped_trend=True
            ).fit()

        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        Genera predicciones.

        Returns:
            DataFrame con forecast y componentes
        """
        if self.decomposition is None:
            raise ValueError("Modelo no ajustado")

        # Forecast de tendencia
        trend_forecast = self.fitted_trend.forecast(steps)

        # Proyectar estacionalidad (repetir patron)
        seasonal = self.decomposition['seasonal']
        seasonal_pattern = seasonal[-self.period:].values

        # Repetir patron para cubrir steps
        n_repeats = (steps // self.period) + 1
        seasonal_extended = np.tile(seasonal_pattern, n_repeats)[:steps]

        # Crear indice futuro
        last_date = self.decomposition['observed'].index[-1]
        freq = pd.infer_freq(self.decomposition['observed'].index)
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=freq[0].lower() if freq else 'D'),
            periods=steps,
            freq=freq
        )

        # Combinar
        forecast = trend_forecast.values + seasonal_extended

        return pd.DataFrame({
            'forecast': forecast,
            'trend': trend_forecast.values,
            'seasonal': seasonal_extended
        }, index=future_index)
```

---

## Facebook Prophet

### Concepto

```
PROPHET: Modelo Aditivo Generalizado
====================================

Prophet modela la serie como suma de componentes:

    y(t) = g(t) + s(t) + h(t) + ε_t

Donde:
- g(t): Tendencia (lineal o logistica con changepoints)
- s(t): Estacionalidad (Fourier series)
- h(t): Efectos de festivos
- ε_t: Error


TENDENCIA CON CHANGEPOINTS
==========================

Prophet detecta automaticamente puntos donde la tendencia cambia:

    │
    │                    ●●●●●
    │              ●●●●●●     ← rate = 0.01
    │        ●●●●●
    │  ●●●●●●               ← changepoint
    │●●      ← rate = 0.05
    └────────────────────────────
         t1      t2

    g(t) = (k + Σ a_j · δ_j(t)) · t + (m + Σ a_j · γ_j(t))

    k: tasa base
    a_j: ajuste en changepoint j
    δ_j: indicador si t >= t_j


ESTACIONALIDAD (Fourier)
========================

    s(t) = Σ [a_n cos(2πnt/P) + b_n sin(2πnt/P)]

    P = periodo (365.25 para anual, 7 para semanal)
    n = numero de terminos Fourier (mas = mas flexible)

    Terminos Fourier tipicos:
    - Anual: 10
    - Semanal: 3
```

### Instalacion y Uso Basico

```python
# pip install prophet

import pandas as pd
import numpy as np
from prophet import Prophet


def prophet_forecast(
    df: pd.DataFrame,
    periods: int = 30,
    freq: str = 'D',
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0
) -> dict:
    """
    Forecast usando Prophet.

    Args:
        df: DataFrame con columnas 'ds' (fecha) y 'y' (valor)
        periods: Periodos a predecir
        freq: Frecuencia ('D', 'H', 'W', etc.)
        yearly_seasonality: Incluir estacionalidad anual
        weekly_seasonality: Incluir estacionalidad semanal
        daily_seasonality: Incluir estacionalidad diaria
        changepoint_prior_scale: Flexibilidad de changepoints (mayor = mas flexible)
        seasonality_prior_scale: Flexibilidad de estacionalidad

    Returns:
        Diccionario con modelo, forecast y componentes
    """
    # Crear modelo
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )

    # Ajustar
    model.fit(df)

    # Crear dataframe futuro
    future = model.make_future_dataframe(periods=periods, freq=freq)

    # Predecir
    forecast = model.predict(future)

    return {
        'model': model,
        'forecast': forecast,
        'train_data': df,
        'components': model.plot_components
    }


def prepare_data_for_prophet(
    series: pd.Series
) -> pd.DataFrame:
    """
    Prepara datos para Prophet.

    Args:
        series: Serie temporal con indice datetime

    Returns:
        DataFrame con columnas 'ds' y 'y'
    """
    return pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })


# Ejemplo basico
if __name__ == "__main__":
    # Crear datos de ejemplo
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=365*2, freq='D')

    # Tendencia + estacionalidad + ruido
    t = np.arange(len(dates))
    trend = 100 + 0.1 * t
    yearly = 20 * np.sin(2 * np.pi * t / 365)
    weekly = 5 * np.sin(2 * np.pi * t / 7)
    noise = np.random.randn(len(dates)) * 5

    values = trend + yearly + weekly + noise

    df = pd.DataFrame({'ds': dates, 'y': values})

    # Prophet
    result = prophet_forecast(df, periods=30)

    print("Prophet Forecast:")
    print(result['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

### Prophet Avanzado

```python
from prophet import Prophet
import pandas as pd
import numpy as np
from typing import List, Dict


class ProphetForecaster:
    """
    Wrapper avanzado para Prophet con features de ciberseguridad.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        interval_width: float = 0.95
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.interval_width = interval_width
        self.model = None

    def add_custom_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int
    ) -> 'ProphetForecaster':
        """
        Anade estacionalidad personalizada.

        Args:
            name: Nombre de la estacionalidad
            period: Periodo en dias
            fourier_order: Numero de terminos Fourier
        """
        if self.model is None:
            self.model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                interval_width=self.interval_width,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False
            )

        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order
        )

        return self

    def add_holidays(self, holidays_df: pd.DataFrame) -> 'ProphetForecaster':
        """
        Anade festivos o eventos especiales.

        Args:
            holidays_df: DataFrame con columnas 'holiday', 'ds', 'lower_window', 'upper_window'
        """
        if self.model is None:
            self._init_model()

        self.model.holidays = holidays_df
        return self

    def add_regressor(self, name: str) -> 'ProphetForecaster':
        """
        Anade regresor externo.

        Args:
            name: Nombre de la columna en el DataFrame
        """
        if self.model is None:
            self._init_model()

        self.model.add_regressor(name)
        return self

    def _init_model(self):
        """Inicializa modelo con parametros por defecto."""
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            interval_width=self.interval_width
        )

    def fit(self, df: pd.DataFrame) -> 'ProphetForecaster':
        """
        Ajusta el modelo.

        Args:
            df: DataFrame con 'ds', 'y', y regresores opcionales
        """
        if self.model is None:
            self._init_model()

        self.model.fit(df)
        return self

    def predict(
        self,
        periods: int,
        freq: str = 'D',
        include_history: bool = True,
        future_regressors: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Genera predicciones.

        Args:
            periods: Periodos a predecir
            freq: Frecuencia
            include_history: Incluir predicciones historicas
            future_regressors: Valores futuros de regresores
        """
        if self.model is None:
            raise ValueError("Modelo no ajustado")

        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )

        # Anadir regresores si existen
        if future_regressors is not None:
            for col in future_regressors.columns:
                if col != 'ds':
                    future[col] = future_regressors[col]

        return self.model.predict(future)

    def get_changepoints(self) -> pd.DataFrame:
        """
        Obtiene changepoints detectados.
        Util para identificar cambios en tendencia.
        """
        if self.model is None:
            raise ValueError("Modelo no ajustado")

        changepoints = self.model.changepoints
        deltas = self.model.params['delta'].mean(axis=0)

        return pd.DataFrame({
            'changepoint': changepoints,
            'delta': deltas[:len(changepoints)]
        }).sort_values('delta', key=abs, ascending=False)

    def get_seasonality_components(self) -> Dict[str, pd.DataFrame]:
        """
        Obtiene componentes de estacionalidad.
        """
        if self.model is None:
            raise ValueError("Modelo no ajustado")

        components = {}

        # Generar un ano completo para ver patrones
        future = self.model.make_future_dataframe(periods=365, freq='D')
        forecast = self.model.predict(future)

        for seasonality in self.model.seasonalities:
            col = seasonality
            if col in forecast.columns:
                components[seasonality] = forecast[['ds', col]]

        return components


def create_security_events_holidays(
    events: List[Dict[str, str]]
) -> pd.DataFrame:
    """
    Crea DataFrame de eventos de seguridad para Prophet.

    Args:
        events: Lista de {'name': str, 'date': str, 'window': int}

    Returns:
        DataFrame compatible con Prophet holidays
    """
    holidays_list = []

    for event in events:
        holidays_list.append({
            'holiday': event['name'],
            'ds': pd.to_datetime(event['date']),
            'lower_window': -event.get('window', 0),
            'upper_window': event.get('window', 0)
        })

    return pd.DataFrame(holidays_list)


# Ejemplo: Forecasting de alertas con eventos de seguridad
def security_alerting_prophet():
    """
    Ejemplo de uso de Prophet para alertas de seguridad.
    """
    np.random.seed(42)

    # Simular alertas diarias (2 anos)
    dates = pd.date_range('2022-01-01', periods=730, freq='D')
    t = np.arange(len(dates))

    # Componentes
    trend = 50 + 0.02 * t
    yearly = 10 * np.sin(2 * np.pi * t / 365)
    weekly = 15 * (pd.Series(dates).dt.dayofweek < 5).astype(float).values  # Mas en dias laborales
    noise = np.random.randn(len(dates)) * 5

    # Spikes en eventos de seguridad
    patch_tuesday = (pd.Series(dates).dt.day == 14) & (pd.Series(dates).dt.dayofweek == 1)
    alerts = trend + yearly + weekly + noise + 30 * patch_tuesday.astype(float).values

    df = pd.DataFrame({'ds': dates, 'y': alerts})

    # Eventos de seguridad conocidos
    security_events = [
        {'name': 'Patch Tuesday', 'date': '2023-01-10', 'window': 1},
        {'name': 'Patch Tuesday', 'date': '2023-02-14', 'window': 1},
        {'name': 'Patch Tuesday', 'date': '2023-03-14', 'window': 1},
        # ... mas patch tuesdays
        {'name': 'Major Vulnerability', 'date': '2022-12-13', 'window': 3},  # Log4j aniversario
    ]

    holidays = create_security_events_holidays(security_events)

    # Modelo Prophet
    forecaster = ProphetForecaster(
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0
    )

    # Anadir estacionalidades custom
    forecaster.add_custom_seasonality('yearly', 365.25, 10)
    forecaster.add_custom_seasonality('weekly', 7, 3)
    forecaster.add_holidays(holidays)

    # Ajustar y predecir
    forecaster.fit(df)
    forecast = forecaster.predict(periods=30)

    print("=== Security Alerting Forecast ===")
    print(f"Proximos 7 dias:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).head(7))

    # Changepoints (cambios en tendencia)
    print("\nChangepoints detectados (top 5):")
    changepoints = forecaster.get_changepoints()
    print(changepoints.head())

    return forecaster, forecast


if __name__ == "__main__":
    security_alerting_prophet()
```

---

## Cross-Validation en Prophet

```python
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd


def prophet_cross_validation(
    model: Prophet,
    initial: str = '365 days',
    period: str = '30 days',
    horizon: str = '90 days'
) -> dict:
    """
    Cross-validation temporal para Prophet.

    Args:
        model: Modelo Prophet ajustado
        initial: Periodo inicial de entrenamiento
        period: Espacio entre cutoffs
        horizon: Horizonte de prediccion

    Returns:
        Metricas de performance
    """
    # Cross-validation
    df_cv = cross_validation(
        model,
        initial=initial,
        period=period,
        horizon=horizon
    )

    # Metricas
    df_metrics = performance_metrics(df_cv)

    # Metricas por horizonte
    metrics_by_horizon = df_cv.groupby('horizon').agg({
        'yhat': 'mean',
        'y': 'mean'
    })

    return {
        'cv_results': df_cv,
        'metrics': df_metrics,
        'metrics_by_horizon': metrics_by_horizon,
        'mape': df_metrics['mape'].mean(),
        'rmse': df_metrics['rmse'].mean(),
        'mae': df_metrics['mae'].mean()
    }


def tune_prophet_hyperparameters(
    df: pd.DataFrame,
    param_grid: dict,
    cv_initial: str = '365 days',
    cv_period: str = '30 days',
    cv_horizon: str = '30 days'
) -> dict:
    """
    Busqueda de hiperparametros para Prophet.

    Args:
        df: DataFrame con 'ds' y 'y'
        param_grid: Grid de parametros a probar
        cv_*: Parametros de cross-validation

    Returns:
        Mejores parametros y resultados
    """
    from itertools import product

    # Generar combinaciones
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))

    results = []

    for combo in combinations:
        params = dict(zip(keys, combo))

        try:
            # Crear y ajustar modelo
            model = Prophet(**params)
            model.fit(df)

            # Cross-validation
            cv_result = prophet_cross_validation(
                model,
                initial=cv_initial,
                period=cv_period,
                horizon=cv_horizon
            )

            results.append({
                **params,
                'mape': cv_result['mape'],
                'rmse': cv_result['rmse'],
                'mae': cv_result['mae']
            })

            print(f"Params: {params}, MAPE: {cv_result['mape']:.4f}")

        except Exception as e:
            print(f"Error con params {params}: {e}")
            continue

    # Encontrar mejor
    results_df = pd.DataFrame(results)
    best_idx = results_df['mape'].idxmin()
    best_params = results_df.loc[best_idx].to_dict()

    return {
        'all_results': results_df,
        'best_params': {k: best_params[k] for k in keys},
        'best_mape': best_params['mape']
    }
```

---

## Comparativa de Metodos de Descomposicion

| Metodo | Ventajas | Desventajas | Uso |
|--------|----------|-------------|-----|
| Clasica | Simple, rapida | Rigida, sensible a outliers | Analisis exploratorio |
| STL | Robusta, flexible | Mas parametros | Series con outliers |
| Prophet | Automatica, holidays | Requiere mas datos | Produccion, series complejas |

---

## Aplicacion: Analisis de Trafico de Red

```python
import pandas as pd
import numpy as np
from prophet import Prophet


class NetworkTrafficAnalyzer:
    """
    Analiza y predice trafico de red usando Prophet.
    """

    def __init__(
        self,
        metric_name: str = 'bytes_per_second'
    ):
        self.metric_name = metric_name
        self.model = None
        self.decomposition = None

    def analyze(self, series: pd.Series) -> dict:
        """
        Analiza la serie de trafico.

        Returns:
            Diccionario con analisis completo
        """
        # Preparar datos
        df = pd.DataFrame({'ds': series.index, 'y': series.values})

        # Ajustar Prophet
        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.1
        )
        self.model.fit(df)

        # Prediccion historica para descomposicion
        future = self.model.make_future_dataframe(periods=0, freq='H')
        forecast = self.model.predict(future)

        # Extraer componentes
        self.decomposition = {
            'trend': forecast['trend'],
            'weekly': forecast['weekly'],
            'daily': forecast.get('daily', pd.Series([0]*len(forecast))),
            'residual': series.values - forecast['yhat'].values
        }

        # Analisis
        return {
            'changepoints': self._analyze_changepoints(),
            'daily_pattern': self._get_daily_pattern(forecast),
            'weekly_pattern': self._get_weekly_pattern(forecast),
            'anomalies': self._detect_anomalies(series, forecast),
            'trend_direction': self._get_trend_direction()
        }

    def _analyze_changepoints(self) -> list:
        """Analiza changepoints como posibles incidentes."""
        if self.model is None:
            return []

        changepoints = self.model.changepoints
        deltas = self.model.params['delta'].mean(axis=0)

        significant = []
        for i, (cp, delta) in enumerate(zip(changepoints, deltas)):
            if abs(delta) > 0.01:  # Umbral de significancia
                significant.append({
                    'date': cp,
                    'change': delta,
                    'type': 'increase' if delta > 0 else 'decrease',
                    'magnitude': abs(delta)
                })

        return sorted(significant, key=lambda x: x['magnitude'], reverse=True)

    def _get_daily_pattern(self, forecast: pd.DataFrame) -> dict:
        """Extrae patron diario."""
        if 'daily' not in forecast.columns:
            return {}

        forecast['hour'] = pd.to_datetime(forecast['ds']).dt.hour
        hourly = forecast.groupby('hour')['daily'].mean()

        return {
            'pattern': hourly.to_dict(),
            'peak_hour': hourly.idxmax(),
            'low_hour': hourly.idxmin(),
            'amplitude': hourly.max() - hourly.min()
        }

    def _get_weekly_pattern(self, forecast: pd.DataFrame) -> dict:
        """Extrae patron semanal."""
        forecast['dow'] = pd.to_datetime(forecast['ds']).dt.dayofweek
        daily = forecast.groupby('dow')['weekly'].mean()

        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        return {
            'pattern': {day_names[i]: v for i, v in daily.items()},
            'peak_day': day_names[daily.idxmax()],
            'low_day': day_names[daily.idxmin()],
            'weekday_avg': daily[:5].mean(),
            'weekend_avg': daily[5:].mean()
        }

    def _detect_anomalies(
        self,
        series: pd.Series,
        forecast: pd.DataFrame,
        threshold: float = 2.5
    ) -> list:
        """Detecta anomalias basadas en prediccion."""
        residuals = series.values - forecast['yhat'].values
        std = residuals.std()

        anomalies = []
        for i, (date, actual, predicted, res) in enumerate(zip(
            series.index,
            series.values,
            forecast['yhat'].values,
            residuals
        )):
            z_score = res / std if std > 0 else 0
            if abs(z_score) > threshold:
                anomalies.append({
                    'date': date,
                    'actual': actual,
                    'predicted': predicted,
                    'z_score': z_score,
                    'type': 'spike' if z_score > 0 else 'drop'
                })

        return anomalies

    def _get_trend_direction(self) -> str:
        """Determina direccion de la tendencia."""
        if self.decomposition is None:
            return 'unknown'

        trend = self.decomposition['trend']
        first_quarter = trend[:len(trend)//4].mean()
        last_quarter = trend[-len(trend)//4:].mean()

        change = (last_quarter - first_quarter) / first_quarter * 100

        if change > 5:
            return f'increasing ({change:.1f}%)'
        elif change < -5:
            return f'decreasing ({change:.1f}%)'
        else:
            return 'stable'

    def forecast(self, periods: int = 24) -> pd.DataFrame:
        """Genera prediccion futura."""
        if self.model is None:
            raise ValueError("Modelo no ajustado")

        future = self.model.make_future_dataframe(periods=periods, freq='H')
        forecast = self.model.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)


# Ejemplo de uso
if __name__ == "__main__":
    # Simular trafico de red
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', periods=24*30, freq='H')
    hours = dates.hour
    dow = dates.dayofweek

    # Componentes
    base = 1000
    daily = 300 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
    weekly = 200 * (dow < 5).astype(float)
    noise = np.random.randn(len(dates)) * 50

    traffic = base + daily + weekly + noise

    # Inyectar anomalias
    traffic[100] = 3000  # Spike
    traffic[200] = 200   # Drop

    series = pd.Series(traffic, index=dates)

    # Analizar
    analyzer = NetworkTrafficAnalyzer('bytes_per_second')
    analysis = analyzer.analyze(series)

    print("=== Network Traffic Analysis ===")
    print(f"\nTrend direction: {analysis['trend_direction']}")

    print(f"\nDaily pattern:")
    print(f"  Peak hour: {analysis['daily_pattern'].get('peak_hour', 'N/A')}")
    print(f"  Low hour: {analysis['daily_pattern'].get('low_hour', 'N/A')}")

    print(f"\nWeekly pattern:")
    print(f"  Peak day: {analysis['weekly_pattern']['peak_day']}")
    print(f"  Weekday avg: {analysis['weekly_pattern']['weekday_avg']:.0f}")
    print(f"  Weekend avg: {analysis['weekly_pattern']['weekend_avg']:.0f}")

    print(f"\nChangepoints: {len(analysis['changepoints'])}")
    for cp in analysis['changepoints'][:3]:
        print(f"  {cp['date']}: {cp['type']} ({cp['magnitude']:.4f})")

    print(f"\nAnomalies: {len(analysis['anomalies'])}")
    for anom in analysis['anomalies'][:3]:
        print(f"  {anom['date']}: {anom['type']} (z={anom['z_score']:.2f})")

    # Forecast
    print("\nForecast (next 6 hours):")
    forecast = analyzer.forecast(6)
    print(forecast)
```

---

## Resumen

```
SELECCION DE METODO DE DESCOMPOSICION
=====================================

Clasica:
├── Datos: Simples, sin outliers
├── Uso: Analisis exploratorio rapido
└── Limitaciones: Estacionalidad fija

STL:
├── Datos: Con outliers, estacionalidad variable
├── Uso: Analisis robusto
└── Ventajas: Flexible, detecta outliers

Prophet:
├── Datos: Series complejas, multiples estacionalidades
├── Uso: Produccion, forecasting automatico
├── Ventajas: Holidays, changepoints, regresores
└── Limitaciones: Requiere suficientes datos


PROPHET CHEATSHEET
==================

1. Preparar datos: df['ds'], df['y']

2. Estacionalidades:
   - yearly_seasonality=True/False
   - weekly_seasonality=True/False
   - add_seasonality(name, period, fourier_order)

3. Holidays: add_holidays(df_holidays)

4. Regresores: add_regressor(name)

5. Tuning:
   - changepoint_prior_scale: 0.01-0.5 (flexibilidad tendencia)
   - seasonality_prior_scale: 0.01-10 (flexibilidad estacionalidad)
```

### Puntos Clave

1. **Descomposicion** separa tendencia, estacionalidad y residuo
2. **STL** es robusto a outliers
3. **Prophet** automatiza el proceso y maneja casos complejos
4. **Changepoints** pueden indicar incidentes de seguridad
5. **Cross-validation temporal** es esencial para evaluar
6. En ciberseguridad: ideal para metricas con patrones conocidos
