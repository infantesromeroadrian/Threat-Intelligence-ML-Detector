# Deteccion de Anomalias en Series Temporales

## Introduccion

La **deteccion de anomalias temporales** identifica observaciones que se desvian significativamente del comportamiento esperado de una serie temporal. Es fundamental en ciberseguridad para detectar intrusiones, ataques DDoS, exfiltracion de datos y comportamiento anomalo.

```
TIPOS DE ANOMALIAS TEMPORALES
=============================

1. POINT ANOMALY (Puntual)
   Un solo punto fuera de lo normal

       │        ●
       │  ~~~~~~~~~~~
       │
       └─────────────────

2. CONTEXTUAL ANOMALY
   Normal en otro contexto, anomalo aqui

       │         (normal a las 3am,
       │  ~~~●~~~ anomalo a las 3pm)
       │
       └─────────────────

3. COLLECTIVE ANOMALY (Patron)
   Secuencia de puntos que juntos son anomalos

       │     ────────
       │  ~~~        ~~~
       │
       └─────────────────
       (cada punto OK, la secuencia no)

4. SEASONAL ANOMALY
   Desviacion del patron estacional esperado

       │  /\  /\  __  /\
       │ /  \/  \/  \/  \  ← patron roto
       └─────────────────
```

## Metodos Estadisticos

### Z-Score (Standard Score)

```
Z-SCORE
=======

z = (x - μ) / σ

Anomalia si |z| > threshold (tipicamente 2.5-3)


Con ventana movil (mejor para series temporales):

    μ_t = mean(x_{t-w}, ..., x_{t-1})
    σ_t = std(x_{t-w}, ..., x_{t-1})
    z_t = (x_t - μ_t) / σ_t


    │      ●
    │  ~~~~│~~~~ μ + 3σ
    │      │
    │ ─────┼───── μ
    │      │
    │  ~~~~│~~~~ μ - 3σ
    └──────┴─────────
          anomalia
```

### Implementacion

```python
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class Anomaly:
    """Representa una anomalia detectada."""
    timestamp: pd.Timestamp
    value: float
    expected: float
    score: float
    anomaly_type: str
    severity: str


class ZScoreDetector:
    """
    Detector de anomalias basado en Z-score con ventana movil.
    """

    def __init__(
        self,
        window_size: int = 24,
        threshold: float = 3.0,
        min_periods: int = 10
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.min_periods = min_periods

    def fit_detect(self, series: pd.Series) -> pd.DataFrame:
        """
        Detecta anomalias en la serie.

        Args:
            series: Serie temporal con indice datetime

        Returns:
            DataFrame con scores y flags de anomalia
        """
        # Estadisticas moviles
        rolling_mean = series.rolling(
            window=self.window_size,
            min_periods=self.min_periods
        ).mean()

        rolling_std = series.rolling(
            window=self.window_size,
            min_periods=self.min_periods
        ).std()

        # Z-score
        z_score = (series - rolling_mean) / rolling_std

        # Detectar anomalias
        is_anomaly = abs(z_score) > self.threshold

        return pd.DataFrame({
            'value': series,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'z_score': z_score,
            'is_anomaly': is_anomaly,
            'anomaly_type': np.where(
                z_score > self.threshold, 'spike',
                np.where(z_score < -self.threshold, 'drop', 'normal')
            )
        })

    def get_anomalies(
        self,
        series: pd.Series
    ) -> List[Anomaly]:
        """Retorna lista de anomalias detectadas."""
        result = self.fit_detect(series)
        anomalies = []

        for idx, row in result[result['is_anomaly']].iterrows():
            severity = (
                'critical' if abs(row['z_score']) > 4 else
                'high' if abs(row['z_score']) > 3.5 else
                'medium'
            )

            anomalies.append(Anomaly(
                timestamp=idx,
                value=row['value'],
                expected=row['rolling_mean'],
                score=row['z_score'],
                anomaly_type=row['anomaly_type'],
                severity=severity
            ))

        return anomalies


class MADDetector:
    """
    Detector basado en Median Absolute Deviation.
    Mas robusto a outliers que Z-score.
    """

    def __init__(
        self,
        window_size: int = 24,
        threshold: float = 3.5,
        min_periods: int = 10
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.min_periods = min_periods
        self.k = 1.4826  # Factor para distribucion normal

    def fit_detect(self, series: pd.Series) -> pd.DataFrame:
        """Detecta anomalias usando MAD."""
        # Mediana movil
        rolling_median = series.rolling(
            window=self.window_size,
            min_periods=self.min_periods
        ).median()

        # MAD movil
        deviation = (series - rolling_median).abs()
        rolling_mad = deviation.rolling(
            window=self.window_size,
            min_periods=self.min_periods
        ).median()

        # Modified Z-score
        modified_z = self.k * (series - rolling_median) / rolling_mad

        # Detectar
        is_anomaly = abs(modified_z) > self.threshold

        return pd.DataFrame({
            'value': series,
            'rolling_median': rolling_median,
            'rolling_mad': rolling_mad,
            'modified_z': modified_z,
            'is_anomaly': is_anomaly
        })
```

### IQR (Interquartile Range)

```python
class IQRDetector:
    """
    Detector basado en rango intercuartilico.
    """

    def __init__(
        self,
        window_size: int = 24,
        k: float = 1.5  # 1.5 para outliers, 3 para extremos
    ):
        self.window_size = window_size
        self.k = k

    def fit_detect(self, series: pd.Series) -> pd.DataFrame:
        """Detecta anomalias usando IQR."""
        rolling = series.rolling(window=self.window_size)

        q1 = rolling.quantile(0.25)
        q3 = rolling.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.k * iqr
        upper_bound = q3 + self.k * iqr

        is_anomaly = (series < lower_bound) | (series > upper_bound)

        return pd.DataFrame({
            'value': series,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'is_anomaly': is_anomaly
        })
```

---

## Metodos Basados en Forecasting

### Anomalia = Desviacion del Forecast

```
DETECCION BASADA EN PREDICCION
==============================

1. Entrenar modelo de forecasting
2. Predecir siguiente valor
3. Comparar prediccion vs real
4. Si |real - predicho| > threshold → anomalia

    │
    │      ● real (anomalia!)
    │  ────○──────────────────
    │      predicho
    │
    └─────────────────────────

Threshold = k * σ_residuos  (tipicamente k=2.5-3)
```

### Implementacion

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


class ForecastAnomalyDetector:
    """
    Detector de anomalias basado en error de prediccion.
    """

    def __init__(
        self,
        model_type: str = 'ets',
        seasonal_period: int = 24,
        threshold_std: float = 3.0
    ):
        self.model_type = model_type
        self.seasonal_period = seasonal_period
        self.threshold_std = threshold_std
        self.model = None
        self.residual_std = None

    def fit(self, series: pd.Series) -> 'ForecastAnomalyDetector':
        """Entrena el modelo."""
        if self.model_type == 'ets':
            self.model = ExponentialSmoothing(
                series,
                seasonal_periods=self.seasonal_period,
                trend='add',
                seasonal='add',
                damped_trend=True
            ).fit()
        else:  # arima
            self.model = ARIMA(
                series,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, self.seasonal_period)
            ).fit()

        # Calcular std de residuos
        self.residual_std = self.model.resid.std()

        return self

    def detect_point(
        self,
        actual: float,
        timestamp: pd.Timestamp | None = None
    ) -> dict:
        """
        Detecta si un punto es anomalo.

        Args:
            actual: Valor observado
            timestamp: Para context-aware detection

        Returns:
            Diccionario con resultado
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        # Predecir
        forecast = self.model.forecast(1)
        predicted = forecast.iloc[0]

        # Calcular error
        error = actual - predicted
        z_score = error / self.residual_std if self.residual_std > 0 else 0

        is_anomaly = abs(z_score) > self.threshold_std

        return {
            'actual': actual,
            'predicted': predicted,
            'error': error,
            'z_score': z_score,
            'is_anomaly': is_anomaly,
            'anomaly_type': (
                'spike' if z_score > self.threshold_std else
                'drop' if z_score < -self.threshold_std else
                'normal'
            ),
            'confidence_interval': (
                predicted - self.threshold_std * self.residual_std,
                predicted + self.threshold_std * self.residual_std
            )
        }

    def detect_series(self, series: pd.Series) -> pd.DataFrame:
        """Detecta anomalias en toda la serie."""
        results = []

        for i in range(len(series)):
            if i < self.seasonal_period:
                continue

            # Reentrenar con datos hasta i-1
            train = series.iloc[:i]
            self.fit(train)

            result = self.detect_point(series.iloc[i])
            result['timestamp'] = series.index[i]
            results.append(result)

        return pd.DataFrame(results)
```

---

## Isolation Forest Temporal

### Concepto

```
ISOLATION FOREST PARA TIME SERIES
=================================

Isolation Forest asume: anomalias son faciles de aislar.

Adaptacion para series temporales:
- Features = ventana de valores pasados
- Cada punto se representa por su contexto temporal

Ventana = 5:
    Serie: [1, 2, 3, 4, 5, 6, 7, 100, 9, 10]

    Punto 8 (valor=100):
    Features = [4, 5, 6, 7, 100]
                          ↑
                       Facil de aislar!

Punto normal:
    Features = [5, 6, 7, 8, 9]
    Dificil de aislar (similar a otros)
```

### Implementacion

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class TemporalIsolationForest:
    """
    Isolation Forest adaptado para series temporales.
    """

    def __init__(
        self,
        window_size: int = 24,
        contamination: float = 0.01,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.window_size = window_size
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def _create_features(self, series: pd.Series) -> np.ndarray:
        """
        Crea features de ventana temporal.
        Incluye el valor actual y estadisticas de la ventana.
        """
        n = len(series)
        features = []

        for i in range(self.window_size, n):
            window = series.iloc[i - self.window_size:i].values
            current = series.iloc[i]

            # Features
            feat = [
                current,                          # Valor actual
                np.mean(window),                  # Media ventana
                np.std(window),                   # Std ventana
                np.min(window),                   # Min ventana
                np.max(window),                   # Max ventana
                current - np.mean(window),        # Desviacion de media
                (current - np.mean(window)) / (np.std(window) + 1e-8),  # Z-score
                np.percentile(window, 25),        # Q1
                np.percentile(window, 75),        # Q3
                series.iloc[i] - series.iloc[i-1],  # Diferencia
            ]

            features.append(feat)

        return np.array(features)

    def fit(self, series: pd.Series) -> 'TemporalIsolationForest':
        """Entrena el modelo."""
        features = self._create_features(series)
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        return self

    def predict(self, series: pd.Series) -> pd.DataFrame:
        """
        Predice anomalias.

        Returns:
            DataFrame con scores y labels
        """
        features = self._create_features(series)
        features_scaled = self.scaler.transform(features)

        # Scores (mas negativo = mas anomalo)
        scores = self.model.decision_function(features_scaled)

        # Labels (-1 = anomalia, 1 = normal)
        labels = self.model.predict(features_scaled)

        # Crear DataFrame alineado con serie original
        result_index = series.index[self.window_size:]

        return pd.DataFrame({
            'value': series.iloc[self.window_size:].values,
            'anomaly_score': scores,
            'is_anomaly': labels == -1,
            'anomaly_probability': 1 - (scores - scores.min()) / (scores.max() - scores.min())
        }, index=result_index)


class StreamingIsolationForest:
    """
    Isolation Forest para deteccion en streaming.
    Re-entrena periodicamente con ventana deslizante.
    """

    def __init__(
        self,
        window_size: int = 24,
        training_size: int = 1000,
        retrain_interval: int = 100,
        contamination: float = 0.01
    ):
        self.window_size = window_size
        self.training_size = training_size
        self.retrain_interval = retrain_interval
        self.contamination = contamination

        self.buffer = []
        self.detector = TemporalIsolationForest(
            window_size=window_size,
            contamination=contamination
        )
        self.samples_since_retrain = 0
        self.is_trained = False

    def update(self, value: float, timestamp: pd.Timestamp) -> dict | None:
        """
        Procesa nuevo valor.

        Args:
            value: Nuevo valor observado
            timestamp: Timestamp del valor

        Returns:
            Resultado de deteccion o None si no hay suficientes datos
        """
        self.buffer.append((timestamp, value))

        # Mantener buffer limitado
        if len(self.buffer) > self.training_size:
            self.buffer = self.buffer[-self.training_size:]

        # Verificar si hay suficientes datos
        if len(self.buffer) < self.window_size + 10:
            return None

        # Re-entrenar si es necesario
        self.samples_since_retrain += 1
        if (not self.is_trained or
            self.samples_since_retrain >= self.retrain_interval):

            series = pd.Series(
                [v for _, v in self.buffer],
                index=[t for t, _ in self.buffer]
            )
            self.detector.fit(series)
            self.is_trained = True
            self.samples_since_retrain = 0

        # Detectar en ultimo punto
        series = pd.Series(
            [v for _, v in self.buffer],
            index=[t for t, _ in self.buffer]
        )
        result = self.detector.predict(series)

        if len(result) > 0:
            last = result.iloc[-1]
            return {
                'timestamp': timestamp,
                'value': value,
                'anomaly_score': last['anomaly_score'],
                'is_anomaly': last['is_anomaly'],
                'probability': last['anomaly_probability']
            }

        return None
```

---

## Autoencoders para Deteccion de Anomalias

### Concepto

```
AUTOENCODER TEMPORAL
====================

Entrenamiento: Solo con datos NORMALES

    Secuencia Normal → Encoder → z → Decoder → Reconstruccion
                                               (error bajo)

Inferencia:

    Secuencia Normal → ... → Reconstruccion similar
                             Error bajo → OK

    Secuencia Anomala → ... → Reconstruccion diferente
                              Error ALTO → ANOMALIA!


          │ Reconstruction Error
          │
          │        ●● anomalias
          │
          │─────────── threshold
          │
          │ ●●●●●●●●● normales
          └─────────────────────
```

### Implementacion

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder para deteccion de anomalias en secuencias.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        seq_len: int = 24
    ):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            reconstruction: [batch, seq_len, input_size]
        """
        batch_size = x.size(0)

        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Preparar input para decoder
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decode
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        # Output
        reconstruction = self.output_layer(decoder_output)

        return reconstruction


class AutoencoderAnomalyDetector:
    """
    Detector de anomalias basado en LSTM Autoencoder.
    """

    def __init__(
        self,
        seq_len: int = 24,
        hidden_size: int = 64,
        threshold_percentile: float = 99.0
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.threshold_percentile = threshold_percentile

        self.model = LSTMAutoencoder(
            input_size=1,
            hidden_size=hidden_size,
            seq_len=seq_len
        )
        self.threshold = None
        self.scaler = None

    def _create_sequences(self, series: pd.Series) -> np.ndarray:
        """Crea secuencias de ventana deslizante."""
        data = series.values.reshape(-1, 1)
        sequences = []

        for i in range(len(data) - self.seq_len + 1):
            sequences.append(data[i:i + self.seq_len])

        return np.array(sequences)

    def fit(
        self,
        series: pd.Series,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3
    ) -> 'AutoencoderAnomalyDetector':
        """
        Entrena autoencoder con datos normales.
        """
        # Normalizar
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        normalized = self.scaler.fit_transform(series.values.reshape(-1, 1))
        normalized_series = pd.Series(normalized.flatten(), index=series.index)

        # Crear secuencias
        sequences = self._create_sequences(normalized_series)
        X = torch.FloatTensor(sequences)

        # Dataset
        dataset = torch.utils.data.TensorDataset(X, X)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Entrenar
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0

            for batch_x, _ in loader:
                optimizer.zero_grad()
                reconstruction = self.model(batch_x)
                loss = criterion(reconstruction, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}")

        # Calcular threshold
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X)
            errors = ((X - reconstructions) ** 2).mean(dim=(1, 2)).numpy()
            self.threshold = np.percentile(errors, self.threshold_percentile)

        print(f"Threshold: {self.threshold:.6f}")

        return self

    def predict(self, series: pd.Series) -> pd.DataFrame:
        """
        Detecta anomalias en la serie.
        """
        # Normalizar
        normalized = self.scaler.transform(series.values.reshape(-1, 1))
        normalized_series = pd.Series(normalized.flatten(), index=series.index)

        # Crear secuencias
        sequences = self._create_sequences(normalized_series)
        X = torch.FloatTensor(sequences)

        # Reconstruir
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X)
            errors = ((X - reconstructions) ** 2).mean(dim=(1, 2)).numpy()

        # Alinear con indices originales
        result_index = series.index[self.seq_len - 1:]

        return pd.DataFrame({
            'value': series.iloc[self.seq_len - 1:].values,
            'reconstruction_error': errors,
            'is_anomaly': errors > self.threshold,
            'anomaly_score': errors / self.threshold
        }, index=result_index)
```

---

## Deteccion de Anomalias Contextuales

### Por Hora del Dia / Dia de la Semana

```python
class ContextualAnomalyDetector:
    """
    Detector que considera el contexto temporal.
    Un valor puede ser normal a las 3am pero anomalo a las 3pm.
    """

    def __init__(
        self,
        threshold_std: float = 3.0,
        min_samples: int = 10
    ):
        self.threshold_std = threshold_std
        self.min_samples = min_samples
        self.context_stats = {}

    def fit(self, series: pd.Series) -> 'ContextualAnomalyDetector':
        """
        Aprende estadisticas por contexto temporal.
        """
        df = pd.DataFrame({
            'value': series,
            'hour': series.index.hour,
            'dayofweek': series.index.dayofweek
        })

        # Calcular estadisticas por (hora, dia_semana)
        for (hour, dow), group in df.groupby(['hour', 'dayofweek']):
            if len(group) >= self.min_samples:
                self.context_stats[(hour, dow)] = {
                    'mean': group['value'].mean(),
                    'std': group['value'].std(),
                    'median': group['value'].median(),
                    'q1': group['value'].quantile(0.25),
                    'q3': group['value'].quantile(0.75),
                    'count': len(group)
                }

        return self

    def predict(self, series: pd.Series) -> pd.DataFrame:
        """
        Detecta anomalias considerando contexto.
        """
        results = []

        for timestamp, value in series.items():
            hour = timestamp.hour
            dow = timestamp.dayofweek
            context = (hour, dow)

            if context in self.context_stats:
                stats = self.context_stats[context]
                z_score = (value - stats['mean']) / (stats['std'] + 1e-8)

                is_anomaly = abs(z_score) > self.threshold_std

                results.append({
                    'timestamp': timestamp,
                    'value': value,
                    'context': f"hour={hour}, dow={dow}",
                    'expected_mean': stats['mean'],
                    'expected_std': stats['std'],
                    'z_score': z_score,
                    'is_anomaly': is_anomaly
                })
            else:
                # Contexto desconocido
                results.append({
                    'timestamp': timestamp,
                    'value': value,
                    'context': f"hour={hour}, dow={dow}",
                    'expected_mean': None,
                    'expected_std': None,
                    'z_score': None,
                    'is_anomaly': False
                })

        return pd.DataFrame(results)
```

---

## Pipeline Completo de Deteccion

```python
import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum


class AnomalyType(Enum):
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    COLLECTIVE = "collective"


@dataclass
class DetectedAnomaly:
    timestamp: pd.Timestamp
    value: float
    expected: float
    score: float
    anomaly_type: AnomalyType
    severity: str
    detector: str
    confidence: float


class EnsembleAnomalyDetector:
    """
    Ensemble de multiples detectores para robustez.
    """

    def __init__(
        self,
        window_size: int = 24,
        seasonal_period: int = 24
    ):
        self.window_size = window_size
        self.seasonal_period = seasonal_period

        # Inicializar detectores
        self.detectors = {
            'zscore': ZScoreDetector(window_size=window_size),
            'mad': MADDetector(window_size=window_size),
            'iqr': IQRDetector(window_size=window_size),
            'isolation_forest': TemporalIsolationForest(window_size=window_size),
            'contextual': ContextualAnomalyDetector()
        }

        self.is_fitted = False

    def fit(self, series: pd.Series) -> 'EnsembleAnomalyDetector':
        """Entrena todos los detectores."""
        # Isolation Forest necesita fit
        self.detectors['isolation_forest'].fit(series)

        # Contextual necesita fit
        self.detectors['contextual'].fit(series)

        self.is_fitted = True
        return self

    def detect(
        self,
        series: pd.Series,
        voting_threshold: int = 2
    ) -> pd.DataFrame:
        """
        Detecta anomalias con votacion.

        Args:
            series: Serie a analizar
            voting_threshold: Minimo de detectores que deben coincidir

        Returns:
            DataFrame con anomalias y votos
        """
        if not self.is_fitted:
            self.fit(series)

        # Ejecutar cada detector
        results = {}

        results['zscore'] = self.detectors['zscore'].fit_detect(series)['is_anomaly']
        results['mad'] = self.detectors['mad'].fit_detect(series)['is_anomaly']
        results['iqr'] = self.detectors['iqr'].fit_detect(series)['is_anomaly']

        if_result = self.detectors['isolation_forest'].predict(series)
        results['isolation_forest'] = pd.Series(
            if_result['is_anomaly'].values,
            index=if_result.index
        ).reindex(series.index, fill_value=False)

        ctx_result = self.detectors['contextual'].predict(series)
        results['contextual'] = pd.Series(
            ctx_result['is_anomaly'].values,
            index=pd.to_datetime(ctx_result['timestamp'])
        ).reindex(series.index, fill_value=False)

        # Combinar votos
        votes = pd.DataFrame(results)
        votes['total_votes'] = votes.sum(axis=1)
        votes['is_anomaly'] = votes['total_votes'] >= voting_threshold
        votes['value'] = series

        # Determinar tipo de anomalia
        zscore_result = self.detectors['zscore'].fit_detect(series)
        votes['anomaly_type'] = np.where(
            votes['is_anomaly'],
            zscore_result['anomaly_type'],
            'normal'
        )

        return votes

    def get_anomalies(
        self,
        series: pd.Series,
        voting_threshold: int = 2
    ) -> List[DetectedAnomaly]:
        """Retorna lista de anomalias detectadas."""
        result = self.detect(series, voting_threshold)
        anomalies = []

        zscore_result = self.detectors['zscore'].fit_detect(series)

        for idx, row in result[result['is_anomaly']].iterrows():
            # Determinar severidad por votos
            severity = (
                'critical' if row['total_votes'] >= 4 else
                'high' if row['total_votes'] >= 3 else
                'medium'
            )

            # Detectores que votaron
            voters = [
                det for det in ['zscore', 'mad', 'iqr', 'isolation_forest', 'contextual']
                if row[det]
            ]

            anomalies.append(DetectedAnomaly(
                timestamp=idx,
                value=row['value'],
                expected=zscore_result.loc[idx, 'rolling_mean'],
                score=zscore_result.loc[idx, 'z_score'],
                anomaly_type=AnomalyType(row['anomaly_type']) if row['anomaly_type'] != 'normal' else AnomalyType.SPIKE,
                severity=severity,
                detector=', '.join(voters),
                confidence=row['total_votes'] / 5
            ))

        return anomalies


# Ejemplo de uso completo
def security_anomaly_detection_demo():
    """
    Demo completo de deteccion de anomalias para ciberseguridad.
    """
    np.random.seed(42)

    # Simular metricas de red (requests por minuto)
    n_points = 24 * 60 * 7  # 1 semana de datos por minuto
    timestamps = pd.date_range('2024-01-01', periods=n_points, freq='T')

    # Patron normal
    hours = timestamps.hour + timestamps.minute / 60
    daily_pattern = 100 + 50 * np.sin(2 * np.pi * hours / 24)
    noise = np.random.randn(n_points) * 10

    traffic = daily_pattern + noise

    # Inyectar anomalias
    # 1. Spike (posible ataque DDoS)
    traffic[1000:1010] = 500

    # 2. Drop (posible caida de servicio)
    traffic[3000:3005] = 10

    # 3. Anomalia contextual (trafico alto a las 3am)
    night_indices = np.where((timestamps.hour >= 2) & (timestamps.hour <= 4))[0]
    traffic[night_indices[100:110]] = 200

    # 4. Trend change
    traffic[5000:5500] = traffic[5000:5500] + np.linspace(0, 100, 500)

    series = pd.Series(traffic, index=timestamps)

    # Deteccion
    print("=== Security Anomaly Detection Demo ===\n")

    detector = EnsembleAnomalyDetector(
        window_size=60,  # 1 hora
        seasonal_period=60 * 24  # 1 dia
    )

    anomalies = detector.get_anomalies(series, voting_threshold=2)

    print(f"Total anomalias detectadas: {len(anomalies)}\n")

    # Agrupar por severidad
    by_severity = {}
    for a in anomalies:
        by_severity.setdefault(a.severity, []).append(a)

    for severity in ['critical', 'high', 'medium']:
        if severity in by_severity:
            print(f"{severity.upper()}: {len(by_severity[severity])} anomalias")
            for a in by_severity[severity][:3]:
                print(f"  - {a.timestamp}: valor={a.value:.1f}, "
                      f"esperado={a.expected:.1f}, tipo={a.anomaly_type.value}")

    return detector, anomalies


if __name__ == "__main__":
    security_anomaly_detection_demo()
```

---

## Comparativa de Metodos

| Metodo | Ventajas | Desventajas | Uso |
|--------|----------|-------------|-----|
| Z-Score | Simple, rapido | Sensible a outliers | Baseline |
| MAD | Robusto | Menos preciso | Series con outliers |
| IQR | No asume normalidad | Menos sensible | General |
| Isolation Forest | No parametrico | Necesita features | Multivariado |
| Autoencoder | Aprende patrones complejos | Requiere datos | Secuencias |
| Contextual | Considera tiempo | Necesita historia | Patrones diarios |
| Ensemble | Robusto, reduce FP | Mas lento | Produccion |

---

## Resumen

```
PIPELINE DE DETECCION
=====================

1. PREPROCESAMIENTO
   - Normalizar datos
   - Manejar missing values
   - Alinear timestamps

2. SELECCION DE METODO
   - Baseline: Z-Score o MAD
   - Multivariado: Isolation Forest
   - Secuencias: LSTM Autoencoder
   - Contextual: Por hora/dia

3. TUNING DE THRESHOLD
   - Muy bajo: Muchos falsos positivos
   - Muy alto: Muchos falsos negativos
   - Usar curva precision-recall

4. ENSEMBLE (PRODUCCION)
   - Combinar multiples detectores
   - Votacion para reducir FP
   - Diferentes metodos capturan diferentes anomalias

5. ALERTING
   - Severidad basada en score/votos
   - Contexto (hora, tendencia)
   - Agregacion para evitar fatiga


METRICAS DE EVALUACION
======================

- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 * P * R / (P + R)
- AUC-PR: Area bajo curva Precision-Recall

En ciberseguridad: Preferir alto RECALL
(mejor detectar anomalia real aunque haya algunos FP)
```

### Puntos Clave

1. **Z-Score** es buen baseline, pero sensible a outliers
2. **MAD** es mas robusto para datos con outliers
3. **Isolation Forest** funciona bien en alta dimension
4. **Autoencoders** aprenden patrones normales complejos
5. **Contexto temporal** es crucial (3am vs 3pm)
6. **Ensemble** reduce falsos positivos
7. En produccion: usar **votacion** de multiples metodos
