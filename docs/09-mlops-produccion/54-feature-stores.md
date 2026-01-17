# Feature Stores

## 1. Que son los Feature Stores

### El Problema sin Feature Store

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PROBLEMA: FEATURE ENGINEERING CAÓTICO                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ESCENARIO TÍPICO:                                                          │
│                                                                             │
│  Data Scientist A (Fraud Detection):                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  # En notebook                                                      │   │
│  │  df['avg_txn_7d'] = df.groupby('user_id')['amount']                │   │
│  │                       .rolling(7).mean()                            │   │
│  │  df['num_devices'] = df.groupby('user_id')['device_id'].nunique()  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Data Scientist B (Risk Scoring):                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  # En otro notebook, misma feature diferente nombre                 │   │
│  │  df['txn_avg_weekly'] = df.groupby('user')['amt']                  │   │
│  │                           .transform(lambda x: x.rolling(7).mean()) │   │
│  │  df['device_count'] = ...  # Implementacion ligeramente diferente   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ML Engineer (Production):                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  # Reimplementa features en Java/Scala para Spark                   │   │
│  │  // avg_txn_7d - pero con bugs sutiles                             │   │
│  │  // Latencia inaceptable para tiempo real                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  PROBLEMAS:                                                                 │
│  ✗ Duplicación de código                                                    │
│  ✗ Inconsistencia entre entrenamiento y producción (Training-Serving Skew) │
│  ✗ Sin reutilización entre equipos                                          │
│  ✗ Imposible auditar qué features se usaron                                 │
│  ✗ Data leakage difícil de detectar                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Qué es un Feature Store

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE STORE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          ┌─────────────────────┐                            │
│                          │   Feature Store     │                            │
│                          │                     │                            │
│                          │  ┌───────────────┐  │                            │
│  Data Sources ──────────►│  │ Feature       │  │                            │
│  • Databases             │  │ Registry      │  │                            │
│  • Streams               │  │ (Metadata)    │  │                            │
│  • Data Lakes            │  └───────────────┘  │                            │
│                          │         │           │                            │
│                          │         ▼           │                            │
│                          │  ┌───────────────┐  │                            │
│                          │  │ Offline Store │◄─┼──── Training               │
│                          │  │ (Batch)       │  │      (Historical)          │
│                          │  └───────────────┘  │                            │
│                          │         │           │                            │
│                          │         ▼           │                            │
│                          │  ┌───────────────┐  │                            │
│                          │  │ Online Store  │◄─┼──── Serving                │
│                          │  │ (Real-time)   │  │      (Low latency)         │
│                          │  └───────────────┘  │                            │
│                          │                     │                            │
│                          └─────────────────────┘                            │
│                                                                             │
│  COMPONENTES CLAVE:                                                         │
│  ───────────────────────────────────────────────────────────────────────   │
│  1. Feature Registry: Catálogo con metadata, linaje, versiones             │
│  2. Offline Store: Features históricas para entrenamiento (data lake)      │
│  3. Online Store: Features actuales para inferencia (Redis, DynamoDB)      │
│  4. Feature Engineering: Pipelines para calcular y actualizar features     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Beneficios

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BENEFICIOS DEL FEATURE STORE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. REUTILIZACIÓN                                                           │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Feature: avg_transaction_7d                                    │    │
│     │                                                                 │    │
│     │  Usado por:                                                     │    │
│     │  • Modelo de fraude (v3)                                        │    │
│     │  • Modelo de riesgo (v2)                                        │    │
│     │  • Modelo de churn (v1)                                         │    │
│     │                                                                 │    │
│     │  Definición ÚNICA, reutilización MÚLTIPLE                       │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. CONSISTENCIA TRAINING-SERVING                                           │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Training: get_historical_features(entity_df, features)         │    │
│     │  Serving:  get_online_features(entity_key, features)            │    │
│     │                                                                 │    │
│     │  MISMA lógica de cálculo → SIN training-serving skew            │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  3. POINT-IN-TIME CORRECTNESS                                               │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Para cada evento en t=T, el feature store retorna              │    │
│     │  los valores de features que existían en t=T                    │    │
│     │                                                                 │    │
│     │  → Evita data leakage automáticamente                           │    │
│     │  → Reproducibilidad garantizada                                 │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  4. DESCUBRIMIENTO Y GOBERNANZA                                             │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  • Catálogo centralizado de features                            │    │
│     │  • Linaje de datos (de dónde viene cada feature)                │    │
│     │  • Estadísticas de uso                                          │    │
│     │  • Control de acceso por equipo                                 │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Feast: Feature Store Open Source

### Arquitectura de Feast

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEAST ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        FEAST COMPONENTS                             │   │
│  │                                                                     │   │
│  │   ┌────────────────┐                                               │   │
│  │   │ Feature Repo   │ ← Definiciones en Python                     │   │
│  │   │ (feature_store │                                               │   │
│  │   │  .yaml)        │                                               │   │
│  │   └───────┬────────┘                                               │   │
│  │           │                                                         │   │
│  │           ▼                                                         │   │
│  │   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐  │   │
│  │   │ Feast Registry │───►│ Offline Store  │    │ Online Store   │  │   │
│  │   │ (SQLite/SQL)   │    │ (Parquet/BQ/   │    │ (Redis/        │  │   │
│  │   │                │    │  Snowflake)    │    │  DynamoDB)     │  │   │
│  │   └────────────────┘    └────────────────┘    └────────────────┘  │   │
│  │           │                     │                     │           │   │
│  │           └─────────────────────┴─────────────────────┘           │   │
│  │                                 │                                  │   │
│  │                                 ▼                                  │   │
│  │                    ┌────────────────────────┐                     │   │
│  │                    │   Feast Python SDK    │                     │   │
│  │                    │   • get_historical_   │                     │   │
│  │                    │     features()        │                     │   │
│  │                    │   • get_online_       │                     │   │
│  │                    │     features()        │                     │   │
│  │                    │   • materialize()     │                     │   │
│  │                    └────────────────────────┘                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  FLUJO DE DATOS:                                                            │
│  ───────────────────────────────────────────────────────────────────────   │
│  1. Definir features (Python)                                               │
│  2. feast apply → Registrar en Registry                                    │
│  3. feast materialize → Poblar Online Store desde Offline                  │
│  4. get_historical_features() → Training                                   │
│  5. get_online_features() → Serving                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Setup de Feast

```python
"""
Configuración inicial de Feast para proyecto de seguridad.
"""
import os
from pathlib import Path


def create_feast_project(project_path: str = "./feature_repo") -> None:
    """
    Crea estructura de proyecto Feast.

    Args:
        project_path: Directorio del proyecto
    """
    Path(project_path).mkdir(exist_ok=True)

    # feature_store.yaml - Configuración principal
    feature_store_yaml = """
project: security_features
registry: data/registry.db
provider: local

offline_store:
    type: file

online_store:
    type: redis
    connection_string: "localhost:6379"

entity_key_serialization_version: 2
"""

    with open(f"{project_path}/feature_store.yaml", "w") as f:
        f.write(feature_store_yaml)

    print(f"Feast project created at {project_path}")
    print("Run 'feast apply' to initialize")


# Para producción con GCP/AWS
def create_feast_project_production() -> str:
    """Configuración para producción."""

    # GCP BigQuery + Redis
    gcp_config = """
project: security_features_prod
registry: gs://my-bucket/registry.db
provider: gcp

offline_store:
    type: bigquery
    project: my-gcp-project
    dataset: feast_features

online_store:
    type: redis
    connection_string: "10.0.0.5:6379"
    redis_type: redis_cluster
"""

    # AWS Redshift + DynamoDB
    aws_config = """
project: security_features_prod
registry: s3://my-bucket/registry.db
provider: aws

offline_store:
    type: redshift
    cluster_id: my-cluster
    region: us-east-1
    database: feast
    user: feast_user
    s3_staging_location: s3://my-bucket/staging/

online_store:
    type: dynamodb
    region: us-east-1
"""

    return gcp_config  # o aws_config
```

### Definición de Features

```python
"""
Definiciones de features para detección de amenazas.

feature_definitions.py - Colocar en feature_repo/
"""
from datetime import timedelta
from feast import (
    Entity,
    Feature,
    FeatureView,
    FileSource,
    Field,
    ValueType,
    BatchFeatureView,
    StreamFeatureView,
)
from feast.types import Float32, Int64, String, UnixTimestamp
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource
)


# =============================================================================
# ENTITIES - Identificadores únicos
# =============================================================================

# Entidad: Usuario
user_entity = Entity(
    name="user_id",
    description="Identificador único de usuario",
    join_keys=["user_id"],
    value_type=ValueType.STRING,
)

# Entidad: IP Address
ip_entity = Entity(
    name="ip_address",
    description="Dirección IP de origen",
    join_keys=["ip_address"],
    value_type=ValueType.STRING,
)

# Entidad: Session
session_entity = Entity(
    name="session_id",
    description="Identificador de sesión",
    join_keys=["session_id"],
    value_type=ValueType.STRING,
)


# =============================================================================
# DATA SOURCES - Orígenes de datos
# =============================================================================

# Fuente: Logs de autenticación (batch/offline)
auth_logs_source = FileSource(
    name="auth_logs",
    path="data/auth_logs.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
    description="Logs de eventos de autenticación",
)

# Fuente: Métricas de tráfico de red
network_traffic_source = FileSource(
    name="network_traffic",
    path="data/network_traffic.parquet",
    timestamp_field="timestamp",
    description="Métricas agregadas de tráfico de red",
)

# Fuente: Información de usuarios (lentamente cambiante)
user_profile_source = FileSource(
    name="user_profiles",
    path="data/user_profiles.parquet",
    timestamp_field="updated_at",
    description="Perfiles de usuario",
)


# =============================================================================
# FEATURE VIEWS - Vistas de features
# =============================================================================

# Feature View: Métricas de autenticación por usuario
user_auth_features = FeatureView(
    name="user_auth_features",
    description="Features de autenticación por usuario",
    entities=[user_entity],
    ttl=timedelta(days=7),  # Time-to-live en online store
    schema=[
        Field(name="failed_logins_1h", dtype=Int64),
        Field(name="failed_logins_24h", dtype=Int64),
        Field(name="successful_logins_24h", dtype=Int64),
        Field(name="unique_ips_24h", dtype=Int64),
        Field(name="unique_devices_24h", dtype=Int64),
        Field(name="login_success_rate_7d", dtype=Float32),
        Field(name="avg_time_between_logins", dtype=Float32),
    ],
    source=auth_logs_source,
    online=True,  # Materializar en online store
    tags={
        "team": "security",
        "domain": "authentication",
        "sensitivity": "high",
    },
)

# Feature View: Métricas de red por IP
ip_network_features = FeatureView(
    name="ip_network_features",
    description="Features de tráfico de red por IP",
    entities=[ip_entity],
    ttl=timedelta(hours=6),
    schema=[
        Field(name="bytes_sent_1h", dtype=Float32),
        Field(name="bytes_received_1h", dtype=Float32),
        Field(name="packets_per_second_avg", dtype=Float32),
        Field(name="unique_destinations_1h", dtype=Int64),
        Field(name="unique_ports_1h", dtype=Int64),
        Field(name="tcp_syn_ratio", dtype=Float32),
        Field(name="dns_queries_1h", dtype=Int64),
    ],
    source=network_traffic_source,
    online=True,
    tags={
        "team": "security",
        "domain": "network",
    },
)

# Feature View: Perfil de usuario (cambia lentamente)
user_profile_features = FeatureView(
    name="user_profile_features",
    description="Features de perfil de usuario",
    entities=[user_entity],
    ttl=timedelta(days=30),  # TTL largo porque cambia poco
    schema=[
        Field(name="account_age_days", dtype=Int64),
        Field(name="is_admin", dtype=Int64),  # Boolean como int
        Field(name="department", dtype=String),
        Field(name="risk_score_historical", dtype=Float32),
        Field(name="num_security_incidents", dtype=Int64),
    ],
    source=user_profile_source,
    online=True,
    tags={
        "team": "security",
        "sensitivity": "medium",
    },
)


# =============================================================================
# ON-DEMAND FEATURES - Calculadas en tiempo real
# =============================================================================

from feast import on_demand_feature_view
from feast.types import Float64
import pandas as pd


@on_demand_feature_view(
    sources=[user_auth_features, user_profile_features],
    schema=[
        Field(name="risk_score_combined", dtype=Float32),
        Field(name="anomaly_indicator", dtype=Int64),
    ],
)
def combined_risk_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Features calculadas on-demand combinando otras features.

    Útil para:
    - Cálculos que dependen de múltiples feature views
    - Lógica de negocio compleja
    - Features que cambian muy rápido
    """
    df = pd.DataFrame()

    # Risk score combinado
    base_risk = inputs["risk_score_historical"].fillna(0.5)
    login_risk = inputs["failed_logins_24h"] / (inputs["successful_logins_24h"] + 1)
    device_risk = (inputs["unique_devices_24h"] > 3).astype(float) * 0.3

    df["risk_score_combined"] = (base_risk * 0.4 + login_risk * 0.4 + device_risk * 0.2)
    df["risk_score_combined"] = df["risk_score_combined"].clip(0, 1)

    # Indicador de anomalía
    df["anomaly_indicator"] = (
        (inputs["failed_logins_24h"] > 10) |
        (inputs["unique_ips_24h"] > 5) |
        (df["risk_score_combined"] > 0.8)
    ).astype(int)

    return df
```

### Materialización y Uso

```python
"""
Uso de Feast para training y serving.
"""
from feast import FeatureStore
from feast.infra.offline_stores.offline_utils import get_offline_store_from_config
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any


class SecurityFeatureStore:
    """
    Wrapper para Feast orientado a casos de uso de seguridad.
    """

    def __init__(self, repo_path: str = "./feature_repo"):
        """
        Inicializa conexión al feature store.

        Args:
            repo_path: Path al repositorio de Feast
        """
        self.store = FeatureStore(repo_path=repo_path)

    def materialize_features(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """
        Materializa features del offline store al online store.

        Esto debe ejecutarse periódicamente (cron job, Dagster, etc.)

        Args:
            start_date: Inicio del rango a materializar
            end_date: Fin del rango a materializar
        """
        self.store.materialize(
            start_date=start_date,
            end_date=end_date
        )
        print(f"Materialized features from {start_date} to {end_date}")

    def materialize_incremental(self, end_date: datetime = None) -> None:
        """
        Materialización incremental (solo nuevos datos).

        Args:
            end_date: Hasta cuándo materializar (default: ahora)
        """
        end_date = end_date or datetime.now()
        self.store.materialize_incremental(end_date=end_date)

    # =========================================================================
    # TRAINING - Features Históricas
    # =========================================================================

    def get_training_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str]
    ) -> pd.DataFrame:
        """
        Obtiene features históricas para entrenamiento.

        IMPORTANTE: Point-in-time join automático
        Para cada fila en entity_df, Feast retorna los valores
        de features que existían en el timestamp de esa fila.

        Args:
            entity_df: DataFrame con entity keys y timestamps
                Columnas requeridas: entity_key(s), event_timestamp
            feature_refs: Lista de features en formato "feature_view:feature"

        Returns:
            DataFrame con entity_df + features
        """
        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        ).to_df()

        return training_df

    def create_training_dataset(
        self,
        events_df: pd.DataFrame,
        label_column: str = "is_malicious"
    ) -> pd.DataFrame:
        """
        Crea dataset de entrenamiento completo.

        Args:
            events_df: DataFrame con eventos y labels
                Columnas: user_id, ip_address, event_timestamp, is_malicious
            label_column: Nombre de la columna de labels

        Returns:
            DataFrame listo para entrenar
        """
        # Definir features a obtener
        feature_refs = [
            # Auth features
            "user_auth_features:failed_logins_1h",
            "user_auth_features:failed_logins_24h",
            "user_auth_features:successful_logins_24h",
            "user_auth_features:unique_ips_24h",
            "user_auth_features:unique_devices_24h",
            "user_auth_features:login_success_rate_7d",

            # Network features
            "ip_network_features:bytes_sent_1h",
            "ip_network_features:bytes_received_1h",
            "ip_network_features:unique_destinations_1h",
            "ip_network_features:tcp_syn_ratio",

            # Profile features
            "user_profile_features:account_age_days",
            "user_profile_features:is_admin",
            "user_profile_features:risk_score_historical",

            # On-demand features
            "combined_risk_features:risk_score_combined",
            "combined_risk_features:anomaly_indicator",
        ]

        # Obtener features
        training_df = self.get_training_features(events_df, feature_refs)

        # Reorganizar columnas
        feature_columns = [f.split(":")[1] for f in feature_refs]
        entity_columns = ["user_id", "ip_address", "event_timestamp"]

        final_columns = entity_columns + feature_columns + [label_column]
        available_columns = [c for c in final_columns if c in training_df.columns]

        return training_df[available_columns]

    # =========================================================================
    # SERVING - Features en Tiempo Real
    # =========================================================================

    def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        feature_refs: List[str]
    ) -> Dict[str, List]:
        """
        Obtiene features para inferencia en tiempo real.

        Latencia típica: < 10ms para Redis

        Args:
            entity_rows: Lista de dicts con entity keys
                Ejemplo: [{"user_id": "user123", "ip_address": "1.2.3.4"}]
            feature_refs: Lista de features a obtener

        Returns:
            Dict con features por nombre
        """
        response = self.store.get_online_features(
            entity_rows=entity_rows,
            features=feature_refs,
        )

        return response.to_dict()

    def get_features_for_prediction(
        self,
        user_id: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """
        Obtiene todas las features para una predicción.

        Args:
            user_id: ID del usuario
            ip_address: IP de origen

        Returns:
            Dict con todas las features para el modelo
        """
        feature_refs = [
            "user_auth_features:failed_logins_1h",
            "user_auth_features:failed_logins_24h",
            "user_auth_features:unique_ips_24h",
            "user_auth_features:unique_devices_24h",
            "ip_network_features:bytes_sent_1h",
            "ip_network_features:unique_destinations_1h",
            "user_profile_features:account_age_days",
            "user_profile_features:is_admin",
            "combined_risk_features:risk_score_combined",
        ]

        entity_rows = [{"user_id": user_id, "ip_address": ip_address}]
        features = self.get_online_features(entity_rows, feature_refs)

        # Convertir a dict plano (primer elemento de cada lista)
        return {k: v[0] if v else None for k, v in features.items()}


# =============================================================================
# EJEMPLO COMPLETO DE USO
# =============================================================================

def example_training_workflow():
    """Workflow de entrenamiento con Feature Store."""

    fs = SecurityFeatureStore()

    # Simular eventos históricos con labels
    events_df = pd.DataFrame({
        "user_id": [f"user_{i}" for i in range(1000)],
        "ip_address": [f"192.168.1.{i % 255}" for i in range(1000)],
        "event_timestamp": pd.date_range("2024-01-01", periods=1000, freq="H"),
        "is_malicious": np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
    })

    # Obtener training dataset
    training_df = fs.create_training_dataset(events_df)

    print(f"Training dataset shape: {training_df.shape}")
    print(f"Features: {training_df.columns.tolist()}")

    return training_df


def example_serving_workflow():
    """Workflow de serving con Feature Store."""

    fs = SecurityFeatureStore()

    # Simular request de autenticación
    user_id = "user_123"
    ip_address = "192.168.1.100"

    # Obtener features en tiempo real
    features = fs.get_features_for_prediction(user_id, ip_address)

    print(f"Features for {user_id}:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    # Usar features para predicción
    # model.predict(features)

    return features
```

## 3. Offline vs Online Features

### Comparación

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OFFLINE vs ONLINE FEATURES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    OFFLINE STORE              ONLINE STORE                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Propósito       │ Training (histórico)  │ Serving (tiempo real)           │
│                                                                             │
│  Almacenamiento  │ Data Lake (Parquet,   │ Key-Value Store (Redis,         │
│                  │ BigQuery, Snowflake)  │ DynamoDB, Cassandra)            │
│                                                                             │
│  Latencia        │ Segundos a minutos    │ Milisegundos                    │
│                                                                             │
│  Volumen         │ TB-PB (todo el        │ GB (solo valores actuales)      │
│                  │ histórico)            │                                  │
│                                                                             │
│  Query Pattern   │ Point-in-time join    │ Key lookup (entity → features)  │
│                  │ (entity + timestamp)  │                                  │
│                                                                             │
│  Actualización   │ Batch (diario/horario)│ Near real-time o stream         │
│                                                                             │
│  Costo           │ Almacenamiento barato │ Compute caro (siempre activo)   │
│                  │                       │                                  │
│                                                                             │
│  FLUJO:                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   Raw Data ──► Offline Store ──► materialize() ──► Online Store    │   │
│  │       │              │                                   │         │   │
│  │       │              │                                   │         │   │
│  │       │              ▼                                   ▼         │   │
│  │       │      get_historical_         get_online_features()        │   │
│  │       │      features()                    │                       │   │
│  │       │              │                     │                       │   │
│  │       │              ▼                     ▼                       │   │
│  │       │         TRAINING              SERVING                      │   │
│  │       │                                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementación de Materialización

```python
"""
Pipeline de materialización de features.
"""
from datetime import datetime, timedelta
from typing import Optional
import logging
from feast import FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureMaterializationPipeline:
    """
    Pipeline para materializar features del offline al online store.

    En producción, esto se ejecutaría con:
    - Dagster (recomendado)
    - Airflow
    - Cron jobs
    - Cloud Functions/Lambda
    """

    def __init__(self, repo_path: str = "./feature_repo"):
        self.store = FeatureStore(repo_path=repo_path)

    def run_full_materialize(
        self,
        lookback_days: int = 7,
        end_date: Optional[datetime] = None
    ) -> None:
        """
        Materialización completa (para inicialización).

        Args:
            lookback_days: Días hacia atrás a materializar
            end_date: Fecha final (default: ahora)
        """
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        logger.info(f"Starting full materialization: {start_date} to {end_date}")

        try:
            self.store.materialize(
                start_date=start_date,
                end_date=end_date,
            )
            logger.info("Full materialization completed successfully")

        except Exception as e:
            logger.error(f"Materialization failed: {e}")
            raise

    def run_incremental_materialize(self) -> None:
        """
        Materialización incremental (para cron jobs).

        Solo materializa desde el último checkpoint.
        """
        logger.info("Starting incremental materialization")

        try:
            self.store.materialize_incremental(
                end_date=datetime.now()
            )
            logger.info("Incremental materialization completed")

        except Exception as e:
            logger.error(f"Incremental materialization failed: {e}")
            raise

    def validate_online_store(self, sample_entities: list) -> bool:
        """
        Valida que el online store tenga datos.

        Args:
            sample_entities: Entidades de prueba

        Returns:
            True si la validación pasa
        """
        feature_refs = [
            "user_auth_features:failed_logins_24h",
            "ip_network_features:bytes_sent_1h",
        ]

        try:
            response = self.store.get_online_features(
                entity_rows=sample_entities,
                features=feature_refs
            ).to_dict()

            # Verificar que hay valores no-null
            for feature, values in response.items():
                if all(v is None for v in values):
                    logger.warning(f"Feature {feature} has all null values")
                    return False

            logger.info("Online store validation passed")
            return True

        except Exception as e:
            logger.error(f"Online store validation failed: {e}")
            return False


# Ejemplo de job de Dagster para materialización
"""
# dagster_jobs.py
from dagster import job, op, schedule

@op
def materialize_features():
    pipeline = FeatureMaterializationPipeline()
    pipeline.run_incremental_materialize()

@op
def validate_features():
    pipeline = FeatureMaterializationPipeline()
    sample = [{"user_id": "test_user", "ip_address": "127.0.0.1"}]
    return pipeline.validate_online_store(sample)

@job
def feature_materialization_job():
    validate_features(materialize_features())

@schedule(cron_schedule="0 * * * *", job=feature_materialization_job)  # Cada hora
def hourly_materialization_schedule():
    return {}
"""
```

## 4. Point-in-Time Correctness

### El Problema del Data Leakage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LEAKAGE SIN POINT-IN-TIME                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INCORRECTO - JOIN Simple:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Events Table              Features Table (actual state)            │   │
│  │  ┌───────────────────┐     ┌────────────────────────────────┐      │   │
│  │  │ user │ timestamp  │     │ user │ failed_logins │ updated │      │   │
│  │  ├───────────────────┤     ├────────────────────────────────┤      │   │
│  │  │ A    │ Jan 1      │     │ A    │ 50            │ Jan 10  │      │   │
│  │  │ A    │ Jan 5      │     │ B    │ 3             │ Jan 10  │      │   │
│  │  │ B    │ Jan 8      │     │                                       │   │
│  │  └───────────────────┘     └────────────────────────────────┘      │   │
│  │                                                                     │   │
│  │  SELECT * FROM events e JOIN features f ON e.user = f.user         │   │
│  │                                                                     │   │
│  │  RESULTADO:                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ user │ timestamp │ failed_logins │                          │   │   │
│  │  ├─────────────────────────────────────────────────────────────┤   │   │
│  │  │ A    │ Jan 1     │ 50            │ ← LEAKAGE! Usa datos     │   │   │
│  │  │ A    │ Jan 5     │ 50            │   del futuro (Jan 10)    │   │   │
│  │  │ B    │ Jan 8     │ 3             │                          │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  El modelo "aprende" con información que no existía en ese momento │   │
│  │  → Métricas infladas artificialmente                               │   │
│  │  → Modelo falla en producción                                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  CORRECTO - Point-in-Time Join:                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Events Table              Features Table (versioned)               │   │
│  │  ┌───────────────────┐     ┌────────────────────────────────┐      │   │
│  │  │ user │ timestamp  │     │ user │ failed │ timestamp      │      │   │
│  │  ├───────────────────┤     ├────────────────────────────────┤      │   │
│  │  │ A    │ Jan 1      │     │ A    │ 5      │ Dec 31         │      │   │
│  │  │ A    │ Jan 5      │     │ A    │ 15     │ Jan 3          │      │   │
│  │  │ B    │ Jan 8      │     │ A    │ 50     │ Jan 10         │      │   │
│  │  └───────────────────┘     │ B    │ 3      │ Jan 7          │      │   │
│  │                            └────────────────────────────────┘      │   │
│  │                                                                     │   │
│  │  Point-in-Time Join: Para cada evento, obtener el valor de feature │   │
│  │  más reciente ANTERIOR al timestamp del evento                     │   │
│  │                                                                     │   │
│  │  RESULTADO:                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ user │ timestamp │ failed_logins │                          │   │   │
│  │  ├─────────────────────────────────────────────────────────────┤   │   │
│  │  │ A    │ Jan 1     │ 5             │ ← Valor de Dec 31        │   │   │
│  │  │ A    │ Jan 5     │ 15            │ ← Valor de Jan 3         │   │   │
│  │  │ B    │ Jan 8     │ 3             │ ← Valor de Jan 7         │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  ✓ Solo usa información disponible en ese momento                   │   │
│  │  ✓ Métricas reflejan performance real                               │   │
│  │  ✓ Modelo se comporta igual en producción                           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementación de Point-in-Time Join

```python
"""
Point-in-Time Join manual vs Feast automático.
"""
import pandas as pd
import numpy as np
from typing import List


def naive_join(events_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    JOIN INCORRECTO - causa data leakage.

    NO USAR EN PRODUCCIÓN.
    """
    # Esto toma el valor más reciente de features, sin importar timestamp
    latest_features = features_df.sort_values("timestamp").groupby("user_id").last()
    return events_df.merge(latest_features, on="user_id", how="left")


def point_in_time_join_manual(
    events_df: pd.DataFrame,
    features_df: pd.DataFrame,
    entity_column: str = "user_id",
    event_timestamp_col: str = "event_timestamp",
    feature_timestamp_col: str = "feature_timestamp"
) -> pd.DataFrame:
    """
    Point-in-Time Join manual (correcto pero lento).

    Para cada evento, obtiene las features más recientes
    que existían ANTES del evento.

    Args:
        events_df: DataFrame con eventos (entity + timestamp + label)
        features_df: DataFrame con features (entity + timestamp + features)
        entity_column: Columna de entidad para join
        event_timestamp_col: Columna de timestamp en events
        feature_timestamp_col: Columna de timestamp en features

    Returns:
        DataFrame con eventos + features point-in-time
    """
    results = []

    for _, event in events_df.iterrows():
        entity = event[entity_column]
        event_ts = event[event_timestamp_col]

        # Filtrar features del mismo entity ANTES del evento
        entity_features = features_df[
            (features_df[entity_column] == entity) &
            (features_df[feature_timestamp_col] < event_ts)
        ]

        if len(entity_features) > 0:
            # Tomar el más reciente
            latest = entity_features.sort_values(feature_timestamp_col).iloc[-1]
            merged = {**event.to_dict(), **latest.to_dict()}
        else:
            # Sin features disponibles
            merged = event.to_dict()

        results.append(merged)

    return pd.DataFrame(results)


def point_in_time_join_vectorized(
    events_df: pd.DataFrame,
    features_df: pd.DataFrame,
    entity_column: str = "user_id",
    event_timestamp_col: str = "event_timestamp",
    feature_timestamp_col: str = "feature_timestamp"
) -> pd.DataFrame:
    """
    Point-in-Time Join vectorizado (más eficiente).

    Usa merge_asof de pandas para join eficiente.
    """
    # Ordenar por timestamp
    events_sorted = events_df.sort_values(event_timestamp_col)
    features_sorted = features_df.sort_values(feature_timestamp_col)

    # merge_asof: para cada fila en left, busca la fila más cercana en right
    # donde right[timestamp] <= left[timestamp]
    result = pd.merge_asof(
        events_sorted,
        features_sorted,
        left_on=event_timestamp_col,
        right_on=feature_timestamp_col,
        by=entity_column,
        direction="backward"  # Solo valores anteriores o iguales
    )

    return result


# Ejemplo de uso
def demonstrate_point_in_time():
    """Demuestra la importancia de point-in-time correctness."""

    # Eventos históricos
    events = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B"],
        "event_timestamp": pd.to_datetime([
            "2024-01-01 10:00",
            "2024-01-05 14:00",
            "2024-01-10 08:00",
            "2024-01-03 12:00",
            "2024-01-08 16:00"
        ]),
        "is_fraud": [0, 1, 0, 0, 1]
    })

    # Features con timestamps
    features = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B"],
        "feature_timestamp": pd.to_datetime([
            "2023-12-31 00:00",  # Antes del primer evento de A
            "2024-01-04 00:00",  # Entre evento 1 y 2 de A
            "2024-01-09 00:00",  # Entre evento 2 y 3 de A
            "2024-01-02 00:00",  # Antes del primer evento de B
            "2024-01-07 00:00",  # Entre eventos de B
        ]),
        "failed_logins_7d": [5, 15, 50, 2, 8]
    })

    print("=== EVENTS ===")
    print(events)
    print("\n=== FEATURES ===")
    print(features)

    # Join incorrecto
    print("\n=== NAIVE JOIN (INCORRECTO) ===")
    naive = features.groupby("user_id")["failed_logins_7d"].last().reset_index()
    naive_result = events.merge(naive, on="user_id")
    print(naive_result)
    print("^ User A siempre tiene 50 failed_logins (valor futuro!)")

    # Join point-in-time
    print("\n=== POINT-IN-TIME JOIN (CORRECTO) ===")
    pit_result = point_in_time_join_vectorized(
        events, features,
        event_timestamp_col="event_timestamp",
        feature_timestamp_col="feature_timestamp"
    )
    print(pit_result[["user_id", "event_timestamp", "failed_logins_7d", "is_fraud"]])
    print("^ Cada evento tiene el valor de feature disponible en ese momento")


# Con Feast, point-in-time es AUTOMÁTICO:
def feast_point_in_time_example():
    """Feast hace point-in-time join automáticamente."""
    from feast import FeatureStore

    store = FeatureStore(repo_path="./feature_repo")

    # Entity DataFrame con timestamps
    entity_df = pd.DataFrame({
        "user_id": ["A", "A", "B"],
        "event_timestamp": pd.to_datetime([
            "2024-01-01",
            "2024-01-05",
            "2024-01-08"
        ])
    })

    # Feast hace point-in-time join automáticamente
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_auth_features:failed_logins_7d",
            "user_auth_features:unique_ips_24h"
        ]
    ).to_df()

    # training_df tiene features correctas para cada timestamp
    return training_df
```

## 5. Feature Store en Producción

### Integración con Model Serving

```python
"""
Integración de Feature Store con API de predicción.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import mlflow
from feast import FeatureStore
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Threat Detection API")


class PredictionRequest(BaseModel):
    """Request para predicción."""
    user_id: str
    ip_address: str
    session_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response de predicción."""
    user_id: str
    is_threat: bool
    threat_probability: float
    risk_level: str
    features_used: Dict[str, Any]
    latency_ms: float


class ThreatDetectionService:
    """
    Servicio de detección de amenazas con Feature Store.
    """

    def __init__(
        self,
        feature_store_path: str = "./feature_repo",
        model_uri: str = "models:/threat_detector/Production"
    ):
        self.feature_store = FeatureStore(repo_path=feature_store_path)
        self.model = mlflow.pyfunc.load_model(model_uri)

        # Features que necesita el modelo
        self.feature_refs = [
            "user_auth_features:failed_logins_1h",
            "user_auth_features:failed_logins_24h",
            "user_auth_features:unique_ips_24h",
            "user_auth_features:unique_devices_24h",
            "user_auth_features:login_success_rate_7d",
            "ip_network_features:bytes_sent_1h",
            "ip_network_features:unique_destinations_1h",
            "ip_network_features:tcp_syn_ratio",
            "user_profile_features:account_age_days",
            "user_profile_features:is_admin",
            "combined_risk_features:risk_score_combined",
        ]

        # Orden de features para el modelo
        self.feature_names = [f.split(":")[1] for f in self.feature_refs]

    def get_features(self, user_id: str, ip_address: str) -> Dict[str, Any]:
        """
        Obtiene features del online store.

        Args:
            user_id: ID del usuario
            ip_address: IP de origen

        Returns:
            Dict con features
        """
        entity_rows = [{"user_id": user_id, "ip_address": ip_address}]

        response = self.feature_store.get_online_features(
            entity_rows=entity_rows,
            features=self.feature_refs
        ).to_dict()

        # Extraer primer valor de cada lista
        features = {k: v[0] for k, v in response.items() if k not in ["user_id", "ip_address"]}

        return features

    def predict(self, user_id: str, ip_address: str) -> Dict[str, Any]:
        """
        Realiza predicción completa.

        Args:
            user_id: ID del usuario
            ip_address: IP de origen

        Returns:
            Dict con predicción y metadata
        """
        start_time = time.time()

        # 1. Obtener features
        features = self.get_features(user_id, ip_address)

        # 2. Preparar input para modelo
        feature_vector = []
        for name in self.feature_names:
            value = features.get(name)
            # Manejar valores faltantes
            if value is None:
                value = 0.0  # Default value
                logger.warning(f"Missing feature {name}, using default")
            feature_vector.append(float(value))

        # 3. Predecir
        X = np.array([feature_vector])
        proba = self.model.predict(X)[0]

        # Si el modelo devuelve probabilidad
        if isinstance(proba, (list, np.ndarray)) and len(proba) > 1:
            threat_prob = float(proba[1])
        else:
            threat_prob = float(proba)

        # 4. Post-procesamiento
        is_threat = threat_prob > 0.5

        if threat_prob < 0.3:
            risk_level = "LOW"
        elif threat_prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        latency_ms = (time.time() - start_time) * 1000

        return {
            "user_id": user_id,
            "is_threat": is_threat,
            "threat_probability": threat_prob,
            "risk_level": risk_level,
            "features_used": features,
            "latency_ms": latency_ms
        }


# Inicializar servicio
service = ThreatDetectionService()


@app.post("/predict", response_model=PredictionResponse)
async def predict_threat(request: PredictionRequest) -> PredictionResponse:
    """
    Endpoint de predicción de amenazas.

    Flujo:
    1. Recibe request con user_id e ip_address
    2. Obtiene features del Feature Store (online)
    3. Pasa features al modelo
    4. Retorna predicción con metadata
    """
    try:
        result = service.predict(request.user_id, request.ip_address)
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/features/{user_id}")
async def get_user_features(user_id: str, ip_address: str = "0.0.0.0"):
    """Endpoint para debug: obtener features de un usuario."""
    features = service.get_features(user_id, ip_address)
    return {"user_id": user_id, "features": features}
```

### Monitoreo de Features

```python
"""
Monitoreo de Feature Store para detectar problemas.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from feast import FeatureStore


@dataclass
class FeatureStats:
    """Estadísticas de una feature."""
    name: str
    mean: float
    std: float
    min: float
    max: float
    null_rate: float
    timestamp: datetime


class FeatureMonitor:
    """
    Monitor de Feature Store para detectar:
    - Feature drift
    - Datos faltantes
    - Valores anómalos
    - Latencia
    """

    def __init__(self, feature_store_path: str = "./feature_repo"):
        self.store = FeatureStore(repo_path=feature_store_path)
        self.baseline_stats: Dict[str, FeatureStats] = {}

    def compute_baseline(
        self,
        feature_refs: List[str],
        entity_df: pd.DataFrame
    ) -> Dict[str, FeatureStats]:
        """
        Computa estadísticas baseline de features.

        Args:
            feature_refs: Lista de features a monitorear
            entity_df: DataFrame con entities y timestamps para baseline

        Returns:
            Dict con estadísticas por feature
        """
        # Obtener features históricas
        df = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()

        stats = {}
        for feature_ref in feature_refs:
            feature_name = feature_ref.split(":")[1]

            if feature_name in df.columns:
                values = df[feature_name].dropna()

                stats[feature_name] = FeatureStats(
                    name=feature_name,
                    mean=values.mean() if len(values) > 0 else 0,
                    std=values.std() if len(values) > 0 else 0,
                    min=values.min() if len(values) > 0 else 0,
                    max=values.max() if len(values) > 0 else 0,
                    null_rate=df[feature_name].isnull().mean(),
                    timestamp=datetime.now()
                )

        self.baseline_stats = stats
        return stats

    def check_feature_drift(
        self,
        current_stats: Dict[str, FeatureStats],
        threshold_std: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Detecta drift en features comparando con baseline.

        Args:
            current_stats: Estadísticas actuales
            threshold_std: Número de desviaciones estándar para alerta

        Returns:
            Lista de alertas de drift
        """
        alerts = []

        for name, current in current_stats.items():
            if name not in self.baseline_stats:
                continue

            baseline = self.baseline_stats[name]

            # Check mean drift
            if baseline.std > 0:
                z_score = abs(current.mean - baseline.mean) / baseline.std

                if z_score > threshold_std:
                    alerts.append({
                        "type": "mean_drift",
                        "feature": name,
                        "baseline_mean": baseline.mean,
                        "current_mean": current.mean,
                        "z_score": z_score,
                        "severity": "HIGH" if z_score > 5 else "MEDIUM"
                    })

            # Check null rate increase
            if current.null_rate > baseline.null_rate + 0.1:
                alerts.append({
                    "type": "null_rate_increase",
                    "feature": name,
                    "baseline_null_rate": baseline.null_rate,
                    "current_null_rate": current.null_rate,
                    "severity": "HIGH" if current.null_rate > 0.5 else "MEDIUM"
                })

            # Check range violation
            if current.max > baseline.max * 2 or current.min < baseline.min * 0.5:
                alerts.append({
                    "type": "range_violation",
                    "feature": name,
                    "baseline_range": (baseline.min, baseline.max),
                    "current_range": (current.min, current.max),
                    "severity": "MEDIUM"
                })

        return alerts

    def monitor_online_latency(
        self,
        entity_rows: List[Dict],
        feature_refs: List[str],
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Mide latencia del online store.

        Args:
            entity_rows: Entities de prueba
            feature_refs: Features a consultar
            num_samples: Número de muestras

        Returns:
            Dict con estadísticas de latencia
        """
        import time

        latencies = []

        for _ in range(num_samples):
            start = time.time()

            self.store.get_online_features(
                entity_rows=entity_rows,
                features=feature_refs
            )

            latencies.append((time.time() - start) * 1000)  # ms

        return {
            "mean_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "max_latency_ms": np.max(latencies)
        }

    def generate_report(self) -> str:
        """Genera reporte de estado del Feature Store."""
        report = ["=" * 60]
        report.append("FEATURE STORE HEALTH REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 60)

        # Feature views registradas
        report.append("\nREGISTERED FEATURE VIEWS:")
        for fv in self.store.list_feature_views():
            report.append(f"  - {fv.name}: {len(fv.features)} features, TTL: {fv.ttl}")

        # Baseline stats
        if self.baseline_stats:
            report.append("\nBASELINE STATISTICS:")
            for name, stats in self.baseline_stats.items():
                report.append(f"  {name}:")
                report.append(f"    mean: {stats.mean:.4f}, std: {stats.std:.4f}")
                report.append(f"    null_rate: {stats.null_rate:.2%}")

        report.append("=" * 60)
        return "\n".join(report)
```

## 6. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESUMEN: Feature Stores                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  QUÉ ES UN FEATURE STORE                                                    │
│  ───────────────────────────────────────────────────────────────────────   │
│  Sistema centralizado para:                                                 │
│  • Almacenar y servir features para ML                                      │
│  • Garantizar consistencia training-serving                                 │
│  • Reutilizar features entre equipos/modelos                                │
│  • Point-in-time correctness automático                                     │
│                                                                             │
│  COMPONENTES                                                                │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Feature Registry: Catálogo y metadata                                    │
│  • Offline Store: Training (data lake, batch)                               │
│  • Online Store: Serving (key-value, real-time)                             │
│  • Materialización: Pipeline offline → online                               │
│                                                                             │
│  OFFLINE vs ONLINE                                                          │
│  ───────────────────────────────────────────────────────────────────────   │
│  Offline: Histórico, TB, segundos, training                                 │
│  Online: Actual, GB, milisegundos, serving                                  │
│                                                                             │
│  POINT-IN-TIME CORRECTNESS                                                  │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Para cada evento, usar features que existían en ese momento              │
│  • Evita data leakage automáticamente                                       │
│  • Garantiza reproducibilidad                                               │
│  • Feast lo hace automático con get_historical_features()                   │
│                                                                             │
│  FEAST                                                                      │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Open source, production-ready                                            │
│  • Entities + FeatureViews + Sources                                        │
│  • feast apply → materialize → get_features                                 │
│  • On-demand features para cálculos en tiempo real                          │
│                                                                             │
│  CONSIDERACIONES CIBERSEGURIDAD                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Features de autenticación, red, comportamiento                           │
│  • TTL cortos para features que cambian rápido                              │
│  • Monitoreo de drift (posible data poisoning)                              │
│  • Audit trail de features usadas                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Model Serving - Triton, TorchServe, BentoML, FastAPI para modelos
