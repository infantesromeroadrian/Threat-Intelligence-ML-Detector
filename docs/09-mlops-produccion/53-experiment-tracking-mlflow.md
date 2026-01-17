# Experiment Tracking con MLflow

## 1. Introduccion a Experiment Tracking

### El Problema del Tracking Manual

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EL CAOS SIN EXPERIMENT TRACKING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  "Cual era el mejor modelo?"                                                │
│                                                                             │
│  Archivos:                                                                  │
│  ├── model_v1.pkl                                                           │
│  ├── model_v2.pkl                                                           │
│  ├── model_v2_final.pkl                                                     │
│  ├── model_v2_final_REAL.pkl                                                │
│  ├── model_v3_test.pkl                                                      │
│  ├── model_BEST.pkl              ← ¿Es realmente el mejor?                  │
│  ├── model_BEST_nuevo.pkl                                                   │
│  └── model_produccion_NO_TOCAR.pkl                                          │
│                                                                             │
│  Notas en Excel:                                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ Model     │ Accuracy │ Learning Rate │ Notas                         │ │
│  ├───────────┼──────────┼───────────────┼───────────────────────────────┤ │
│  │ v1        │ 0.85     │ 0.01          │ Primer intento                │ │
│  │ v2        │ 0.89     │ ?             │ Mejor! (creo)                 │ │
│  │ v3        │ ?        │ 0.001         │ Lo probe en casa              │ │
│  │ BEST      │ 0.91     │ ?             │ Este es el bueno... o no?     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  PROBLEMAS:                                                                 │
│  • No hay trazabilidad codigo <-> modelo                                   │
│  • Hiperparametros no documentados                                          │
│  • No se sabe que datos se usaron                                           │
│  • Imposible reproducir resultados                                          │
│  • Colaboracion = pesadilla                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Que Resuelve MLflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLflow COMPONENTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. TRACKING                                                        │   │
│  │     • Registrar metricas, parametros, artefactos                    │   │
│  │     • Comparar experimentos                                         │   │
│  │     • UI para visualizacion                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  2. PROJECTS                                                        │   │
│  │     • Empaquetar codigo ML reproducible                             │   │
│  │     • Definir dependencias y entrypoints                            │   │
│  │     • Ejecutar en cualquier plataforma                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3. MODELS                                                          │   │
│  │     • Formato estandar para modelos                                 │   │
│  │     • Flavors: sklearn, pytorch, tensorflow, etc.                   │   │
│  │     • Servir modelos con un comando                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  4. MODEL REGISTRY                                                  │   │
│  │     • Lifecycle: Staging → Production → Archived                    │   │
│  │     • Versionado centralizado                                       │   │
│  │     • Control de acceso                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. MLflow Tracking

### Conceptos Basicos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        JERARQUIA MLflow TRACKING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EXPERIMENT                                                                 │
│  └── Agrupacion logica de runs (ej: "malware_detection_v2")                │
│      │                                                                      │
│      ├── RUN 1                                                              │
│      │   ├── Parameters: {"lr": 0.01, "epochs": 100}                       │
│      │   ├── Metrics: {"accuracy": 0.92, "f1": 0.89}                       │
│      │   ├── Tags: {"author": "juan", "model_type": "rf"}                  │
│      │   └── Artifacts: [model.pkl, confusion_matrix.png, ...]            │
│      │                                                                      │
│      ├── RUN 2                                                              │
│      │   ├── Parameters: {"lr": 0.001, "epochs": 200}                      │
│      │   ├── Metrics: {"accuracy": 0.95, "f1": 0.93}                       │
│      │   └── Artifacts: [...]                                              │
│      │                                                                      │
│      └── RUN 3                                                              │
│          └── ...                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Setup Inicial

```python
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import os


def setup_mlflow(
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "default"
) -> str:
    """
    Configura MLflow para tracking.

    Args:
        tracking_uri: URI del servidor MLflow
        experiment_name: Nombre del experimento

    Returns:
        experiment_id
    """
    # Configurar URI de tracking
    mlflow.set_tracking_uri(tracking_uri)

    # Crear o obtener experimento
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            tags={
                "team": "security",
                "project": "threat_detection"
            }
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    return experiment_id


# Para desarrollo local sin servidor
def setup_mlflow_local(experiment_name: str = "default") -> str:
    """Configura MLflow con almacenamiento local."""
    # Usar directorio local para artefactos
    mlflow.set_tracking_uri("file:./mlruns")
    return setup_mlflow("file:./mlruns", experiment_name)
```

### Logging de Experimentos

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Any
import json


@dataclass
class ExperimentConfig:
    """Configuracion de experimento."""
    experiment_name: str
    model_name: str
    random_state: int = 42
    test_size: float = 0.2
    tags: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class SecurityModelTrainer:
    """
    Entrenador de modelos de seguridad con MLflow tracking.

    Ejemplo de uso para detector de malware/anomalias.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = MlflowClient()

        # Setup experiment
        mlflow.set_experiment(config.experiment_name)

    def train_and_log(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hyperparameters: Dict[str, Any],
        data_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Entrena modelo y registra todo en MLflow.

        Args:
            X: Features
            y: Labels
            hyperparameters: Hiperparametros del modelo
            data_info: Metadata sobre los datos

        Returns:
            run_id del experimento
        """
        # Split datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        with mlflow.start_run(run_name=self.config.model_name) as run:
            # ========================================
            # 1. LOG TAGS
            # ========================================
            mlflow.set_tags({
                "model_type": "classification",
                "framework": "sklearn",
                "security_domain": "threat_detection",
                **self.config.tags
            })

            # ========================================
            # 2. LOG PARAMETERS
            # ========================================
            # Hiperparametros del modelo
            mlflow.log_params(hyperparameters)

            # Parametros del experimento
            mlflow.log_params({
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
                "n_features": X.shape[1],
                "class_balance_train": dict(zip(*np.unique(y_train, return_counts=True)))
            })

            # Info de datos si se proporciona
            if data_info:
                mlflow.log_params({f"data_{k}": v for k, v in data_info.items()})

            # ========================================
            # 3. ENTRENAR MODELO
            # ========================================
            model = RandomForestClassifier(
                **hyperparameters,
                random_state=self.config.random_state
            )
            model.fit(X_train, y_train)

            # ========================================
            # 4. LOG METRICS
            # ========================================
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }

            # Log metricas individuales
            mlflow.log_metrics(metrics)

            # Metricas por threshold (para curva precision-recall)
            for threshold in [0.3, 0.5, 0.7, 0.9]:
                y_pred_t = (y_proba >= threshold).astype(int)
                mlflow.log_metric(f"precision_t{threshold}", precision_score(y_test, y_pred_t, zero_division=0))
                mlflow.log_metric(f"recall_t{threshold}", recall_score(y_test, y_pred_t, zero_division=0))

            # ========================================
            # 5. LOG ARTIFACTS
            # ========================================
            # Confusion Matrix
            self._log_confusion_matrix(y_test, y_pred)

            # Feature Importance
            self._log_feature_importance(model, X.shape[1])

            # Classification Report
            report = classification_report(y_test, y_pred, output_dict=True)
            with open("classification_report.json", "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact("classification_report.json")
            os.remove("classification_report.json")

            # ========================================
            # 6. LOG MODEL
            # ========================================
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=self.config.model_name,
                signature=mlflow.models.infer_signature(X_train, y_pred),
                input_example=X_train[:5]
            )

            print(f"Run ID: {run.info.run_id}")
            print(f"Metrics: {metrics}")

            return run.info.run_id

    def _log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Genera y registra matriz de confusion."""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xlabel='Predicted label',
            ylabel='True label',
            title='Confusion Matrix'
        )

        # Anotar valores
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        os.remove("confusion_matrix.png")

    def _log_feature_importance(self, model: RandomForestClassifier, n_features: int) -> None:
        """Genera y registra grafico de importancia de features."""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:20]  # Top 20

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importance[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([f"Feature {i}" for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Feature Importance')

        fig.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        os.remove("feature_importance.png")


# Ejemplo de uso para detector de amenazas
def train_threat_detector():
    """Entrena detector de amenazas con tracking."""

    # Generar datos simulados de red
    np.random.seed(42)
    n_samples = 10000

    # Features de trafico de red
    X = np.random.randn(n_samples, 15)
    # Simular patrones maliciosos
    malicious_mask = np.random.random(n_samples) < 0.1  # 10% malicioso
    X[malicious_mask, 0] += 3  # Anomalia en feature 0
    X[malicious_mask, 5] += 2  # Anomalia en feature 5
    y = malicious_mask.astype(int)

    config = ExperimentConfig(
        experiment_name="threat_detection_v2",
        model_name="rf_threat_detector",
        tags={
            "author": "security_team",
            "dataset": "network_traffic_2024"
        }
    )

    trainer = SecurityModelTrainer(config)

    # Entrenar con diferentes hiperparametros
    hyperparams_list = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5},
    ]

    for hp in hyperparams_list:
        run_id = trainer.train_and_log(
            X, y,
            hyperparameters=hp,
            data_info={"source": "network_traffic", "version": "v2.1"}
        )
        print(f"Completed run: {run_id}\n")
```

### Logging Avanzado: Metricas en Tiempo Real

```python
import mlflow
import time
from typing import Iterator, Tuple
import numpy as np


class StreamingMetricsLogger:
    """
    Logger para metricas en tiempo real durante entrenamiento.

    Util para:
    - Monitorear convergencia
    - Detectar overfitting temprano
    - Visualizar en UI de MLflow en tiempo real
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id
        self.step = 0

    def log_training_step(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        Registra metricas de un paso de entrenamiento.

        Args:
            epoch: Numero de epoch
            train_loss: Loss de entrenamiento
            val_loss: Loss de validacion
            train_acc: Accuracy de entrenamiento
            val_acc: Accuracy de validacion
            learning_rate: Learning rate actual
        """
        metrics = {"train_loss": train_loss}

        if val_loss is not None:
            metrics["val_loss"] = val_loss
            metrics["loss_gap"] = val_loss - train_loss  # Indicador de overfitting

        if train_acc is not None:
            metrics["train_accuracy"] = train_acc

        if val_acc is not None:
            metrics["val_accuracy"] = val_acc

        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate

        mlflow.log_metrics(metrics, step=epoch)
        self.step = epoch

    def log_batch_metrics(
        self,
        batch_idx: int,
        batch_loss: float,
        batch_size: int
    ) -> None:
        """Log metricas por batch (menos frecuente)."""
        if batch_idx % 100 == 0:  # Cada 100 batches
            mlflow.log_metric("batch_loss", batch_loss, step=self.step * 1000 + batch_idx)


# Ejemplo: Entrenamiento con logging en tiempo real
def train_with_streaming_metrics():
    """Simula entrenamiento con metricas en tiempo real."""

    with mlflow.start_run(run_name="streaming_training"):
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 32)

        logger = StreamingMetricsLogger()

        for epoch in range(50):
            # Simular entrenamiento
            train_loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.05)
            val_loss = 1.0 / (epoch + 1) + 0.1 + np.random.normal(0, 0.05)
            train_acc = 1.0 - train_loss * 0.3 + np.random.normal(0, 0.02)
            val_acc = 1.0 - val_loss * 0.3 + np.random.normal(0, 0.02)
            lr = 0.01 * (0.95 ** epoch)  # Decay

            logger.log_training_step(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                learning_rate=lr
            )

            # Early stopping check
            if val_loss - train_loss > 0.3:
                mlflow.log_param("early_stopped", True)
                mlflow.log_param("stopped_epoch", epoch)
                break

            time.sleep(0.1)  # Simular tiempo de entrenamiento
```

## 3. MLflow Projects

### Estructura de Proyecto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ESTRUCTURA MLflow PROJECT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  my_ml_project/                                                             │
│  ├── MLproject                    ← Definicion del proyecto                │
│  ├── conda.yaml                   ← Dependencias (o requirements.txt)      │
│  ├── train.py                     ← Script principal                       │
│  ├── src/                                                                   │
│  │   ├── __init__.py                                                       │
│  │   ├── data_loader.py                                                    │
│  │   ├── model.py                                                          │
│  │   └── utils.py                                                          │
│  └── tests/                                                                 │
│      └── test_model.py                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Archivo MLproject

```yaml
# MLproject
name: security_threat_detector

# Opcion 1: Conda environment
conda_env: conda.yaml

# Opcion 2: Docker (recomendado para produccion)
# docker_env:
#   image: security-ml:latest

entry_points:
  # Entry point principal
  main:
    parameters:
      data_path: {type: str, default: "data/threats.csv"}
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}
      test_size: {type: float, default: 0.2}
    command: "python train.py --data-path {data_path} --n-estimators {n_estimators} --max-depth {max_depth} --test-size {test_size}"

  # Entry point para evaluacion
  evaluate:
    parameters:
      model_uri: {type: str}
      test_data_path: {type: str}
    command: "python evaluate.py --model-uri {model_uri} --test-data {test_data_path}"

  # Entry point para inference
  predict:
    parameters:
      model_uri: {type: str}
      input_path: {type: str}
      output_path: {type: str, default: "predictions.csv"}
    command: "python predict.py --model-uri {model_uri} --input {input_path} --output {output_path}"
```

### Archivo conda.yaml

```yaml
# conda.yaml
name: threat_detector
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - mlflow>=2.10.0
      - scikit-learn>=1.3.0
      - pandas>=2.0.0
      - numpy>=1.24.0
      - matplotlib>=3.7.0
      - click>=8.0.0
```

### Script de Entrenamiento para Project

```python
#!/usr/bin/env python
"""
train.py - Script de entrenamiento para MLflow Project.
"""
import click
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--data-path", type=str, required=True, help="Path to training data")
@click.option("--n-estimators", type=int, default=100, help="Number of trees")
@click.option("--max-depth", type=int, default=10, help="Max tree depth")
@click.option("--test-size", type=float, default=0.2, help="Test split ratio")
def main(data_path: str, n_estimators: int, max_depth: int, test_size: float) -> None:
    """
    Entrena modelo de deteccion de amenazas.

    Args:
        data_path: Ruta a datos CSV
        n_estimators: Numero de arboles
        max_depth: Profundidad maxima
        test_size: Proporcion de test
    """
    logger.info(f"Loading data from {data_path}")

    # Cargar datos (simulados para ejemplo)
    # En produccion: df = pd.read_csv(data_path)
    np.random.seed(42)
    n_samples = 5000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    logger.info(f"Data shape: {X.shape}, Target distribution: {np.bincount(y)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "data_path": data_path,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_size": test_size,
            "n_samples": n_samples,
            "n_features": n_features
        })

        # Train
        logger.info("Training model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }

        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )

        logger.info(f"Run completed: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
```

### Ejecutar MLflow Project

```python
import mlflow

# Ejecutar proyecto local
mlflow.projects.run(
    uri="./my_ml_project",
    entry_point="main",
    parameters={
        "data_path": "data/threats.csv",
        "n_estimators": 200,
        "max_depth": 15
    }
)

# Ejecutar desde Git
mlflow.projects.run(
    uri="https://github.com/myorg/threat-detector.git",
    version="v1.0.0",  # Tag o branch
    entry_point="main",
    parameters={"n_estimators": 100}
)

# Ejecutar con backend especifico (Kubernetes, Databricks, etc.)
mlflow.projects.run(
    uri="./my_ml_project",
    backend="kubernetes",
    backend_config={
        "kube-context": "my-cluster",
        "repository-uri": "my-registry/ml-project",
        "kube-job-template-path": "kubernetes/job-template.yaml"
    }
)
```

## 4. MLflow Models

### Formato de Modelo MLflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ESTRUCTURA MLflow MODEL                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  model/                                                                     │
│  ├── MLmodel                     ← Metadata del modelo                     │
│  ├── conda.yaml                  ← Dependencias                            │
│  ├── requirements.txt            ← Dependencias pip                        │
│  ├── python_env.yaml             ← Entorno Python                          │
│  ├── model.pkl                   ← Artefacto del modelo (sklearn)          │
│  └── input_example.json          ← Ejemplo de input                        │
│                                                                             │
│  MLmodel (contenido):                                                       │
│  ───────────────────────────────────────────────────────────────────────   │
│  artifact_path: model                                                       │
│  flavors:                                                                   │
│    python_function:                                                         │
│      env:                                                                   │
│        conda: conda.yaml                                                   │
│      loader_module: mlflow.sklearn                                         │
│      model_path: model.pkl                                                 │
│      python_version: 3.11.0                                                │
│    sklearn:                                                                 │
│      code: null                                                            │
│      pickled_model: model.pkl                                              │
│      sklearn_version: 1.3.0                                                │
│  signature:                                                                 │
│    inputs: '[{"type": "double", "name": "feature_0"}, ...]'               │
│    outputs: '[{"type": "long", "name": "prediction"}]'                    │
│  model_uuid: abc123...                                                      │
│  run_id: def456...                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Guardar Modelos con Diferentes Flavors

```python
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List


# ==============================================================================
# SKLEARN MODEL
# ==============================================================================

def save_sklearn_model_example():
    """Guardar modelo sklearn con MLflow."""

    X_train = np.random.randn(100, 5)
    y_train = (X_train[:, 0] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    with mlflow.start_run():
        # Guardar con signature (recomendado)
        signature = mlflow.models.infer_signature(
            X_train,
            model.predict(X_train)
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:3],  # Ejemplo de input
            registered_model_name="sklearn_threat_detector"
        )


# ==============================================================================
# PYTORCH MODEL
# ==============================================================================

class ThreatDetectorNN(nn.Module):
    """Red neuronal para deteccion de amenazas."""

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def save_pytorch_model_example():
    """Guardar modelo PyTorch con MLflow."""

    model = ThreatDetectorNN(input_size=10)

    # Ejemplo de input
    example_input = torch.randn(5, 10)

    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=mlflow.models.infer_signature(
                example_input.numpy(),
                model(example_input).detach().numpy()
            ),
            input_example=example_input.numpy(),
            registered_model_name="pytorch_threat_detector"
        )


# ==============================================================================
# CUSTOM PYFUNC MODEL (Para modelos complejos)
# ==============================================================================

class CustomSecurityModel(mlflow.pyfunc.PythonModel):
    """
    Modelo personalizado con preprocessing integrado.

    Util cuando:
    - Necesitas preprocessing custom
    - Tienes ensemble de modelos
    - Requieres post-processing
    """

    def __init__(self, model, preprocessor, threshold: float = 0.5):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Cargar artefactos adicionales."""
        # Cargar modelos auxiliares, configs, etc.
        pass

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prediccion con preprocessing.

        Args:
            context: Contexto MLflow
            model_input: DataFrame con features

        Returns:
            DataFrame con predicciones
        """
        # Preprocess
        processed = self.preprocessor.transform(model_input)

        # Predict
        probas = self.model.predict_proba(processed)[:, 1]

        # Post-process
        predictions = (probas >= self.threshold).astype(int)

        return pd.DataFrame({
            "prediction": predictions,
            "probability": probas,
            "risk_level": pd.cut(
                probas,
                bins=[0, 0.3, 0.7, 1.0],
                labels=["low", "medium", "high"]
            )
        })


def save_custom_model_example():
    """Guardar modelo custom con MLflow."""
    from sklearn.preprocessing import StandardScaler

    # Entrenar modelo y preprocessor
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)

    preprocessor = StandardScaler().fit(X)
    model = RandomForestClassifier().fit(preprocessor.transform(X), y)

    # Crear modelo custom
    custom_model = CustomSecurityModel(model, preprocessor, threshold=0.5)

    # Ejemplo de input
    example_input = pd.DataFrame(X[:3], columns=[f"feat_{i}" for i in range(5)])

    with mlflow.start_run():
        # Guardar artefactos adicionales
        mlflow.log_artifact("config.yaml")  # Si existe

        # Guardar modelo custom
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=custom_model,
            signature=mlflow.models.infer_signature(
                example_input,
                custom_model.predict(None, example_input)
            ),
            input_example=example_input,
            pip_requirements=["scikit-learn>=1.3.0", "pandas>=2.0.0"]
        )
```

### Cargar y Usar Modelos

```python
import mlflow
import pandas as pd
import numpy as np


def load_model_examples():
    """Diferentes formas de cargar modelos."""

    # 1. Cargar desde run_id
    model_uri = "runs:/abc123def456/model"
    model = mlflow.pyfunc.load_model(model_uri)

    # 2. Cargar desde Model Registry
    model_uri = "models:/threat_detector/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    # 3. Cargar version especifica del registry
    model_uri = "models:/threat_detector/3"
    model = mlflow.pyfunc.load_model(model_uri)

    # 4. Cargar flavor especifico (sklearn)
    model = mlflow.sklearn.load_model("models:/threat_detector/Production")

    # Predecir
    data = pd.DataFrame(np.random.randn(10, 5), columns=[f"feat_{i}" for i in range(5)])
    predictions = model.predict(data)

    return predictions


def serve_model_locally():
    """Servir modelo localmente con MLflow."""
    import subprocess

    # Servir modelo como REST API
    # mlflow models serve -m models:/threat_detector/Production -p 5001

    # O programaticamente
    subprocess.run([
        "mlflow", "models", "serve",
        "-m", "models:/threat_detector/Production",
        "-p", "5001",
        "--no-conda"  # Usar entorno actual
    ])
```

## 5. MLflow Model Registry

### Workflow del Registry

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MODEL REGISTRY WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Training   │───►│  Register   │───►│   Staging   │───►│ Production  │  │
│  │    Run      │    │   Model     │    │   Testing   │    │  Deployed   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                           │                   │                   │         │
│                           ▼                   ▼                   ▼         │
│                    ┌─────────────────────────────────────────────────────┐  │
│                    │              MODEL REGISTRY                        │  │
│                    │                                                     │  │
│                    │  threat_detector                                    │  │
│                    │  ├── Version 1  [Archived]                         │  │
│                    │  ├── Version 2  [Archived]                         │  │
│                    │  ├── Version 3  [Staging]    ← Testing            │  │
│                    │  └── Version 4  [Production] ← Live               │  │
│                    │                                                     │  │
│                    │  malware_classifier                                 │  │
│                    │  ├── Version 1  [Production]                       │  │
│                    │  └── Version 2  [Staging]                          │  │
│                    │                                                     │  │
│                    └─────────────────────────────────────────────────────┘  │
│                                                                             │
│  STAGES:                                                                    │
│  • None: Recien registrado, sin stage                                      │
│  • Staging: En pruebas, validacion                                         │
│  • Production: Desplegado en produccion                                    │
│  • Archived: Retirado, historico                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Gestion del Registry

```python
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from typing import List, Optional, Dict, Any
from datetime import datetime


class ModelRegistryManager:
    """
    Gestor del Model Registry.

    Maneja lifecycle de modelos:
    - Registro
    - Promocion entre stages
    - Versionado
    - Metadata
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        run_id: str,
        artifact_path: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> ModelVersion:
        """
        Registra modelo desde un run.

        Args:
            run_id: ID del run de MLflow
            artifact_path: Path al modelo en artifacts
            model_name: Nombre para registrar
            description: Descripcion del modelo
            tags: Tags adicionales

        Returns:
            ModelVersion creada
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"

        # Registrar modelo
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        # Añadir descripcion
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )

        # Añadir tags
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )

        return model_version

    def promote_to_staging(
        self,
        model_name: str,
        version: int,
        description: Optional[str] = None
    ) -> None:
        """Promueve modelo a Staging."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Staging",
            archive_existing_versions=False  # No archivar staging anterior
        )

        if description:
            self.client.update_model_version(
                name=model_name,
                version=str(version),
                description=f"[STAGING] {description}"
            )

        print(f"Model {model_name} v{version} promoted to Staging")

    def promote_to_production(
        self,
        model_name: str,
        version: int,
        archive_previous: bool = True
    ) -> None:
        """
        Promueve modelo a Production.

        Args:
            model_name: Nombre del modelo
            version: Version a promover
            archive_previous: Si archivar version anterior en prod
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Production",
            archive_existing_versions=archive_previous
        )

        # Añadir timestamp de promocion
        self.client.set_model_version_tag(
            name=model_name,
            version=str(version),
            key="promoted_to_production_at",
            value=datetime.now().isoformat()
        )

        print(f"Model {model_name} v{version} promoted to Production")

    def rollback_production(
        self,
        model_name: str,
        target_version: Optional[int] = None
    ) -> None:
        """
        Rollback a version anterior.

        Args:
            model_name: Nombre del modelo
            target_version: Version destino (None = ultima archived)
        """
        # Obtener version actual en prod
        current_prod = self.get_production_version(model_name)

        if target_version is None:
            # Buscar ultima version archived
            versions = self.client.search_model_versions(f"name='{model_name}'")
            archived = [v for v in versions if v.current_stage == "Archived"]
            if not archived:
                raise ValueError("No archived versions available for rollback")
            target_version = max(int(v.version) for v in archived)

        # Archivar version actual
        if current_prod:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_prod.version,
                stage="Archived"
            )

        # Promover target a prod
        self.promote_to_production(model_name, target_version, archive_previous=False)

        print(f"Rolled back {model_name} to v{target_version}")

    def get_production_version(self, model_name: str) -> Optional[ModelVersion]:
        """Obtiene version en produccion."""
        versions = self.client.get_latest_versions(model_name, stages=["Production"])
        return versions[0] if versions else None

    def get_staging_version(self, model_name: str) -> Optional[ModelVersion]:
        """Obtiene version en staging."""
        versions = self.client.get_latest_versions(model_name, stages=["Staging"])
        return versions[0] if versions else None

    def compare_versions(
        self,
        model_name: str,
        version_a: int,
        version_b: int
    ) -> Dict[str, Any]:
        """Compara metricas de dos versiones."""
        client = MlflowClient()

        v_a = client.get_model_version(model_name, str(version_a))
        v_b = client.get_model_version(model_name, str(version_b))

        # Obtener runs asociados
        run_a = client.get_run(v_a.run_id)
        run_b = client.get_run(v_b.run_id)

        comparison = {
            "version_a": {
                "version": version_a,
                "metrics": run_a.data.metrics,
                "params": run_a.data.params
            },
            "version_b": {
                "version": version_b,
                "metrics": run_b.data.metrics,
                "params": run_b.data.params
            },
            "metric_diff": {}
        }

        # Calcular diferencias de metricas
        for metric in run_a.data.metrics:
            if metric in run_b.data.metrics:
                diff = run_b.data.metrics[metric] - run_a.data.metrics[metric]
                comparison["metric_diff"][metric] = {
                    "diff": diff,
                    "improved": diff > 0  # Asumiendo higher is better
                }

        return comparison

    def list_models(self) -> List[str]:
        """Lista todos los modelos registrados."""
        return [m.name for m in self.client.search_registered_models()]

    def get_model_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Obtiene historial de versiones de un modelo."""
        versions = self.client.search_model_versions(f"name='{model_name}'")

        history = []
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            history.append({
                "version": v.version,
                "stage": v.current_stage,
                "created_at": v.creation_timestamp,
                "run_id": v.run_id,
                "description": v.description
            })

        return history


# Uso del Registry Manager
def example_registry_workflow():
    """Ejemplo de workflow con Model Registry."""

    manager = ModelRegistryManager()

    # 1. Despues de entrenar, registrar modelo
    # (Asumiendo que tenemos un run_id de un entrenamiento exitoso)
    run_id = "abc123"  # Ejemplo

    model_version = manager.register_model(
        run_id=run_id,
        artifact_path="model",
        model_name="threat_detector",
        description="Random Forest para deteccion de amenazas",
        tags={
            "author": "security_team",
            "framework": "sklearn",
            "dataset_version": "v2.1"
        }
    )

    print(f"Registered model version: {model_version.version}")

    # 2. Promover a Staging para testing
    manager.promote_to_staging(
        model_name="threat_detector",
        version=int(model_version.version),
        description="Testing new feature engineering"
    )

    # 3. Despues de validar en staging, promover a prod
    manager.promote_to_production(
        model_name="threat_detector",
        version=int(model_version.version)
    )

    # 4. Si hay problemas, rollback
    # manager.rollback_production("threat_detector")

    # 5. Comparar versiones
    comparison = manager.compare_versions("threat_detector", 1, 2)
    print(f"Metric differences: {comparison['metric_diff']}")
```

## 6. Comparacion: MLflow vs Weights & Biases vs Neptune

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   COMPARACION DE PLATAFORMAS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CARACTERISTICA        │  MLflow      │  W&B          │  Neptune           │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Pricing               │  Open Source │  Freemium     │  Freemium          │
│                        │  (gratis)    │  (Teams: $$)  │  (Teams: $$)       │
│                                                                             │
│  Self-hosted           │  ✓ Facil     │  Enterprise   │  Enterprise        │
│                        │              │  only         │  only              │
│                                                                             │
│  UI/UX                 │  Basico      │  Excelente    │  Muy bueno         │
│                        │              │               │                     │
│  Colaboracion          │  Limitada    │  Excelente    │  Buena             │
│                        │              │  (reports)    │                     │
│                                                                             │
│  Model Registry        │  ✓ Incluido  │  Artifacts    │  Model Registry    │
│                                                                             │
│  Hyperparameter Tuning │  Manual      │  Sweeps       │  Via integracion   │
│                        │              │  (integrado)  │                     │
│                                                                             │
│  Real-time viz         │  Basico      │  Excelente    │  Bueno             │
│                                                                             │
│  Data versioning       │  No          │  Artifacts    │  No                │
│                        │  (usar DVC)  │  (limitado)   │  (usar DVC)        │
│                                                                             │
│  Alertas               │  No          │  ✓            │  ✓                 │
│                                                                             │
│  Compliance/Security   │  Self-hosted │  SOC2         │  SOC2              │
│                        │  (tu control)│               │                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

RECOMENDACION:

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  USAR MLflow SI:                                                            │
│  • Necesitas self-hosting (datos sensibles, compliance)                    │
│  • Presupuesto limitado                                                     │
│  • Quieres control total                                                    │
│  • Ya usas Databricks                                                       │
│                                                                             │
│  USAR W&B SI:                                                               │
│  • Priorizas UX y colaboracion                                              │
│  • Equipo mediano-grande                                                    │
│  • Necesitas hyperparameter tuning integrado                                │
│  • Presupuesto disponible                                                   │
│                                                                             │
│  USAR Neptune SI:                                                           │
│  • Alternativa a W&B con pricing diferente                                  │
│  • Necesitas buen UI pero no el precio de W&B                               │
│                                                                             │
│  PARA CIBERSEGURIDAD:                                                       │
│  → MLflow self-hosted (control de datos sensibles)                         │
│  → O W&B con plan Enterprise si compliance lo permite                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Codigo para W&B (Comparacion)

```python
"""
Ejemplo equivalente en Weights & Biases para comparacion.
"""
import wandb
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def train_with_wandb():
    """Entrenamiento con W&B tracking."""

    # Inicializar run
    run = wandb.init(
        project="threat-detection",
        config={
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.01
        },
        tags=["security", "random-forest"]
    )

    config = wandb.config

    # Datos simulados
    X = np.random.randn(1000, 10)
    y = (X[:, 0] > 0).astype(int)

    # Entrenar
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth
    )
    model.fit(X, y)

    # Log metricas
    accuracy = model.score(X, y)
    wandb.log({"accuracy": accuracy})

    # Log confusion matrix (W&B tiene visualizaciones built-in)
    y_pred = model.predict(X)
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y,
            preds=y_pred,
            class_names=["benign", "malicious"]
        )
    })

    # Guardar modelo como artifact
    artifact = wandb.Artifact("threat_detector", type="model")
    # artifact.add_file("model.pkl")
    run.log_artifact(artifact)

    run.finish()


def wandb_sweep_example():
    """Hyperparameter sweep con W&B."""

    sweep_config = {
        "method": "bayes",  # bayesian optimization
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "n_estimators": {"min": 50, "max": 300},
            "max_depth": {"min": 5, "max": 30},
            "min_samples_split": {"min": 2, "max": 20}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="threat-detection")
    wandb.agent(sweep_id, train_with_wandb, count=20)
```

## 7. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RESUMEN: Experiment Tracking con MLflow                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COMPONENTES MLflow                                                         │
│  ───────────────────────────────────────────────────────────────────────   │
│  1. Tracking: Registrar params, metrics, artifacts                          │
│  2. Projects: Empaquetar codigo reproducible                                │
│  3. Models: Formato estandar multi-framework                                │
│  4. Registry: Lifecycle y versionado centralizado                           │
│                                                                             │
│  TRACKING BEST PRACTICES                                                    │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Log TODOS los parametros (incluyendo random_state)                       │
│  • Log metricas en train Y validation                                       │
│  • Usar signatures para validar inputs                                      │
│  • Incluir input_example para documentacion                                 │
│  • Tags para organizar (author, dataset_version, etc.)                      │
│                                                                             │
│  MODEL REGISTRY WORKFLOW                                                    │
│  ───────────────────────────────────────────────────────────────────────   │
│  Register → Staging (testing) → Production → (Archived)                    │
│                      │                                                      │
│                      └── Rollback si hay problemas                          │
│                                                                             │
│  ALTERNATIVAS                                                               │
│  ───────────────────────────────────────────────────────────────────────   │
│  • MLflow: Self-hosted, gratis, control total                               │
│  • W&B: Mejor UX, sweeps, colaboracion (pago)                               │
│  • Neptune: Similar a W&B, pricing diferente                                │
│                                                                             │
│  PARA CIBERSEGURIDAD                                                        │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Preferir MLflow self-hosted (datos sensibles)                            │
│  • Audit trail para compliance                                              │
│  • Rollback rapido en caso de degradacion                                   │
│  • Versionado estricto de modelos en produccion                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Feature Stores - Feast, offline vs online features, point-in-time correctness
