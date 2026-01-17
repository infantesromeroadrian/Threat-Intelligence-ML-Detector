# Model Serving

## 1. Introduccion al Model Serving

### El Desafio de Servir Modelos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEL NOTEBOOK A PRODUCCION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NOTEBOOK (Desarrollo)              PRODUCCION (Serving)                    │
│  ┌──────────────────────┐           ┌──────────────────────────────────┐   │
│  │ model.predict(X)     │    vs     │ • Miles de requests/segundo     │   │
│  │                      │           │ • Latencia < 100ms               │   │
│  │ Sin preocupaciones:  │           │ • Alta disponibilidad (99.9%)   │   │
│  │ • Latencia           │           │ • Versionado y rollback          │   │
│  │ • Concurrencia       │           │ • Monitoring y alertas           │   │
│  │ • Escalabilidad      │           │ • Costo optimizado               │   │
│  │ • Errores de red     │           │ • Multi-modelo                   │   │
│  └──────────────────────┘           └──────────────────────────────────┘   │
│                                                                             │
│  REQUERIMIENTOS POR CASO DE USO:                                            │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  ┌─────────────┬────────────┬────────────┬───────────────────────────────┐ │
│  │ Caso de Uso │ Latencia   │ Throughput │ Ejemplo Ciberseguridad        │ │
│  ├─────────────┼────────────┼────────────┼───────────────────────────────┤ │
│  │ Real-time   │ < 50ms     │ 10K+ RPS   │ Deteccion de intrusion        │ │
│  │ Near-RT     │ < 500ms    │ 1K RPS     │ Clasificacion de malware      │ │
│  │ Batch       │ Minutos    │ Alto vol.  │ Scoring de riesgo diario      │ │
│  │ Edge        │ < 10ms     │ Local      │ Sensor IoT, firewall          │ │
│  └─────────────┴────────────┴────────────┴───────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Arquitectura General de Serving

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ARQUITECTURA DE MODEL SERVING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           CLIENTS                                   │   │
│  │    [Web App]    [Mobile]    [IoT]    [Internal Services]           │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         LOAD BALANCER                               │   │
│  │              [nginx / HAProxy / Cloud LB]                           │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        API GATEWAY                                  │   │
│  │    • Rate limiting    • Authentication    • Request validation     │   │
│  │    • Routing          • Logging          • Metrics                 │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│            ┌────────────────────┼────────────────────┐                     │
│            │                    │                    │                     │
│            ▼                    ▼                    ▼                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │  Model Server   │  │  Model Server   │  │  Model Server   │            │
│  │   Instance 1    │  │   Instance 2    │  │   Instance N    │            │
│  │                 │  │                 │  │                 │            │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │            │
│  │ │  Model v2   │ │  │ │  Model v2   │ │  │ │  Model v2   │ │            │
│  │ │ (Production)│ │  │ │ (Production)│ │  │ │ (Production)│ │            │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │            │
│  │ ┌─────────────┐ │  │                 │  │                 │            │
│  │ │  Model v3   │ │  │                 │  │                 │            │
│  │ │  (Canary)   │ │  │                 │  │                 │            │
│  │ └─────────────┘ │  │                 │  │                 │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│            │                    │                    │                     │
│            └────────────────────┼────────────────────┘                     │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         MONITORING                                  │   │
│  │    [Prometheus]    [Grafana]    [Alerting]    [Logging]            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. FastAPI para Model Serving

### Implementacion Basica

```python
"""
Model Serving con FastAPI - Detector de Amenazas.

Estructura recomendada:
app/
├── __init__.py
├── main.py           # FastAPI app
├── models/           # Modelos ML
│   ├── __init__.py
│   └── threat_detector.py
├── schemas/          # Pydantic schemas
│   ├── __init__.py
│   └── prediction.py
├── services/         # Logica de negocio
│   └── prediction_service.py
└── utils/
    └── metrics.py
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import numpy as np
import time
import logging
import asyncio
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMAS
# =============================================================================

class PredictionRequest(BaseModel):
    """Request de prediccion."""
    features: List[float] = Field(..., min_items=1, description="Vector de features")
    request_id: Optional[str] = Field(None, description="ID para tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5, -0.3, 1.2, 0.0, 0.8, -0.5, 0.1, 0.9, -0.2, 0.4],
                "request_id": "req-12345"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request de prediccion batch."""
    instances: List[List[float]] = Field(..., description="Lista de vectores")


class PredictionResponse(BaseModel):
    """Response de prediccion."""
    prediction: int
    probability: float
    risk_level: str
    latency_ms: float
    model_version: str
    request_id: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response de prediccion batch."""
    predictions: List[PredictionResponse]
    total_latency_ms: float
    batch_size: int


class ModelInfo(BaseModel):
    """Informacion del modelo."""
    name: str
    version: str
    framework: str
    input_shape: List[int]
    loaded_at: str
    requests_served: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float


# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager:
    """
    Gestor de modelos con carga lazy y versionado.
    """

    def __init__(self):
        self.model = None
        self.model_version = "unknown"
        self.model_name = "threat_detector"
        self.loaded_at = None
        self.requests_served = 0
        self.start_time = time.time()

    async def load_model(self, model_path: str = None) -> None:
        """
        Carga modelo de forma asincrona.

        En produccion, cargar desde MLflow:
        self.model = mlflow.sklearn.load_model("models:/threat_detector/Production")
        """
        logger.info("Loading model...")

        # Simular carga de modelo (en produccion: cargar de MLflow/S3)
        await asyncio.sleep(0.5)  # Simular IO

        # Para ejemplo, crear modelo dummy
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Entrenar con datos dummy (en produccion: cargar modelo entrenado)
        X_dummy = np.random.randn(100, 10)
        y_dummy = (X_dummy[:, 0] > 0).astype(int)
        self.model.fit(X_dummy, y_dummy)

        self.model_version = "1.0.0"
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"Model loaded: {self.model_name} v{self.model_version}")

    def predict(self, features: np.ndarray) -> tuple:
        """
        Realiza prediccion.

        Args:
            features: Array de features (n_samples, n_features)

        Returns:
            (predictions, probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self.requests_served += 1

        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]

        return predictions, probabilities

    def get_info(self) -> ModelInfo:
        """Retorna informacion del modelo."""
        return ModelInfo(
            name=self.model_name,
            version=self.model_version,
            framework="sklearn",
            input_shape=[10],
            loaded_at=self.loaded_at or "not loaded",
            requests_served=self.requests_served
        )

    @property
    def uptime(self) -> float:
        """Tiempo de actividad en segundos."""
        return time.time() - self.start_time


# =============================================================================
# FASTAPI APP
# =============================================================================

# Singleton del model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: carga modelo al iniciar."""
    await model_manager.load_model()
    yield
    # Cleanup al cerrar
    logger.info("Shutting down...")


app = FastAPI(
    title="Threat Detection API",
    description="API para deteccion de amenazas en tiempo real",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_model_manager() -> ModelManager:
    """Dependency para obtener model manager."""
    return model_manager


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """
    Health check endpoint.

    Usado por load balancers y Kubernetes para verificar salud del servicio.
    """
    return HealthResponse(
        status="healthy" if manager.model is not None else "unhealthy",
        model_loaded=manager.model is not None,
        uptime_seconds=manager.uptime
    )


@app.get("/ready")
async def readiness_check(manager: ModelManager = Depends(get_model_manager)):
    """
    Readiness check - usado por Kubernetes.

    Retorna 200 solo si el modelo esta listo para servir.
    """
    if manager.model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(manager: ModelManager = Depends(get_model_manager)):
    """Informacion del modelo cargado."""
    return manager.get_info()


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Endpoint de prediccion individual.

    Recibe vector de features y retorna prediccion con probabilidad.
    """
    start_time = time.time()

    try:
        # Validar tamano de input
        if len(request.features) != 10:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 10 features, got {len(request.features)}"
            )

        # Convertir a numpy
        features = np.array([request.features])

        # Predecir
        predictions, probabilities = manager.predict(features)

        prediction = int(predictions[0])
        probability = float(probabilities[0])

        # Determinar nivel de riesgo
        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        latency_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            latency_ms=latency_ms,
            model_version=manager.model_version,
            request_id=request.request_id
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Endpoint de prediccion batch.

    Mas eficiente para multiples predicciones.
    """
    start_time = time.time()

    try:
        # Validar
        for i, instance in enumerate(request.instances):
            if len(instance) != 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Instance {i}: expected 10 features, got {len(instance)}"
                )

        # Convertir a numpy (batch)
        features = np.array(request.instances)

        # Predecir batch
        predictions, probabilities = manager.predict(features)

        # Construir responses
        responses = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if prob < 0.3:
                risk_level = "LOW"
            elif prob < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            responses.append(PredictionResponse(
                prediction=int(pred),
                probability=float(prob),
                risk_level=risk_level,
                latency_ms=0,  # Individual latency not meaningful in batch
                model_version=manager.model_version
            ))

        total_latency = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=responses,
            total_latency_ms=total_latency,
            batch_size=len(request.instances)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# METRICAS PROMETHEUS
# =============================================================================

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Metricas
REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)


@app.get("/metrics")
async def metrics():
    """Endpoint de metricas Prometheus."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Middleware para metricas
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware para registrar metricas de cada request."""
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    # Registrar metricas para endpoints de prediccion
    if request.url.path in ["/predict", "/predict/batch"]:
        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)

    return response


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Dockerfile para FastAPI

```dockerfile
# Dockerfile
# Multi-stage build para imagen optimizada

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Instalar uv
RUN pip install uv

# Copiar dependencias
COPY pyproject.toml uv.lock ./

# Instalar dependencias
RUN uv pip install --system --no-cache

# Stage 2: Runtime
FROM python:3.11-slim

# Crear usuario no-root
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copiar dependencias instaladas
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar codigo
COPY app/ ./app/

# Cambiar a usuario no-root
USER appuser

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## 3. NVIDIA Triton Inference Server

### Arquitectura de Triton

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA TRITON INFERENCE SERVER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Caracteristicas:                                                           │
│  • Multi-framework: PyTorch, TensorFlow, ONNX, TensorRT, sklearn           │
│  • Multi-model: Servir multiples modelos simultaneamente                   │
│  • Batching dinamico: Agrupa requests para mejor throughput GPU            │
│  • Model ensembles: Encadenar modelos                                       │
│  • GPU optimizado: TensorRT, cuDNN                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         TRITON SERVER                               │   │
│  │                                                                     │   │
│  │   Clients ─────► HTTP/gRPC ─────► Request Handler                  │   │
│  │                                        │                            │   │
│  │                                        ▼                            │   │
│  │                              ┌─────────────────┐                    │   │
│  │                              │ Dynamic Batcher │                    │   │
│  │                              └────────┬────────┘                    │   │
│  │                                       │                             │   │
│  │         ┌─────────────────────────────┼─────────────────────────┐  │   │
│  │         │                             │                         │  │   │
│  │         ▼                             ▼                         ▼  │   │
│  │   ┌──────────┐               ┌──────────────┐          ┌────────┐ │   │
│  │   │ PyTorch  │               │ TensorRT     │          │ ONNX   │ │   │
│  │   │ Backend  │               │ Backend      │          │Backend │ │   │
│  │   └──────────┘               └──────────────┘          └────────┘ │   │
│  │         │                             │                         │  │   │
│  │         └─────────────────────────────┼─────────────────────────┘  │   │
│  │                                       │                             │   │
│  │                                       ▼                             │   │
│  │                              ┌─────────────────┐                    │   │
│  │                              │   Model Repo    │                    │   │
│  │                              │ models/         │                    │   │
│  │                              │ ├── model_a/    │                    │   │
│  │                              │ │   └── 1/      │                    │   │
│  │                              │ └── model_b/    │                    │   │
│  │                              │     └── 1/      │                    │   │
│  │                              └─────────────────┘                    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuracion de Modelo para Triton

```python
"""
Preparar modelo para Triton Inference Server.
"""
import os
import json
import numpy as np
from typing import Dict, Any


def create_triton_model_repo(
    model_name: str,
    model_path: str,
    repo_path: str = "./model_repository"
) -> str:
    """
    Crea estructura de repositorio de modelos para Triton.

    Estructura requerida:
    model_repository/
    └── <model_name>/
        ├── config.pbtxt
        └── 1/
            └── model.onnx (o model.pt, model.savedmodel, etc.)
    """
    model_dir = os.path.join(repo_path, model_name)
    version_dir = os.path.join(model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)

    return model_dir


def create_config_pbtxt(
    model_name: str,
    backend: str,
    input_name: str,
    input_dims: list,
    output_name: str,
    output_dims: list,
    max_batch_size: int = 8,
    instance_count: int = 1,
    kind: str = "KIND_GPU"
) -> str:
    """
    Genera config.pbtxt para modelo Triton.

    Args:
        model_name: Nombre del modelo
        backend: Backend (onnxruntime, pytorch, tensorflow, etc.)
        input_name: Nombre del input tensor
        input_dims: Dimensiones del input (sin batch)
        output_name: Nombre del output tensor
        output_dims: Dimensiones del output (sin batch)
        max_batch_size: Tamano maximo de batch
        instance_count: Numero de instancias del modelo
        kind: KIND_GPU o KIND_CPU

    Returns:
        Contenido del config.pbtxt
    """
    config = f"""
name: "{model_name}"
backend: "{backend}"
max_batch_size: {max_batch_size}

input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: {input_dims}
  }}
]

output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: {output_dims}
  }}
]

instance_group [
  {{
    count: {instance_count}
    kind: {kind}
  }}
]

dynamic_batching {{
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100
}}
"""
    return config


def export_sklearn_to_onnx(
    sklearn_model,
    model_name: str,
    n_features: int,
    repo_path: str = "./model_repository"
) -> str:
    """
    Exporta modelo sklearn a ONNX para Triton.

    Args:
        sklearn_model: Modelo sklearn entrenado
        model_name: Nombre para el modelo
        n_features: Numero de features de entrada
        repo_path: Path al repositorio de modelos

    Returns:
        Path al modelo exportado
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    # Crear directorio
    model_dir = create_triton_model_repo(model_name, "", repo_path)
    version_dir = os.path.join(model_dir, "1")

    # Convertir a ONNX
    initial_type = [('input', FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)

    # Guardar modelo
    model_path = os.path.join(version_dir, "model.onnx")
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Crear config.pbtxt
    config = create_config_pbtxt(
        model_name=model_name,
        backend="onnxruntime",
        input_name="input",
        input_dims=[n_features],
        output_name="output_probability",
        output_dims=[2],  # Probabilidades para 2 clases
        max_batch_size=32,
        kind="KIND_CPU"
    )

    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config)

    print(f"Model exported to {model_path}")
    print(f"Config created at {config_path}")

    return model_path


def export_pytorch_to_triton(
    pytorch_model,
    model_name: str,
    input_shape: tuple,
    repo_path: str = "./model_repository"
) -> str:
    """
    Exporta modelo PyTorch a formato Triton.

    Args:
        pytorch_model: Modelo PyTorch
        model_name: Nombre del modelo
        input_shape: Shape del input (sin batch)
        repo_path: Path al repositorio

    Returns:
        Path al modelo exportado
    """
    import torch

    model_dir = create_triton_model_repo(model_name, "", repo_path)
    version_dir = os.path.join(model_dir, "1")

    # Exportar a TorchScript
    pytorch_model.eval()
    example_input = torch.randn(1, *input_shape)

    traced_model = torch.jit.trace(pytorch_model, example_input)
    model_path = os.path.join(version_dir, "model.pt")
    traced_model.save(model_path)

    # O exportar a ONNX (mejor para optimizacion)
    onnx_path = os.path.join(version_dir, "model.onnx")
    torch.onnx.export(
        pytorch_model,
        example_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # Config para PyTorch
    config = create_config_pbtxt(
        model_name=model_name,
        backend="pytorch",  # o "onnxruntime" si usas ONNX
        input_name="input",
        input_dims=list(input_shape),
        output_name="output",
        output_dims=[1],
        max_batch_size=32,
        kind="KIND_GPU"
    )

    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config)

    return model_path
```

### Cliente de Triton

```python
"""
Cliente para Triton Inference Server.
"""
import numpy as np
from typing import Dict, List, Any, Optional
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient


class TritonClient:
    """
    Cliente para interactuar con Triton Inference Server.
    """

    def __init__(
        self,
        url: str = "localhost:8000",
        protocol: str = "http"  # "http" o "grpc"
    ):
        """
        Inicializa cliente Triton.

        Args:
            url: URL del servidor Triton
            protocol: Protocolo a usar (http o grpc)
        """
        self.url = url
        self.protocol = protocol

        if protocol == "http":
            self.client = httpclient.InferenceServerClient(url=url)
        else:
            self.client = grpcclient.InferenceServerClient(url=url)

    def is_server_live(self) -> bool:
        """Verifica si el servidor esta activo."""
        return self.client.is_server_live()

    def is_model_ready(self, model_name: str) -> bool:
        """Verifica si un modelo esta listo."""
        return self.client.is_model_ready(model_name)

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Obtiene metadata del modelo."""
        metadata = self.client.get_model_metadata(model_name)
        return {
            "name": metadata.name,
            "versions": metadata.versions,
            "inputs": [{"name": i.name, "shape": i.shape, "datatype": i.datatype}
                      for i in metadata.inputs],
            "outputs": [{"name": o.name, "shape": o.shape, "datatype": o.datatype}
                       for o in metadata.outputs]
        }

    def predict(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        output_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Realiza prediccion en Triton.

        Args:
            model_name: Nombre del modelo
            inputs: Dict de nombre_input -> array
            output_names: Lista de nombres de outputs

        Returns:
            Dict de nombre_output -> array
        """
        # Crear inputs
        triton_inputs = []
        for name, data in inputs.items():
            if self.protocol == "http":
                inp = httpclient.InferInput(
                    name,
                    data.shape,
                    "FP32"
                )
                inp.set_data_from_numpy(data.astype(np.float32))
            else:
                inp = grpcclient.InferInput(
                    name,
                    data.shape,
                    "FP32"
                )
                inp.set_data_from_numpy(data.astype(np.float32))
            triton_inputs.append(inp)

        # Crear outputs
        if self.protocol == "http":
            triton_outputs = [
                httpclient.InferRequestedOutput(name) for name in output_names
            ]
        else:
            triton_outputs = [
                grpcclient.InferRequestedOutput(name) for name in output_names
            ]

        # Inferencia
        response = self.client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs
        )

        # Extraer resultados
        results = {}
        for name in output_names:
            results[name] = response.as_numpy(name)

        return results

    def predict_batch(
        self,
        model_name: str,
        batch_inputs: List[Dict[str, np.ndarray]],
        output_names: List[str]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Prediccion batch usando async requests.
        """
        results = []
        for inputs in batch_inputs:
            result = self.predict(model_name, inputs, output_names)
            results.append(result)
        return results


# Ejemplo de uso
def triton_inference_example():
    """Ejemplo de inferencia con Triton."""

    # Conectar a Triton
    client = TritonClient(url="localhost:8000", protocol="http")

    # Verificar servidor
    if not client.is_server_live():
        print("Triton server not available")
        return

    # Verificar modelo
    model_name = "threat_detector"
    if not client.is_model_ready(model_name):
        print(f"Model {model_name} not ready")
        return

    # Obtener metadata
    metadata = client.get_model_metadata(model_name)
    print(f"Model metadata: {metadata}")

    # Hacer prediccion
    input_data = np.random.randn(1, 10).astype(np.float32)

    results = client.predict(
        model_name=model_name,
        inputs={"input": input_data},
        output_names=["output_probability"]
    )

    print(f"Predictions: {results}")
```

## 4. BentoML

### Creacion de BentoService

```python
"""
BentoML - Framework para empaquetar y servir modelos.

Ventajas:
- API declarativa simple
- Empaquetado automatico (dependencias, modelo, codigo)
- Adaptive batching
- Multiples runners
- Deployment a Kubernetes, AWS Lambda, etc.
"""
import bentoml
from bentoml.io import JSON, NumpyNdarray
import numpy as np
from pydantic import BaseModel
from typing import List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# GUARDAR MODELO EN BENTOML
# =============================================================================

def save_model_to_bentoml():
    """Guarda modelo sklearn en BentoML model store."""
    from sklearn.ensemble import RandomForestClassifier

    # Entrenar modelo (o cargar de MLflow)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X = np.random.randn(1000, 10)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)

    # Guardar en BentoML
    saved_model = bentoml.sklearn.save_model(
        "threat_detector",
        model,
        signatures={
            "predict": {
                "batchable": True,
                "batch_dim": 0,
            },
            "predict_proba": {
                "batchable": True,
                "batch_dim": 0,
            }
        },
        labels={
            "owner": "security-team",
            "stage": "production",
        },
        metadata={
            "accuracy": 0.95,
            "dataset_version": "v2.1",
        },
    )

    print(f"Model saved: {saved_model.tag}")
    return saved_model


# =============================================================================
# SERVICIO BENTOML
# =============================================================================

# Cargar modelo
threat_detector_runner = bentoml.sklearn.get("threat_detector:latest").to_runner()

# Crear servicio
svc = bentoml.Service("threat_detection_service", runners=[threat_detector_runner])


# Schemas
class ThreatInput(BaseModel):
    features: List[float]


class ThreatOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str


# API endpoints
@svc.api(input=JSON(pydantic_model=ThreatInput), output=JSON(pydantic_model=ThreatOutput))
async def predict(input_data: ThreatInput) -> ThreatOutput:
    """
    Endpoint de prediccion individual.
    """
    features = np.array([input_data.features])

    # BentoML hace batching automatico si hay multiples requests
    predictions = await threat_detector_runner.predict.async_run(features)
    probabilities = await threat_detector_runner.predict_proba.async_run(features)

    prob = float(probabilities[0, 1])

    if prob < 0.3:
        risk = "LOW"
    elif prob < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return ThreatOutput(
        prediction=int(predictions[0]),
        probability=prob,
        risk_level=risk
    )


@svc.api(input=NumpyNdarray(dtype="float32", shape=(-1, 10)), output=NumpyNdarray())
async def predict_batch(input_array: np.ndarray) -> np.ndarray:
    """
    Endpoint de prediccion batch.

    Recibe array numpy directamente.
    """
    predictions = await threat_detector_runner.predict.async_run(input_array)
    return predictions


# =============================================================================
# CUSTOM RUNNER (para modelos complejos)
# =============================================================================

import bentoml
from bentoml import Runnable


class ThreatDetectorRunnable(Runnable):
    """
    Runner personalizado para logica compleja.

    Util cuando necesitas:
    - Preprocessing custom
    - Ensemble de modelos
    - Feature engineering en serving
    """

    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = bentoml.sklearn.load_model("threat_detector:latest")
        # Cargar otros componentes
        # self.scaler = load_scaler()
        # self.feature_extractor = load_feature_extractor()

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def predict_with_preprocessing(self, input_array: np.ndarray) -> np.ndarray:
        """
        Prediccion con preprocessing integrado.
        """
        # Preprocessing
        # processed = self.scaler.transform(input_array)

        # Prediccion
        predictions = self.model.predict(input_array)

        return predictions

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def predict_with_confidence(self, input_array: np.ndarray) -> dict:
        """
        Prediccion con intervalos de confianza.
        """
        # Para RF, podemos estimar varianza usando individual trees
        predictions = []
        for tree in self.model.estimators_:
            predictions.append(tree.predict_proba(input_array)[:, 1])

        predictions = np.array(predictions)

        return {
            "mean": predictions.mean(axis=0),
            "std": predictions.std(axis=0),
            "prediction": (predictions.mean(axis=0) > 0.5).astype(int)
        }
```

### Bentofile y Deployment

```yaml
# bentofile.yaml
service: "service:svc"
description: "Threat Detection Service"

labels:
  owner: security-team
  stage: production

include:
  - "*.py"
  - "requirements.txt"

python:
  packages:
    - scikit-learn>=1.3.0
    - numpy>=1.24.0
    - pydantic>=2.0.0

docker:
  distro: debian
  python_version: "3.11"
  system_packages:
    - curl
  env:
    - name: BENTOML_CONFIG
      value: ./configuration.yaml
```

```python
"""
Build y deployment de BentoML.
"""
import subprocess


def build_bento():
    """Construye el Bento (paquete)."""
    subprocess.run(["bentoml", "build"], check=True)


def containerize_bento(tag: str = "threat_detector:latest"):
    """Construye imagen Docker."""
    subprocess.run([
        "bentoml", "containerize",
        "threat_detection_service:latest",
        "-t", tag
    ], check=True)


def serve_locally(port: int = 3000):
    """Sirve localmente para desarrollo."""
    subprocess.run([
        "bentoml", "serve",
        "service:svc",
        "--reload",
        "--port", str(port)
    ])


def deploy_to_kubernetes():
    """
    Genera manifests de Kubernetes.

    Requiere: pip install bentoml[kubernetes]
    """
    # Generar manifests
    subprocess.run([
        "bentoml", "deployment", "generate",
        "threat_detection_service:latest",
        "--kubernetes",
        "-o", "k8s-manifests/"
    ], check=True)

    # Aplicar
    subprocess.run([
        "kubectl", "apply", "-f", "k8s-manifests/"
    ], check=True)
```

## 5. Batching y Optimizacion

### Dynamic Batching

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DYNAMIC BATCHING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SIN BATCHING:                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Req 1 ──► [Model] ──► Resp 1    (10ms GPU idle entre requests)            │
│  Req 2 ──► [Model] ──► Resp 2                                              │
│  Req 3 ──► [Model] ──► Resp 3                                              │
│  Req 4 ──► [Model] ──► Resp 4                                              │
│                                                                             │
│  Total: 4 * 10ms = 40ms, GPU utilization: ~10%                             │
│                                                                             │
│  CON DYNAMIC BATCHING:                                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Req 1 ─┐                                                                   │
│  Req 2 ─┼──► [Batch 4] ──► [Model] ──► Responses                           │
│  Req 3 ─┤      ▲                                                            │
│  Req 4 ─┘      │                                                            │
│                │                                                            │
│         Wait 5ms or batch_size=4                                           │
│                                                                             │
│  Total: 15ms (5ms wait + 10ms inference), GPU utilization: ~80%            │
│                                                                             │
│  TRADEOFF:                                                                  │
│  • Mayor wait time = Mayor throughput, Mayor latencia individual           │
│  • Menor wait time = Menor throughput, Menor latencia individual           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion de Batching

```python
"""
Implementacion de dynamic batching para serving.
"""
import asyncio
import numpy as np
from typing import List, Any, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Request individual en un batch."""
    request_id: str
    data: np.ndarray
    future: asyncio.Future
    received_at: float


class DynamicBatcher:
    """
    Implementacion de dynamic batching.

    Agrupa requests individuales en batches para mejor utilizacion de GPU.
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        max_batch_size: int = 32,
        max_latency_ms: float = 50.0,
        min_batch_size: int = 1
    ):
        """
        Inicializa el batcher.

        Args:
            model_fn: Funcion de prediccion del modelo
            max_batch_size: Tamano maximo de batch
            max_latency_ms: Latencia maxima de espera
            min_batch_size: Tamano minimo para procesar batch
        """
        self.model_fn = model_fn
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        self.min_batch_size = min_batch_size

        self.queue: List[BatchRequest] = []
        self.lock = asyncio.Lock()
        self.batch_event = asyncio.Event()

        # Metricas
        self.batches_processed = 0
        self.requests_processed = 0
        self.total_batch_size = 0

    async def start(self):
        """Inicia el procesador de batches."""
        asyncio.create_task(self._batch_processor())

    async def predict(self, data: np.ndarray, request_id: str = "") -> np.ndarray:
        """
        Agrega request a la cola y espera resultado.

        Args:
            data: Datos de entrada (1 sample)
            request_id: ID para tracking

        Returns:
            Prediccion
        """
        future = asyncio.get_event_loop().create_future()

        request = BatchRequest(
            request_id=request_id,
            data=data,
            future=future,
            received_at=time.time()
        )

        async with self.lock:
            self.queue.append(request)
            self.batch_event.set()

        # Esperar resultado
        return await future

    async def _batch_processor(self):
        """Loop de procesamiento de batches."""
        while True:
            await self.batch_event.wait()

            # Esperar para acumular mas requests
            await asyncio.sleep(self.max_latency_ms / 1000)

            async with self.lock:
                if len(self.queue) == 0:
                    self.batch_event.clear()
                    continue

                # Tomar hasta max_batch_size requests
                batch = self.queue[:self.max_batch_size]
                self.queue = self.queue[self.max_batch_size:]

                if len(self.queue) == 0:
                    self.batch_event.clear()

            # Procesar batch
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[BatchRequest]):
        """Procesa un batch de requests."""
        if len(batch) == 0:
            return

        try:
            # Combinar inputs
            batch_data = np.vstack([req.data for req in batch])

            # Prediccion batch
            start_time = time.time()
            results = self.model_fn(batch_data)
            inference_time = (time.time() - start_time) * 1000

            logger.debug(
                f"Batch processed: size={len(batch)}, "
                f"inference_time={inference_time:.2f}ms"
            )

            # Distribuir resultados
            for i, request in enumerate(batch):
                if not request.future.done():
                    request.future.set_result(results[i:i+1])

            # Actualizar metricas
            self.batches_processed += 1
            self.requests_processed += len(batch)
            self.total_batch_size += len(batch)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Propagar error a todos los requests
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

    @property
    def average_batch_size(self) -> float:
        """Tamano promedio de batch."""
        if self.batches_processed == 0:
            return 0
        return self.total_batch_size / self.batches_processed

    def get_stats(self) -> dict:
        """Retorna estadisticas del batcher."""
        return {
            "batches_processed": self.batches_processed,
            "requests_processed": self.requests_processed,
            "average_batch_size": self.average_batch_size,
            "queue_size": len(self.queue)
        }


# Ejemplo de uso con FastAPI
from fastapi import FastAPI

app = FastAPI()
batcher: DynamicBatcher = None


@app.on_event("startup")
async def startup():
    global batcher

    # Cargar modelo
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)
    X = np.random.randn(100, 10)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)

    # Crear batcher
    batcher = DynamicBatcher(
        model_fn=model.predict_proba,
        max_batch_size=32,
        max_latency_ms=50
    )
    await batcher.start()


@app.post("/predict")
async def predict(features: List[float]):
    data = np.array([features])
    result = await batcher.predict(data)
    return {"probability": float(result[0, 1])}
```

## 6. Comparacion de Frameworks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   COMPARACION DE FRAMEWORKS DE SERVING                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│              │ FastAPI   │ Triton    │ BentoML   │ TorchServe │ vLLM      │
│  ────────────┼───────────┼───────────┼───────────┼────────────┼───────────│
│  Complejidad │ Baja      │ Alta      │ Media     │ Media      │ Media     │
│  ────────────┼───────────┼───────────┼───────────┼────────────┼───────────│
│  Multi-model │ Manual    │ Native    │ Si        │ Si         │ Si        │
│  ────────────┼───────────┼───────────┼───────────┼────────────┼───────────│
│  Batching    │ Manual    │ Dinamico  │ Adaptivo  │ Dinamico   │ Continuous│
│  ────────────┼───────────┼───────────┼───────────┼────────────┼───────────│
│  GPU Support │ Manual    │ Excelente │ Bueno     │ Bueno      │ Excelente │
│  ────────────┼───────────┼───────────┼───────────┼────────────┼───────────│
│  Frameworks  │ Todos     │ Todos     │ Todos     │ PyTorch    │ LLMs      │
│  ────────────┼───────────┼───────────┼───────────┼────────────┼───────────│
│  Cloud Ready │ Docker    │ K8s       │ K8s/Cloud │ K8s        │ K8s       │
│  ────────────┼───────────┼───────────┼───────────┼────────────┼───────────│
│  Monitoreo   │ Manual    │ Integrado │ Integrado │ Integrado  │ Integrado │
│  ────────────┼───────────┼───────────┼───────────┼────────────┼───────────│
│                                                                             │
│  RECOMENDACIONES:                                                           │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  • FastAPI: POCs, modelos simples, control total                           │
│  • Triton: Produccion GPU, multi-modelo, alto throughput                   │
│  • BentoML: Balance flexibilidad-facilidad, multi-cloud                    │
│  • TorchServe: Ecosistema PyTorch, AWS integrado                           │
│  • vLLM: LLMs exclusivamente, mejor performance para LLMs                  │
│                                                                             │
│  PARA CIBERSEGURIDAD:                                                       │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Real-time (IDS/IPS): Triton con GPU, latencia < 10ms                    │
│  • Clasificacion malware: BentoML o FastAPI                                │
│  • Analisis de logs: Batch processing, cualquier framework                 │
│  • LLM para SOC: vLLM para chat/analisis                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 7. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESUMEN: Model Serving                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CONCEPTOS CLAVE                                                            │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Latencia: Tiempo de respuesta (p50, p95, p99)                           │
│  • Throughput: Requests por segundo                                         │
│  • Batching: Agrupar requests para mejor GPU utilization                   │
│  • Scaling: Horizontal (replicas) o vertical (recursos)                     │
│                                                                             │
│  FRAMEWORKS                                                                 │
│  ───────────────────────────────────────────────────────────────────────   │
│  • FastAPI: Simple, flexible, para POCs y produccion ligera                │
│  • Triton: NVIDIA, multi-framework, GPU optimizado, produccion pesada      │
│  • BentoML: Balance entre facilidad y features, deployment multi-cloud     │
│  • vLLM: Especializado en LLMs, continuous batching                        │
│                                                                             │
│  BEST PRACTICES                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Health checks (/health, /ready)                                         │
│  • Metricas Prometheus                                                      │
│  • Timeout y retry con backoff                                              │
│  • Graceful shutdown                                                        │
│  • Versionado de modelos                                                    │
│  • Batching para GPU                                                        │
│                                                                             │
│  DOCKER                                                                     │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Multi-stage builds                                                       │
│  • Non-root user                                                            │
│  • Health check en Dockerfile                                               │
│  • Imagenes slim/distroless                                                │
│                                                                             │
│  CIBERSEGURIDAD                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│  • IDS/IPS: Latencia < 10ms, Triton con GPU                                │
│  • Malware classification: BentoML/FastAPI                                 │
│  • Log analysis: Batch processing                                          │
│  • SOC Assistant: vLLM para LLMs                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** CI/CD para ML - GitHub Actions, testing de modelos, validacion automatica
