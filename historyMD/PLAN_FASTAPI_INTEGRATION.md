# 🚀 PLAN: FastAPI Integration

**Objetivo:** Añadir API REST manteniendo la arquitectura hexagonal limpia

---

## 📍 UBICACIÓN EN LA ARQUITECTURA

### ✅ Respuesta Corta:
**FastAPI va en `infrastructure/api/`** - Al mismo nivel que `infrastructure/cli/`

### 🏗️ Arquitectura Actualizada:

```
src/ml_engineer_course/
├── domain/              [NÚCLEO - No cambia]
│   ├── entities/
│   ├── ports/
│   └── services/
│
├── application/         [USE CASES - No cambia]
│   ├── use_cases/
│   │   ├── classify_email.py      ← Reutilizamos
│   │   └── list_models.py         ← Reutilizamos
│   └── container.py               ← Reutilizamos
│
├── infrastructure/      [ADAPTADORES]
│   ├── adapters/        ← Ya existe
│   │
│   ├── cli/             ← Ya existe (FASE 4)
│   │   ├── main.py
│   │   └── commands.py
│   │
│   └── api/             ← AQUÍ VA FASTAPI (NUEVA)
│       ├── __init__.py
│       ├── main.py                # FastAPI app
│       ├── routers/               # Endpoints
│       │   ├── __init__.py
│       │   ├── classify.py        # POST /classify
│       │   └── models.py          # GET /models
│       ├── schemas/               # Pydantic request/response
│       │   ├── __init__.py
│       │   ├── requests.py
│       │   └── responses.py
│       ├── dependencies.py        # DI para FastAPI
│       └── middleware.py          # CORS, logging, etc.
│
└── config/              ← Actualizar settings
    └── settings.py      # Añadir API_HOST, API_PORT, etc.
```

---

## 🎯 ¿POR QUÉ EN `infrastructure/api/`?

### Razones de Arquitectura Hexagonal:

1. **FastAPI es un DETALLE de infraestructura**
   - No es lógica de negocio → NO va en `domain/`
   - No es orquestación → NO va en `application/`
   - Es un **adaptador** de entrada (driving adapter) → SÍ va en `infrastructure/`

2. **Simetría con CLI**
   ```
   infrastructure/
   ├── cli/    ← Driving adapter (entrada por terminal)
   └── api/    ← Driving adapter (entrada por HTTP)
   ```

3. **Ambos usan los MISMOS use cases**
   ```python
   # CLI usa:
   use_case = container.get_classify_use_case()
   result = use_case.execute(email_text)
   
   # API usa LO MISMO:
   use_case = container.get_classify_use_case()
   result = use_case.execute(email_text)
   ```

4. **Separación de concerns**
   - `domain/` → Business logic (Email, ClassificationResult, etc.)
   - `application/` → Use cases (ClassifyEmailUseCase)
   - `infrastructure/cli/` → Terminal interface
   - `infrastructure/api/` → HTTP interface
   - `infrastructure/adapters/` → External services (ML models, formatters)

---

## 📋 ESTRUCTURA DETALLADA DE `infrastructure/api/`

### 1. `main.py` - FastAPI Application

```python
"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ...application import Container
from ...config import settings
from .routers import classify, models

# Create app
app = FastAPI(
    title="Email Classifier API",
    description="SPAM & PHISHING detection API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DI container (singleton for app lifetime)
container = Container(settings)

# Include routers
app.include_router(classify.router, prefix="/api/v1", tags=["classify"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])

@app.get("/")
def root():
    return {"message": "Email Classifier API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

---

### 2. `routers/classify.py` - Classification Endpoints

```python
"""Classification endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from ....application import Container
from ..schemas.requests import ClassifyEmailRequest
from ..schemas.responses import ClassificationResponse
from ..dependencies import get_container

router = APIRouter()


@router.post("/classify", response_model=ClassificationResponse)
def classify_email(
    request: ClassifyEmailRequest,
    container: Container = Depends(get_container)
) -> ClassificationResponse:
    """
    Classify an email as SPAM/PHISHING.
    
    - **email_text**: Email body text (required)
    - **subject**: Email subject (optional)
    - **sender**: Email sender (optional)
    """
    try:
        # Get use case (REUTILIZA application layer)
        use_case = container.get_classify_use_case()
        
        # Execute classification (usa execute_raw para obtener entity)
        result = use_case.execute_raw(
            email_text=request.email_text,
            subject=request.subject,
            sender=request.sender
        )
        
        # Convert domain entity → API response
        return ClassificationResponse.from_domain(result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
```

---

### 3. `routers/models.py` - Models Management Endpoints

```python
"""Models management endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from ....application import Container
from ....domain.entities import ModelType
from ..schemas.responses import ModelInfoResponse, ModelsListResponse
from ..dependencies import get_container

router = APIRouter()


@router.get("/models/{model_name}", response_model=ModelsListResponse)
def list_models(
    model_name: ModelType,
    container: Container = Depends(get_container)
) -> ModelsListResponse:
    """List all available versions of a model."""
    try:
        use_case = container.get_list_models_use_case()
        models = use_case.execute(model_name)
        
        return ModelsListResponse(
            model_name=model_name,
            total_versions=len(models),
            models=[ModelInfoResponse.from_domain(m) for m in models]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models/{model_name}/latest", response_model=ModelInfoResponse)
def get_latest_model(
    model_name: ModelType,
    container: Container = Depends(get_container)
) -> ModelInfoResponse:
    """Get info about latest model version."""
    use_case = container.get_list_models_use_case()
    latest = use_case.get_latest(model_name)
    
    if not latest:
        raise HTTPException(
            status_code=404,
            detail=f"No models found for '{model_name}'"
        )
    
    return ModelInfoResponse.from_domain(latest)
```

---

### 4. `schemas/requests.py` - Request Schemas

```python
"""API request schemas (Pydantic)."""

from pydantic import BaseModel, Field


class ClassifyEmailRequest(BaseModel):
    """Request schema for email classification."""
    
    email_text: str = Field(
        ...,
        min_length=1,
        description="Email body text to classify",
        example="WINNER! You have won $1000! Click here NOW!"
    )
    subject: str | None = Field(
        None,
        description="Email subject line",
        example="Congratulations!"
    )
    sender: str | None = Field(
        None,
        description="Email sender address",
        example="scam@fake.com"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "email_text": "URGENT! Your account has been suspended.",
                "subject": "Account Alert",
                "sender": "phishing@fake.com"
            }
        }
```

---

### 5. `schemas/responses.py` - Response Schemas

```python
"""API response schemas (Pydantic)."""

from pydantic import BaseModel, Field

from ....domain.entities import ClassificationResult, ModelMetadata


class ClassificationResponse(BaseModel):
    """Response schema for email classification."""
    
    verdict: str = Field(..., description="Final verdict")
    risk_level: str = Field(..., description="Risk level")
    is_malicious: bool = Field(..., description="Is email malicious")
    
    spam_label: str
    spam_probability: float = Field(..., ge=0.0, le=1.0)
    spam_model_version: str
    
    phishing_label: str
    phishing_probability: float = Field(..., ge=0.0, le=1.0)
    phishing_model_version: str
    
    execution_time_ms: float
    
    @classmethod
    def from_domain(cls, result: ClassificationResult) -> "ClassificationResponse":
        """Convert domain entity to API response."""
        return cls(
            verdict=result.final_verdict,
            risk_level=result.risk_level,
            is_malicious=result.is_malicious,
            spam_label=result.spam_prediction.label,
            spam_probability=result.spam_prediction.probability,
            spam_model_version=result.spam_prediction.model_timestamp,
            phishing_label=result.phishing_prediction.label,
            phishing_probability=result.phishing_prediction.probability,
            phishing_model_version=result.phishing_prediction.model_timestamp,
            execution_time_ms=result.execution_time_ms
        )


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    
    name: str
    timestamp: str
    accuracy: float
    accuracy_percent: float
    train_samples: int
    vocabulary_size: int
    file_size_mb: float
    
    @classmethod
    def from_domain(cls, metadata: ModelMetadata) -> "ModelInfoResponse":
        """Convert domain entity to API response."""
        return cls(
            name=metadata.name,
            timestamp=metadata.timestamp,
            accuracy=metadata.accuracy,
            accuracy_percent=metadata.accuracy_percent,
            train_samples=metadata.train_samples,
            vocabulary_size=metadata.vocabulary_size,
            file_size_mb=metadata.file_size_mb
        )


class ModelsListResponse(BaseModel):
    """Response schema for models list."""
    
    model_name: str
    total_versions: int
    models: list[ModelInfoResponse]
```

---

### 6. `dependencies.py` - FastAPI Dependencies

```python
"""FastAPI dependency injection."""

from fastapi import Request

from ...application import Container


def get_container(request: Request) -> Container:
    """
    Get DI container from app state.
    
    FastAPI dependency that provides the container to endpoints.
    """
    return request.app.state.container
```

---

### 7. Actualizar `config/settings.py`

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=False, description="Hot reload")
    api_cors_origins: list[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    api_workers: int = Field(default=1, description="Uvicorn workers")
```

---

## 🔄 FLUJO DE PETICIÓN HTTP

```
1. HTTP Request
   POST /api/v1/classify
   {
     "email_text": "WINNER! Click here!",
     "subject": "Urgent",
     "sender": "scam@fake.com"
   }
   ↓
2. FastAPI Router (infrastructure/api/routers/classify.py)
   ↓
3. Pydantic Validation (infrastructure/api/schemas/requests.py)
   ↓
4. Get Container via Dependency (infrastructure/api/dependencies.py)
   ↓
5. Get Use Case from Container
   use_case = container.get_classify_use_case()
   ↓
6. Execute Use Case (application/use_cases/classify_email.py)
   result = use_case.execute_raw(...)
   ↓
7. Domain Service (domain/services/email_classifier.py)
   ↓
8. Predictors (infrastructure/adapters/sklearn_predictor.py)
   ↓
9. Return Domain Entity (ClassificationResult)
   ↓
10. Convert to API Response (infrastructure/api/schemas/responses.py)
    ClassificationResponse.from_domain(result)
    ↓
11. HTTP Response
    {
      "verdict": "SPAM+PHISHING",
      "risk_level": "CRITICAL",
      "is_malicious": true,
      ...
    }
```

---

## 📝 EJEMPLO DE USO

### Iniciar API

```bash
# Opción 1: Uvicorn directo
uvicorn ml_engineer_course.infrastructure.api.main:app --reload

# Opción 2: Script de inicio
python -m ml_engineer_course.infrastructure.api.main

# Opción 3: Via pyproject.toml script
[project.scripts]
email-classifier-api = "ml_engineer_course.infrastructure.api:run_api"
```

### Llamar a la API

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "email_text": "WINNER! You won $1000!",
    "subject": "Urgent Prize",
    "sender": "scam@fake.com"
  }'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/classify",
    json={
        "email_text": "WINNER! You won $1000!",
        "subject": "Urgent Prize",
        "sender": "scam@fake.com"
    }
)

result = response.json()
print(result["verdict"])  # "SPAM+PHISHING"
print(result["risk_level"])  # "CRITICAL"
```

**JavaScript:**
```javascript
fetch('http://localhost:8000/api/v1/classify', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email_text: 'WINNER! You won $1000!',
    subject: 'Urgent Prize',
    sender: 'scam@fake.com'
  })
})
.then(res => res.json())
.then(data => console.log(data.verdict));
```

---

## 🎯 VENTAJAS DE ESTA ARQUITECTURA

### 1. **Reutilización Total**
```python
# CLI usa:
use_case.execute(email_text, detail_level="simple")

# API usa:
use_case.execute_raw(email_text)

# Mismo use case, diferente interfaz
```

### 2. **Testeo Independiente**
- Tests de domain: No cambian
- Tests de use cases: No cambian
- Tests de API: Nuevos, pero aislados

### 3. **Cambio de Framework Fácil**
```
infrastructure/
├── api-fastapi/    ← Actual
├── api-flask/      ← Alternativa
└── api-grpc/       ← Otra alternativa
```

Solo cambias el adaptador, domain/application intactos.

### 4. **Deploy Flexible**
```bash
# Solo CLI
pip install email-classifier

# Solo API
docker run email-classifier-api

# Ambos en mismo container
docker run email-classifier-full
```

---

## 📊 COMPARACIÓN: CLI vs API

| Aspecto | CLI | API |
|---------|-----|-----|
| **Ubicación** | `infrastructure/cli/` | `infrastructure/api/` |
| **Framework** | Typer | FastAPI |
| **Input** | Args, file, stdin | HTTP JSON |
| **Output** | Terminal (Rich) | HTTP JSON |
| **Use Cases** | ✅ Reutiliza | ✅ Reutiliza |
| **Domain** | ✅ Reutiliza | ✅ Reutiliza |
| **Container** | ✅ Reutiliza | ✅ Reutiliza |

---

## ✅ CHECKLIST IMPLEMENTACIÓN

- [ ] Crear estructura en `infrastructure/api/`
- [ ] Implementar `main.py` (FastAPI app)
- [ ] Crear routers (`classify.py`, `models.py`)
- [ ] Definir schemas (requests, responses)
- [ ] Configurar dependencies (DI)
- [ ] Añadir settings API en `config/`
- [ ] Middleware (CORS, logging)
- [ ] Tests de API (pytest + TestClient)
- [ ] Documentación OpenAPI (automática)
- [ ] Script de inicio (uvicorn)
- [ ] Dockerfile para deploy
- [ ] docker-compose.yml

---

## 🚀 SIGUIENTE NIVEL: Microservicios

Si escalaras más:

```
infrastructure/
├── api/                # API Gateway
├── services/
│   ├── spam-service/   # Microservicio spam (puerto 8001)
│   └── phishing-service/  # Microservicio phishing (puerto 8002)
└── cli/
```

Pero para tu caso actual, **una API monolítica en `infrastructure/api/` es perfecta**.

---

**Resumen:** FastAPI va en `infrastructure/api/` porque es un **driving adapter** (interfaz de entrada) que reutiliza los use cases existentes, igual que hace el CLI. ✅
