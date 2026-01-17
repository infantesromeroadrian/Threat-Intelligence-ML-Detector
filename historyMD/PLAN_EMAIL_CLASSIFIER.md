# 🎯 PLAN: Email Classifier Tool (Spam + Phishing Detector)

**Fecha:** 2026-01-05  
**Objetivo:** Herramienta CLI/API para clasificar emails como SPAM y PHISHING con probabilidades

---

## 📋 REQUISITOS FUNCIONALES

### RF-01: Clasificación Dual
- ✅ Detectar si email es **SPAM** (con probabilidad 0-100%)
- ✅ Detectar si email es **PHISHING** (con probabilidad 0-100%)
- ✅ Mostrar resultado agregado: `HAM`, `SPAM`, `PHISHING`, `SPAM+PHISHING`

### RF-02: Input Múltiple
- ✅ Leer email desde **string directo** (CLI)
- ✅ Leer email desde **archivo .txt**
- ✅ Leer email desde **stdin** (pipe)

### RF-03: Output Formateado
- ✅ Formato **texto** (CLI human-readable)
- ✅ Formato **JSON** (para integración)
- ✅ Nivel de detalle: `simple` | `detailed` | `debug`

### RF-04: Model Management
- ✅ Auto-cargar modelos más recientes
- ✅ Permitir especificar timestamp de modelo
- ✅ Caché en memoria (no recargar en cada predicción)

---

## 🏗️ ARQUITECTURA HEXAGONAL

```
src/ml_engineer_course/
├── domain/                          # ⬡ DOMINIO (sin dependencias externas)
│   ├── __init__.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── email.py                # Email (text, metadata)
│   │   ├── prediction.py           # Prediction result
│   │   └── classifier_metadata.py  # Model metadata
│   ├── ports/
│   │   ├── __init__.py
│   │   ├── model_loader.py         # Protocol para cargar modelos
│   │   ├── predictor.py            # Protocol para predicción
│   │   └── output_formatter.py     # Protocol para formateo
│   └── services/
│       ├── __init__.py
│       └── email_classifier.py     # Servicio de clasificación (orquesta)
│
├── application/                     # ⬡ CASOS DE USO
│   ├── __init__.py
│   └── use_cases/
│       ├── __init__.py
│       ├── classify_email.py       # UC: Clasificar un email
│       └── list_models.py          # UC: Listar modelos disponibles
│
├── infrastructure/                  # ⬡ ADAPTADORES (implementaciones)
│   ├── __init__.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── joblib_model_loader.py  # Carga modelos .joblib
│   │   ├── sklearn_predictor.py    # Predicción con sklearn
│   │   ├── json_formatter.py       # Formato JSON
│   │   └── text_formatter.py       # Formato texto
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py                 # Entry point CLI
│   │   └── commands.py             # Comandos CLI (click/typer)
│   └── api/                         # [FUTURO] FastAPI
│       ├── __init__.py
│       └── main.py
│
└── config/
    ├── __init__.py
    └── settings.py                  # Pydantic Settings
```

---

## 🎨 DISEÑO DE CLASES (POO Atomizada)

### 1️⃣ DOMAIN ENTITIES

#### `Email` (domain/entities/email.py)
```python
@dataclass(frozen=True)
class Email:
    """Email inmutable para clasificación."""
    text: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("Email text cannot be empty")
```

#### `ClassificationResult` (domain/entities/prediction.py)
```python
@dataclass(frozen=True)
class ClassificationResult:
    """Resultado de clasificación dual."""
    email: Email
    spam_probability: float      # 0.0 - 1.0
    phishing_probability: float  # 0.0 - 1.0
    spam_label: str              # "HAM" | "SPAM"
    phishing_label: str          # "LEGIT" | "PHISHING"
    final_verdict: str           # "HAM" | "SPAM" | "PHISHING" | "SPAM+PHISHING"
    confidence: float            # Max probability
    execution_time_ms: float
    models_used: dict[str, str]  # {"spam": "timestamp", "phishing": "timestamp"}
```

#### `ModelMetadata` (domain/entities/classifier_metadata.py)
```python
@dataclass(frozen=True)
class ModelMetadata:
    """Metadata de modelo cargado."""
    name: str                    # "spam_detector" | "phishing_detector"
    timestamp: str
    accuracy: float
    train_samples: int
    vocabulary_size: int
    file_size_mb: float
```

---

### 2️⃣ DOMAIN PORTS (Protocols)

#### `IModelLoader` (domain/ports/model_loader.py)
```python
class IModelLoader(Protocol):
    """Interface para carga de modelos."""
    
    def load(self, model_name: str, timestamp: Optional[str] = None) -> tuple:
        """Returns: (vectorizer, model, metadata)"""
        ...
    
    def list_available(self, model_name: str) -> list[ModelMetadata]:
        ...
```

#### `IPredictor` (domain/ports/predictor.py)
```python
class IPredictor(Protocol):
    """Interface para predicción."""
    
    def predict(self, email: Email) -> ClassificationResult:
        ...
```

#### `IOutputFormatter` (domain/ports/output_formatter.py)
```python
class IOutputFormatter(Protocol):
    """Interface para formateo de output."""
    
    def format(self, result: ClassificationResult, detail_level: str) -> str:
        ...
```

---

### 3️⃣ DOMAIN SERVICE

#### `EmailClassifierService` (domain/services/email_classifier.py)
```python
class EmailClassifierService:
    """Servicio de dominio que orquesta clasificación dual."""
    
    def __init__(
        self,
        spam_predictor: IPredictor,
        phishing_predictor: IPredictor
    ):
        self._spam = spam_predictor
        self._phishing = phishing_predictor
    
    def classify(self, email: Email) -> ClassificationResult:
        """Clasifica email con ambos modelos."""
        # Lógica de orquestación
        ...
```

---

### 4️⃣ APPLICATION USE CASES

#### `ClassifyEmailUseCase` (application/use_cases/classify_email.py)
```python
class ClassifyEmailUseCase:
    """Caso de uso: Clasificar un email."""
    
    def __init__(
        self,
        classifier_service: EmailClassifierService,
        formatter: IOutputFormatter
    ):
        self._classifier = classifier_service
        self._formatter = formatter
    
    def execute(
        self,
        email_text: str,
        detail_level: str = "simple"
    ) -> str:
        """Ejecuta clasificación y retorna resultado formateado."""
        ...
```

---

### 5️⃣ INFRASTRUCTURE ADAPTERS

#### `JoblibModelLoader` (infrastructure/adapters/joblib_model_loader.py)
```python
class JoblibModelLoader:
    """Carga modelos desde .joblib (implementa IModelLoader)."""
    
    def __init__(self, models_dir: Path):
        self._models_dir = models_dir
        self._cache: dict = {}  # Cache de modelos cargados
    
    def load(self, model_name: str, timestamp: Optional[str] = None):
        # Implementación con cache
        ...
```

#### `SklearnPredictor` (infrastructure/adapters/sklearn_predictor.py)
```python
class SklearnPredictor:
    """Predictor usando sklearn (implementa IPredictor)."""
    
    def __init__(
        self,
        vectorizer,
        model,
        metadata: ModelMetadata,
        model_type: str  # "spam" | "phishing"
    ):
        self._vectorizer = vectorizer
        self._model = model
        self._metadata = metadata
        self._type = model_type
    
    def predict(self, email: Email) -> dict:
        # Retorna {"label": "SPAM", "probability": 0.85}
        ...
```

---

## 🔧 COMPONENTES TÉCNICOS

### Settings (Pydantic)
```python
class Settings(BaseSettings):
    models_dir: Path = Path("models")
    default_detail_level: str = "simple"
    cache_models: bool = True
    min_confidence_threshold: float = 0.5
```

### CLI Commands (Typer)
```bash
# Clasificar email desde texto
email-classifier predict "Your account has been suspended..."

# Desde archivo
email-classifier predict --file email.txt

# Formato JSON
email-classifier predict --format json "URGENT! Click here..."

# Con detalle
email-classifier predict --detail debug "Free money!"

# Listar modelos
email-classifier models list

# Info de modelos
email-classifier models info
```

---

## 📦 DEPENDENCIAS NUEVAS

```toml
[project.dependencies]
# Ya tenemos: numpy, pandas, scikit-learn, joblib

# AÑADIR:
typer = "^0.12.0"         # CLI framework
rich = "^13.7.0"          # Terminal formatting
pydantic = "^2.5.0"       # Settings + validation
pydantic-settings = "^2.1.0"
```

---

## 🚀 FASES DE IMPLEMENTACIÓN

### FASE 1: Domain Layer (2-3 horas)
- [ ] Entities: `Email`, `ClassificationResult`, `ModelMetadata`
- [ ] Ports: Protocols para `IModelLoader`, `IPredictor`, `IOutputFormatter`
- [ ] Service: `EmailClassifierService`
- [ ] Tests unitarios (pytest)

### FASE 2: Infrastructure Adapters (3-4 horas)
- [ ] `JoblibModelLoader` con cache
- [ ] `SklearnPredictor` (spam + phishing)
- [ ] `TextFormatter` (output bonito con Rich)
- [ ] `JsonFormatter`
- [ ] Tests de integración

### FASE 3: Application Use Cases (1-2 horas)
- [ ] `ClassifyEmailUseCase`
- [ ] `ListModelsUseCase`
- [ ] Dependency injection setup

### FASE 4: CLI Interface (2-3 horas)
- [ ] Typer commands
- [ ] Argument parsing
- [ ] Error handling
- [ ] Help messages
- [ ] Tests end-to-end

### FASE 5: Polish & Extras (1-2 horas)
- [ ] Logging estructurado
- [ ] Progress bars (Rich)
- [ ] Colored output
- [ ] README con ejemplos
- [ ] Dockerfile (opcional)

---

## 📊 MÉTRICAS DE CALIDAD

### Límites de Complejidad
- ✅ Funciones: **MAX 20 líneas**
- ✅ Archivos: **MAX 250 líneas**
- ✅ Clases: **MAX 7 métodos públicos**
- ✅ Métodos: **MAX 5 parámetros**

### Coverage
- ✅ Unit tests: **>80%**
- ✅ Integration tests: **>60%**
- ✅ Type hints: **100%**
- ✅ Docstrings: **100% en APIs públicas**

### Static Analysis
- ✅ `mypy --strict` sin errores
- ✅ `ruff check` sin warnings
- ✅ `ruff format` aplicado

---

## 🎯 EJEMPLO DE USO FINAL

```bash
$ email-classifier predict "WINNER! You have won $1000! Click here NOW!"

╔══════════════════════════════════════════════════════════════╗
║                    EMAIL CLASSIFICATION                      ║
╚══════════════════════════════════════════════════════════════╝

Email Preview: WINNER! You have won $1000! Click here NOW!

🔴 SPAM Detection:
   Verdict:     SPAM
   Confidence:  85.3%
   
🔴 PHISHING Detection:
   Verdict:     PHISHING
   Confidence:  92.7%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  FINAL VERDICT: SPAM + PHISHING (High Risk)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Models Used:
  • spam_detector: 20260105_194125 (accuracy: 97.4%)
  • phishing_detector: 20260105_195830 (accuracy: 98.1%)
  
Execution Time: 45ms
```

---

## 🔄 FLUJO DE EJECUCIÓN

```
1. CLI recibe comando
   ↓
2. Parse args (Typer)
   ↓
3. Load settings (Pydantic)
   ↓
4. Initialize DI container
   ├── JoblibModelLoader (con cache)
   ├── SklearnPredictor (spam)
   ├── SklearnPredictor (phishing)
   ├── EmailClassifierService
   └── TextFormatter/JsonFormatter
   ↓
5. Execute Use Case
   ├── Create Email entity
   ├── Validate
   ├── classifier_service.classify()
   │   ├── spam_predictor.predict()
   │   └── phishing_predictor.predict()
   ├── Build ClassificationResult
   └── formatter.format()
   ↓
6. Output to stdout
```

---

## 🎓 PRINCIPIOS APLICADOS

### SOLID
- ✅ **S**ingle Responsibility: Cada clase hace UNA cosa
- ✅ **O**pen/Closed: Extensible vía nuevos adapters
- ✅ **L**iskov Substitution: Protocols permiten sustitución
- ✅ **I**nterface Segregation: Ports pequeños y específicos
- ✅ **D**ependency Inversion: Domain no depende de infra

### Clean Architecture
- ✅ Domain en el centro (sin deps externas)
- ✅ Use cases orquestan
- ✅ Adapters implementan detalles
- ✅ Testeable sin I/O

### DDD
- ✅ Entities inmutables
- ✅ Value Objects (Email, Prediction)
- ✅ Domain Services (EmailClassifierService)
- ✅ Ubiquitous Language

---

## 📝 CHECKLIST ANTES DE EMPEZAR

- [ ] ✅ Entorno virtual activo (`ml-course-venv`)
- [ ] ✅ Modelos guardados en `models/`
- [ ] ✅ Dependencias instaladas (`typer`, `rich`, `pydantic`)
- [ ] ✅ Estructura `src/ml_engineer_course/` creada
- [ ] ✅ Git tracking activo
- [ ] ✅ Tests configurados (`pytest.ini`)

---

**ESTADO:** 📋 PLAN COMPLETO - READY TO IMPLEMENT

¿Aprobado para proceder, tronco?
