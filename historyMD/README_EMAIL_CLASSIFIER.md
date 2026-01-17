# 📧 Email Classifier - SPAM & PHISHING Detection

[![Coverage](https://img.shields.io/badge/coverage-92.20%25-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-108%20passed-success)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![Code Style](https://img.shields.io/badge/code%20style-ruff-black)]()

**Herramienta CLI profesional para detección dual de SPAM y PHISHING usando Machine Learning**

---

## ✨ Características

- 🎯 **Detección Dual**: SPAM + PHISHING en una sola clasificación
- 🚀 **CLI Rápida**: Comando `email-classifier` listo para usar
- 📊 **Múltiples Formatos**: Output en texto (Rich) o JSON
- 📁 **Input Flexible**: Desde argumento, archivo o stdin
- 🎨 **Rich UI**: Colores, emojis y tablas en terminal
- ⚙️ **Configurable**: Via CLI, env vars o .env file
- 🧪 **Testeado**: 108 tests con 92% coverage
- 🏗️ **Clean Architecture**: Hexagonal + DDD + SOLID

---

## 🚀 Instalación Rápida

```bash
# Clonar repo
git clone <repo-url>
cd Ml-Engineer

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar
pip install -e .

# Verificar
email-classifier --help
```

---

## 📖 Uso

### Clasificar Email

**Desde texto:**
```bash
email-classifier predict "WINNER! You won $1000!"
# Output: 🚨 SPAM+PHISHING (95.4% confidence)
```

**Desde archivo:**
```bash
email-classifier predict --file email.txt
# Output: 🔴 PHISHING (98.2% confidence)
```

**Desde stdin:**
```bash
cat email.txt | email-classifier predict
echo "Hi John, meeting at 3 PM" | email-classifier predict
# Output: ✅ HAM (97.9% confidence)
```

### Formatos de Output

**Simple (default):**
```bash
$ email-classifier predict "Test"
✅ HAM (95.1% confidence)
```

**Detailed:**
```bash
$ email-classifier predict "Test" --detail detailed
  Email          Test email     
  🟢 SPAM        HAM (95.1%)    
  🟢 PHISHING    LEGIT (69.5%)  
  ✅ VERDICT     HAM (LOW)      
```

**JSON:**
```bash
$ email-classifier predict "Test" --format json
{
  "verdict": "HAM",
  "confidence": 0.9510,
  "is_malicious": false,
  "risk_level": "LOW"
}
```

### Gestionar Modelos

```bash
# Listar versiones disponibles
email-classifier models list

# Info del último modelo
email-classifier models info

# Listar phishing detector
email-classifier models list phishing_detector
```

---

## 🎯 Ejemplos Reales

### 1. Email Normal
```bash
$ email-classifier predict "Hi team, please review the attached report."
✅ HAM (98.5% confidence)
```

### 2. Spam Obvio
```bash
$ email-classifier predict "CONGRATULATIONS! You won $1M! Click NOW!"
🚨 SPAM+PHISHING (97.3% confidence)
```

### 3. Phishing Sofisticado
```bash
$ email-classifier predict --file phishing_email.txt --detail detailed

  Email          URGENT! Your PayPal account has been suspended...
  🟢 SPAM        HAM (61.3%)
  🔴 PHISHING    PHISHING (98.6%)
  🔴 VERDICT     PHISHING (HIGH)
```

---

## 🔧 Opciones Avanzadas

### Custom Models Directory
```bash
email-classifier --models-dir /custom/path predict "Test"
```

### Environment Variables
```bash
export EMAIL_CLASSIFIER_MODELS_DIR=/custom/models
export EMAIL_CLASSIFIER_VERBOSE=true
email-classifier predict "Test"
```

### .env File
```env
EMAIL_CLASSIFIER_MODELS_DIR=/custom/models
EMAIL_CLASSIFIER_DEFAULT_FORMAT=json
EMAIL_CLASSIFIER_VERBOSE=true
```

---

## 🛠️ Integración

### Bash Script
```bash
#!/bin/bash
for email in inbox/*.txt; do
    result=$(email-classifier predict --file "$email" --format json)
    verdict=$(echo "$result" | jq -r '.verdict')
    
    if [ "$verdict" != "HAM" ]; then
        echo "⚠️  Suspicious: $email"
        mv "$email" quarantine/
    fi
done
```

### Python API
```python
import subprocess
import json

def classify_email(text: str) -> dict:
    result = subprocess.run(
        ["email-classifier", "predict", text, "--format", "json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

# Usar
classification = classify_email("WINNER! Click here!")
if classification["is_malicious"]:
    print(f"🚨 Blocked: {classification['verdict']}")
```

---

## 🏗️ Arquitectura

### Clean Architecture (Hexagonal)

```
┌─────────────────────────────────────────┐
│            CLI Interface                │
│         (Typer + Rich)                  │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         Application Layer               │
│  (Use Cases + DI Container)             │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│          Domain Layer                   │
│  (Entities + Services + Ports)          │
│  ⬡ Email, Prediction, Metadata         │
│  ⬡ EmailClassifierService              │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│      Infrastructure Layer               │
│  (Adapters: Joblib, Sklearn, Rich)     │
└─────────────────────────────────────────┘
```

### Principios Aplicados

✅ **SOLID**
- Single Responsibility
- Open/Closed
- Liskov Substitution
- Interface Segregation
- Dependency Inversion

✅ **DDD**
- Value Objects (Email, Prediction)
- Domain Services
- Ubiquitous Language

✅ **Clean Code**
- Funciones <20 líneas
- Archivos <250 líneas
- Type hints 100%
- Docstrings completos

---

## 📊 Testing

```bash
# Run all tests
pytest tests/

# Coverage report
pytest tests/ --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

**Coverage:** 92.20% (108 tests)

---

## 📁 Estructura del Proyecto

```
Ml-Engineer/
├── src/ml_engineer_course/
│   ├── domain/              # Core business logic
│   ├── application/         # Use cases + DI
│   ├── infrastructure/      # Adapters + CLI
│   └── config/              # Settings
│
├── models/                  # Trained models (.joblib)
├── tests/                   # 108 tests
├── notebooks/               # Training notebooks
├── docs/                    # Documentation
├── pyproject.toml           # Project config
└── README.md
```

---

## 🧪 Development

### Setup Dev Environment
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linter
ruff check src/

# Run type checker
mypy src/

# Run tests
pytest tests/ -v
```

### Training New Models

Los notebooks en `notebooks/` contienen el entrenamiento:
- `02-RegresionLogistica-DeteccionSPAM-ConPersistencia.ipynb`
- `03-RegresionLogistica-DeteccionPhishing-ConPersistencia.ipynb`

Los modelos se guardan automáticamente en `models/` con timestamp.

---

## 📝 Configuración

### Settings Disponibles

```python
# Via code
from ml_engineer_course.config import Settings

settings = Settings(
    models_dir=Path("/custom/models"),
    default_format="json",
    default_detail_level="detailed",
    verbose=True
)

# Via environment
EMAIL_CLASSIFIER_MODELS_DIR=/custom/models
EMAIL_CLASSIFIER_DEFAULT_FORMAT=json
EMAIL_CLASSIFIER_VERBOSE=true
```

---

## 🤝 Contribuir

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/amazing`)
3. Commit cambios (`git commit -m 'Add amazing feature'`)
4. Push al branch (`git push origin feature/amazing`)
5. Abre un Pull Request

**Requisitos:**
- Tests pasando
- Coverage >80%
- Type hints completos
- Docstrings en APIs públicas
- ruff check sin warnings

---

## 📜 Licencia

MIT License - Ver LICENSE file

---

## 🙏 Agradecimientos

- **scikit-learn** - ML models
- **Rich** - Terminal UI
- **Typer** - CLI framework
- **Pydantic** - Settings validation

---

## 📧 Contacto

**AIR** - ML Engineer Course

---

## 🎓 Aprendizajes Clave

Este proyecto demuestra:
- ✅ Arquitectura Hexagonal en Python
- ✅ Domain-Driven Design práctico
- ✅ Dependency Injection manual
- ✅ Testing exhaustivo (unit + integration)
- ✅ CLI profesional con Typer
- ✅ Type safety con mypy
- ✅ Clean Code principles
- ✅ MLOps basics (model versioning, caching)

**Perfect para portfolio de ML Engineer** 💼

---

**Hecho con ❤️ por AIR**
