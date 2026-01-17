# 💾 Persistencia de Modelos ML

## Resumen Ejecutivo

Se ha implementado un sistema completo de persistencia de modelos para los notebooks de Regresión Logística.

---

## 📁 Archivos Creados

```
Ml-Engineer/
├── models/                          # Directorio para modelos (gitignored)
│   └── .gitkeep
├── .gitignore                       # Actualizado con exclusiones
├── INSTRUCCIONES_PERSISTENCIA_MODELOS.md  # Guía detallada
└── README_PERSISTENCIA.md          # Este archivo
```

---

## 🎯 Qué Hace

### 1. **Guardar Modelos**
Después de entrenar, guarda:
- **Vectorizador TF-IDF** → `{model_name}_vectorizer_{timestamp}.joblib`
- **Modelo LogisticRegression** → `{model_name}_model_{timestamp}.joblib`
- **Metadata** → `{model_name}_metadata_{timestamp}.joblib`

### 2. **Cargar Modelos**
Funciones para:
- Cargar el modelo más reciente automáticamente
- Cargar un modelo específico por timestamp
- Listar todos los modelos guardados

### 3. **Reutilizar Modelos**
Sin reentrenar:
- Hacer predicciones con modelos cargados
- Ver metadata de entrenamientos previos
- Comparar versiones de modelos

---

## 🚀 Cómo Usar

### En los Notebooks (después de ejecutar):

```python
# 1. Guardar modelo (se ejecuta automáticamente)
# Se crean 3 archivos en models/

# 2. Listar modelos guardados
list_saved_models('spam_detector')

# 3. Cargar modelo más reciente
vectorizer, model, metadata = load_classifier('spam_detector')

# 4. Usar modelo cargado
text = "Test email"
text_vec = vectorizer.transform([text])
prediction = model.predict(text_vec)
```

### Desde Python scripts:

```python
import joblib
from pathlib import Path

# Cargar componentes
models_dir = Path('models')
vectorizer = joblib.load(models_dir / 'spam_detector_vectorizer_20260105_193000.joblib')
model = joblib.load(models_dir / 'spam_detector_model_20260105_193000.joblib')

# Predecir
text_vec = vectorizer.transform(["SPAM email text"])
prediction = model.predict(text_vec)
```

---

## 📊 Metadata Guardada

Cada modelo incluye:

```python
{
    'model_name': 'spam_detector',
    'timestamp': '20260105_193000',
    'train_samples': 4458,
    'test_samples': 1115,
    'accuracy': 0.9876,
    'vocabulary_size': 5000,
    'max_features': 5000,
    'ngram_range': (1, 2)
}
```

---

## 🔧 Implementación

### Tecnología:
- **joblib**: Serialización eficiente (mejor que pickle para sklearn)
- **Pathlib**: Manejo de rutas multiplataforma
- **Timestamp**: Versioning automático

### Ventajas:
✅ **No reentrenar** - Reutilizar modelos entrenados
✅ **Versioning** - Timestamp único para cada modelo
✅ **Trazabilidad** - Metadata completa del entrenamiento
✅ **Portabilidad** - Archivos .joblib compartibles
✅ **Eficiencia** - joblib optimizado para arrays NumPy

---

## 📋 Checklist de Implementación

### Para `02-SPAM`:
- [ ] Añadir imports (joblib, Path, datetime)
- [ ] Añadir sección "Guardar Modelo"
- [ ] Cambiar `model_name = 'spam_detector'`
- [ ] Añadir sección "Cargar Modelo"
- [ ] Añadir sección "Probar Modelo Cargado"
- [ ] Ejecutar notebook completo
- [ ] Verificar 3 archivos en `models/`

### Para `03-Phishing`:
- [ ] Añadir imports (joblib, Path, datetime)
- [ ] Añadir sección "Guardar Modelo"
- [ ] Cambiar `model_name = 'phishing_detector'`
- [ ] Añadir sección "Cargar Modelo"
- [ ] Añadir sección "Probar Modelo Cargado"
- [ ] Ejecutar notebook completo
- [ ] Verificar 3 archivos en `models/`

---

## 🗂️ Estructura Final

```
models/
├── .gitkeep
├── spam_detector_vectorizer_20260105_193000.joblib    (5 MB)
├── spam_detector_model_20260105_193000.joblib         (2 MB)
├── spam_detector_metadata_20260105_193000.joblib      (1 KB)
├── phishing_detector_vectorizer_20260105_194500.joblib (15 MB)
├── phishing_detector_model_20260105_194500.joblib      (5 MB)
└── phishing_detector_metadata_20260105_194500.joblib   (1 KB)
```

**Total**: ~30 MB (gitignored, no se suben a repo)

---

## 🔒 Seguridad

### .gitignore configurado:
```gitignore
# Models (NO commitear)
models/*.joblib
models/*.pkl
models/*.h5

# Data (NO commitear)
data/*.csv
```

### Por qué NO commitear modelos:
- ❌ Archivos grandes (>10 MB)
- ❌ Cambios frecuentes
- ❌ Específicos del entrenamiento local
- ✅ Se regeneran fácilmente ejecutando notebook

### Alternativas para compartir:
- **Git LFS** - Large File Storage
- **DVC** - Data Version Control
- **MLflow** - ML experiment tracking
- **Cloud Storage** - S3, GCS, Azure Blob

---

## 📖 Referencias

### Documentación:
- [joblib](https://joblib.readthedocs.io/)
- [sklearn model persistence](https://scikit-learn.org/stable/model_persistence.html)
- [pathlib](https://docs.python.org/3/library/pathlib.html)

### Archivos de ayuda:
- `INSTRUCCIONES_PERSISTENCIA_MODELOS.md` - Guía detallada paso a paso
- `model_persistence_cells.md` - Código de las celdas (deprecated)

---

## 🎓 Próximos Pasos

### Mejoras futuras:
1. **Model Registry** - Registro centralizado de modelos
2. **Experiment Tracking** - MLflow, Weights & Biases
3. **Model Monitoring** - Performance en producción
4. **A/B Testing** - Comparar versiones de modelos
5. **AutoML** - Búsqueda automática de hiperparámetros

---

## ✅ Resumen

**3 secciones nuevas en cada notebook:**
1. Guardar modelo (3 archivos: vectorizer, model, metadata)
2. Cargar modelo (funciones `load_classifier` y `list_saved_models`)
3. Probar modelo cargado (verificación)

**Resultado:**
- ✅ Modelos persistidos y reutilizables
- ✅ Versioning automático con timestamps
- ✅ Metadata completa de entrenamiento
- ✅ .gitignore actualizado
- ✅ Listo para MLOps profesional

---

**Creado:** 2026-01-05  
**Autor:** AIR  
**Proyecto:** ML Engineer Course
