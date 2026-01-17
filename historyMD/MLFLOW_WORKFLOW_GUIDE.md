# 🚀 MLflow + Git LFS Workflow Guide

Guía completa para trabajar con el sistema de versionado de modelos MLflow + Git LFS.

---

## 📊 **ARQUITECTURA DEL SISTEMA**

```
┌─────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT CYCLE                        │
└─────────────────────────────────────────────────────────────┘

1. Experimenta       2. Compara         3. Promociona       4. Versionado
   (MLflow)             (MLflow UI)        (Local)             (Git LFS)

┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
│ Jupyter  │──────▶│ MLflow   │──────▶│ models/  │──────▶│ GitHub   │
│ Notebook │       │ Tracking │       │ v1.x.x/  │       │ Releases │
└──────────┘       └──────────┘       └──────────┘       └──────────┘
    │                   │                   │                   │
    │ Log metrics       │ Compare runs      │ Best model        │ Tagged
    │ Log params        │ Select best       │ Symlink           │ version
    │ Save artifacts    │                   │                   │
```

---

## 🔄 **WORKFLOW COMPLETO**

### **FASE 1: Experimentación (MLflow)**

#### 1.1 Modificar Notebooks para usar MLflow

Añade esto al principio de tus notebooks de entrenamiento:

```python
# En notebooks/02-RegresionLogistica-DeteccionSPAM-ConPersistencia.ipynb

import mlflow
import mlflow.sklearn
from mlflow_config import setup_mlflow

# Setup MLflow
setup_mlflow()

# Iniciar run
with mlflow.start_run(run_name="spam_detector_experiment_1"):
    
    # ═══════════════════════════════════════════════
    # 1. Log Parameters (ANTES de entrenar)
    # ═══════════════════════════════════════════════
    mlflow.log_param("algorithm", "LogisticRegression")
    mlflow.log_param("max_features_tfidf", 5000)
    mlflow.log_param("ngram_range", "(1, 2)")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("training_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    
    # ═══════════════════════════════════════════════
    # 2. Entrenar modelo (TU CÓDIGO EXISTENTE)
    # ═══════════════════════════════════════════════
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # ═══════════════════════════════════════════════
    # 3. Evaluar y Log Metrics
    # ═══════════════════════════════════════════════
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)
    
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    
    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_metric("true_negatives", int(cm[0, 0]))
    mlflow.log_metric("false_positives", int(cm[0, 1]))
    mlflow.log_metric("false_negatives", int(cm[1, 0]))
    mlflow.log_metric("true_positives", int(cm[1, 1]))
    
    # ═══════════════════════════════════════════════
    # 4. Log Model Artifacts
    # ═══════════════════════════════════════════════
    mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(vectorizer, "vectorizer")
    
    # Log metadata as dict
    metadata = {
        "model_name": "spam_detector",
        "version": "1.0.0",
        "algorithm": "LogisticRegression",
        "features": "TF-IDF",
        "accuracy": accuracy,
        "trained_on": datetime.now().isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    import json
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    mlflow.log_artifact("metadata.json")
    
    # ═══════════════════════════════════════════════
    # 5. Log Tags
    # ═══════════════════════════════════════════════
    mlflow.set_tag("model_type", "spam_detector")
    mlflow.set_tag("stage", "development")
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("author", "Adrian Infantes")
    
    # ═══════════════════════════════════════════════
    # 6. Print Summary
    # ═══════════════════════════════════════════════
    print(f"✅ Run logged to MLflow")
    print(f"   Run ID: {mlflow.active_run().info.run_id}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1: {f1:.4f}")
    print(f"   AUC: {auc:.4f}")

# NOTA: El mismo patrón para phishing_detector en notebook 03
```

#### 1.2 Ejecutar Experimentos

```bash
# Correr notebooks con tracking
jupyter nbconvert --execute notebooks/02-RegresionLogistica-DeteccionSPAM-ConPersistencia.ipynb
jupyter nbconvert --execute notebooks/03-RegresionLogistica-DeteccionPhishing-ConPersistencia.ipynb

# O desde Jupyter Lab normalmente
jupyter lab
```

---

### **FASE 2: Comparación (MLflow UI)**

#### 2.1 Iniciar MLflow UI

```bash
mlflow ui
# Abre: http://localhost:5000
```

#### 2.2 Comparar Experimentos

En la UI de MLflow:

1. **Ver todos los runs:**
   - Columnas: Run Name, Parameters, Metrics, Duration
   - Ordenar por accuracy, f1_score, etc.

2. **Comparar múltiples runs:**
   - Seleccionar 2+ runs (checkboxes)
   - Click "Compare"
   - Ver gráficos de métricas lado a lado

3. **Buscar mejores modelos:**
   - Filter por `accuracy > 0.95`
   - Sort by `f1_score` descending

4. **Inspeccionar Run:**
   - Click en run específico
   - Ver parámetros, métricas, artifacts
   - Descargar model artifacts

#### 2.3 Seleccionar Mejor Modelo

Criterios de selección:

```
SPAM Detector:
✅ Accuracy > 95%
✅ Precision > 94% (minimizar falsos positivos)
✅ Recall > 90% (no perder spam real)
✅ F1 > 92%

PHISHING Detector:
✅ Accuracy > 92%
✅ Precision > 90%
✅ Recall > 88% (crítico: no perder phishing)
✅ F1 > 89%
```

---

### **FASE 3: Promoción a Producción (Local)**

#### 3.1 Extraer Modelo desde MLflow

Una vez identificado el mejor run en MLflow UI:

```bash
# Anotar el run_id del mejor modelo (ej: a1b2c3d4e5f6)
RUN_ID="a1b2c3d4e5f6"
MODEL_TYPE="spam_detector"
VERSION="v1.1.0"

# Copiar artifacts de MLflow a models/
cp mlruns/0/${RUN_ID}/artifacts/model/model.pkl models/${MODEL_TYPE}_model_${VERSION}.joblib
cp mlruns/0/${RUN_ID}/artifacts/vectorizer/vectorizer.pkl models/${MODEL_TYPE}_vectorizer_${VERSION}.joblib
cp mlruns/0/${RUN_ID}/artifacts/metadata.json models/${MODEL_TYPE}_metadata_${VERSION}.json

# Convertir .pkl a .joblib si es necesario
python << EOF
import joblib
model = joblib.load("mlruns/0/${RUN_ID}/artifacts/model/model.pkl")
vectorizer = joblib.load("mlruns/0/${RUN_ID}/artifacts/vectorizer/vectorizer.pkl")

joblib.dump(model, "models/${MODEL_TYPE}_model_${VERSION}.joblib")
joblib.dump(vectorizer, "models/${MODEL_TYPE}_vectorizer_${VERSION}.joblib")
print("✅ Converted to .joblib format")
EOF
```

#### 3.2 Actualizar Symlinks "latest"

```bash
cd models/

# Update latest symlinks
ln -sf ${MODEL_TYPE}_model_${VERSION}.joblib ${MODEL_TYPE}_model_latest.joblib
ln -sf ${MODEL_TYPE}_vectorizer_${VERSION}.joblib ${MODEL_TYPE}_vectorizer_latest.joblib
ln -sf ${MODEL_TYPE}_metadata_${VERSION}.json ${MODEL_TYPE}_metadata_latest.json

cd ..
```

#### 3.3 Verificar Modelo Localmente

```bash
# Test con CLI
email-classifier predict "URGENT! You won $1M!" --detail detailed

# Verificar que carga el modelo correcto
# Debería mostrar metadata actualizada
```

---

### **FASE 4: Versionado en Git (Git LFS)**

#### 4.1 Añadir Modelos a Git LFS

```bash
# Git LFS ya trackea *.joblib (configurado en setup)
# Simplemente add y commit

git add models/${MODEL_TYPE}_*.joblib
git add models/${MODEL_TYPE}_*.json

# Commit con mensaje descriptivo
git commit -m "feat(models): release ${MODEL_TYPE} ${VERSION}

Improvements:
- Accuracy: 95.2% → 96.1% (+0.9%)
- Precision: 94.8% → 95.3% (+0.5%)
- F1 Score: 93.9% → 94.7% (+0.8%)

Training:
- Dataset: 6,000 emails (↑1,000)
- Features: TF-IDF 5000 features
- Algorithm: LogisticRegression (C=1.5)

MLflow Run ID: ${RUN_ID}
Trained: $(date +'%Y-%m-%d')"
```

#### 4.2 Tag Release

```bash
# Create annotated tag
git tag -a ${VERSION} -m "Release ${VERSION} - ${MODEL_TYPE}

SPAM Detector v1.1.0
====================

Performance Improvements:
- Accuracy: 96.1% (↑0.9%)
- Precision: 95.3% (↑0.5%)
- Recall: 93.5% (↑0.4%)
- F1 Score: 94.7% (↑0.8%)

Changes:
- Increased training dataset to 6,000 emails
- Tuned hyperparameter C=1.5 (from 1.0)
- Added bi-gram features for better context

Metrics:
- False Positive Rate: 4.7% (↓0.5%)
- False Negative Rate: 6.5% (↓0.4%)

Testing:
- Validated on 1,500 test emails
- Cross-validated with 5-fold CV

MLflow:
- Experiment: spam-phishing-detection
- Run ID: ${RUN_ID}
- Tracking URI: file://./mlruns

Backwards Compatible: Yes
Migration Required: No"

# Ver tags
git tag -l -n9 ${VERSION}
```

#### 4.3 Push a GitHub

```bash
# Push commits y tags
git push origin main
git push origin ${VERSION}

# Verificar en GitHub
# https://github.com/infantesromeroadrian/ML-Spam-Phising-Detector/releases
```

#### 4.4 (Opcional) Crear GitHub Release

```bash
# Via GitHub CLI
gh release create ${VERSION} \
  --title "${VERSION} - ${MODEL_TYPE} Performance Upgrade" \
  --notes-file RELEASE_NOTES.md \
  models/${MODEL_TYPE}_*_${VERSION}.*

# O via web:
# 1. Ir a Releases en GitHub
# 2. "Draft a new release"
# 3. Tag: v1.1.0
# 4. Upload binaries (opcional, ya están en LFS)
```

---

## 📊 **EJEMPLO COMPLETO: Nuevo Modelo SPAM v1.1.0**

### Paso a paso real:

```bash
# ═══════════════════════════════════════════════════════════
# 1. EXPERIMENTACIÓN (Jupyter Notebook)
# ═══════════════════════════════════════════════════════════

jupyter lab notebooks/02-RegresionLogistica-DeteccionSPAM-ConPersistencia.ipynb

# Modificas hiperparámetros:
# - C = 1.5 (antes 1.0)
# - max_features = 7000 (antes 5000)

# Ejecutas celdas → MLflow logea automáticamente
# Run ID generado: a1b2c3d4e5f6

# ═══════════════════════════════════════════════════════════
# 2. COMPARACIÓN (MLflow UI)
# ═══════════════════════════════════════════════════════════

mlflow ui
# Abre http://localhost:5000

# Comparas runs:
# - Run anterior (v1.0.0): 95.2% accuracy
# - Run nuevo (a1b2c3d4e5f6): 96.1% accuracy ← MEJOR

# Decides promocionar a1b2c3d4e5f6

# ═══════════════════════════════════════════════════════════
# 3. PROMOCIÓN
# ═══════════════════════════════════════════════════════════

RUN_ID="a1b2c3d4e5f6"
VERSION="v1.1.0"

# Extraer de MLflow
python << 'EXTRACT'
import mlflow
import joblib

# Load from MLflow
logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.sklearn.load_model(logged_model)
vectorizer = mlflow.sklearn.load_model(f'runs:/{RUN_ID}/vectorizer')

# Save to models/
joblib.dump(model, f'models/spam_detector_model_{VERSION}.joblib')
joblib.dump(vectorizer, f'models/spam_detector_vectorizer_{VERSION}.joblib')
print(f'✅ Extracted {VERSION}')
EXTRACT

# Update symlinks
cd models/
ln -sf spam_detector_model_${VERSION}.joblib spam_detector_model_latest.joblib
ln -sf spam_detector_vectorizer_${VERSION}.joblib spam_detector_vectorizer_latest.joblib
cd ..

# Test
email-classifier predict "WIN FREE IPHONE!" --detail detailed
# ✅ Verifica que accuracy mostrada = 96.1%

# ═══════════════════════════════════════════════════════════
# 4. VERSIONADO
# ═══════════════════════════════════════════════════════════

git add models/spam_detector_*_${VERSION}.joblib
git commit -m "feat(models): release spam_detector ${VERSION} - 96.1% accuracy

Improvements over v1.0.0:
- Accuracy: 95.2% → 96.1% (+0.9pp)
- Precision: 94.8% → 95.3% (+0.5pp)
- F1 Score: 93.9% → 94.7% (+0.8pp)

Changes:
- Hyperparameter C: 1.0 → 1.5
- Max features: 5000 → 7000
- Training samples: 5000 → 6000

MLflow Run: ${RUN_ID}"

git tag -a ${VERSION} -m "Release ${VERSION}"
git push origin main --tags

# ═══════════════════════════════════════════════════════════
# DONE! 🎉
# ═══════════════════════════════════════════════════════════

echo "✅ Modelo ${VERSION} desplegado y versionado"
```

---

## 📋 **CHEATSHEET: Comandos Rápidos**

```bash
# MLflow UI
mlflow ui

# Ver experimentos
mlflow experiments list

# Buscar runs con accuracy > 95%
mlflow runs list --experiment-id 0 --filter "metrics.accuracy > 0.95"

# Git LFS status
git lfs ls-files

# Ver tamaño de archivos LFS
git lfs ls-files --size

# Ver historial de modelos
git log --oneline models/

# Ver tags de versiones
git tag -l "v*"

# Rollback a versión anterior
cd models/
ln -sf spam_detector_model_v1.0.0.joblib spam_detector_model_latest.joblib
```

---

## 🔍 **TROUBLESHOOTING**

### Problema: MLflow no encuentra experimentos

```bash
# Verificar tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"
# Debe ser: file://./mlruns

# Re-setup
python mlflow_config.py
```

### Problema: Git LFS no sube archivos grandes

```bash
# Ver qué está trackeado
git lfs track

# Añadir patrón si falta
git lfs track "models/*.joblib"

# Ver status LFS
git lfs status

# Ver logs
git lfs logs last
```

### Problema: Modelo no carga después de update

```bash
# Verificar symlinks
ls -la models/*_latest.joblib

# Recrear symlinks
cd models/
rm *_latest.joblib
ln -s spam_detector_model_v1.1.0.joblib spam_detector_model_latest.joblib
# etc...
```

---

## 🎯 **BEST PRACTICES**

### ✅ DO

- Log TODOS los hiperparámetros en MLflow
- Log métricas de validación Y test
- Usar run names descriptivos: `spam_v2_increased_C`
- Taggear runs: `stage=production`, `experiment=hyperparameter_tuning`
- Commit modelos solo cuando accuracy mejora >0.5%
- Usar semantic versioning: v1.0.0, v1.1.0, v2.0.0
- Documentar cambios en mensajes de commit

### ❌ DON'T

- No commitear archivos grandes sin Git LFS
- No sobreescribir modelos de producción sin backup
- No subir models/ directamente a git (usar LFS)
- No olvidar actualizar symlinks `latest`
- No hacer push sin probar modelo localmente

---

## 📚 **RECURSOS**

- **MLflow Docs:** https://mlflow.org/docs/latest/index.html
- **Git LFS:** https://git-lfs.github.com/
- **GitHub Releases:** https://docs.github.com/en/repositories/releasing-projects-on-github

---

**¡Workflow profesional de ML listo!** 🚀
