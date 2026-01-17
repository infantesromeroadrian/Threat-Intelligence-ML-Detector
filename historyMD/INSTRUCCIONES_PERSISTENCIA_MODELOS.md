# Instrucciones: Añadir Persistencia de Modelos a los Notebooks

## Resumen

Añadir 3 secciones nuevas al **final** de los notebooks 02 y 03 para:
1. **Guardar** el modelo entrenado
2. **Cargar** modelos guardados
3. **Probar** el modelo cargado

## Archivos a modificar:

- `notebooks/02-RegresionLogistica-DeteccionSPAM-Simple.ipynb`
- `notebooks/03-RegresionLogistica-DeteccionPhishing.ipynb`

---

## PASO 1: Añadir imports al inicio (Sección 1)

En la primera celda de imports, añadir:

```python
import joblib
from pathlib import Path
from datetime import datetime
```

---

## PASO 2: Nueva sección después de "Resumen" - GUARDAR MODELO

### Markdown Cell:
```markdown
## 11. Guardar Modelo y Componentes

**MLOps Best Practice:** Persistir modelos entrenados para reutilización en producción.
```

### Code Cell:
```python
# ============================================================================
# GUARDAR MODELO Y COMPONENTES
# ============================================================================

# Crear directorio si no existe
models_dir = Path('../models')
models_dir.mkdir(exist_ok=True)

# Timestamp para versioning
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Nombre del modelo
# CAMBIAR según notebook:
#   - Para 02-SPAM: model_name = 'spam_detector'
#   - Para 03-Phishing: model_name = 'phishing_detector'
model_name = 'spam_detector'  # <-- CAMBIAR AQUÍ

print("Guardando modelo y componentes...")
print("="*70)

# 1. Guardar vectorizador TF-IDF
vectorizer_path = models_dir / f'{model_name}_vectorizer_{timestamp}.joblib'
joblib.dump(vectorizer, vectorizer_path)
print(f"✓ Vectorizer: {vectorizer_path.name}")

# 2. Guardar modelo entrenado
model_path = models_dir / f'{model_name}_model_{timestamp}.joblib'
joblib.dump(model, model_path)
print(f"✓ Modelo:     {model_path.name}")

# 3. Guardar metadatos
metadata = {
    'model_name': model_name,
    'timestamp': timestamp,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': float(accuracy),
    'vocabulary_size': len(vectorizer.vocabulary_),
    'max_features': vectorizer.max_features,
    'ngram_range': vectorizer.ngram_range,
}

metadata_path = models_dir / f'{model_name}_metadata_{timestamp}.joblib'
joblib.dump(metadata, metadata_path)
print(f"✓ Metadata:   {metadata_path.name}")

print("="*70)
print("\n✅ Modelo guardado exitosamente en models/")
```

---

## PASO 3: Nueva sección - CARGAR MODELO

### Markdown Cell:
```markdown
## 12. Cargar Modelo Guardado

Funciones para cargar y reutilizar modelos previamente entrenados.
```

### Code Cell:
```python
# ============================================================================
# FUNCIÓN PARA CARGAR MODELOS
# ============================================================================

def load_classifier(model_name, timestamp=None):
    """
    Carga un clasificador guardado.
    
    Args:
        model_name: 'spam_detector' o 'phishing_detector'
        timestamp: Timestamp específico. Si None, carga el más reciente.
    
    Returns:
        tuple: (vectorizer, model, metadata)
    
    Example:
        >>> vec, mod, meta = load_classifier('spam_detector')
    """
    models_dir = Path('../models')
    
    if timestamp is None:
        # Buscar el modelo más reciente
        model_files = sorted(models_dir.glob(f'{model_name}_model_*.joblib'))
        if not model_files:
            raise FileNotFoundError(
                f"No se encontraron modelos '{model_name}' en {models_dir}"
            )
        model_path = model_files[-1]
        # Extraer timestamp del nombre
        timestamp = model_path.stem.split('_')[-1]
    else:
        model_path = models_dir / f'{model_name}_model_{timestamp}.joblib'
    
    # Rutas de componentes
    vectorizer_path = models_dir / f'{model_name}_vectorizer_{timestamp}.joblib'
    metadata_path = models_dir / f'{model_name}_metadata_{timestamp}.joblib'
    
    # Cargar componentes
    print("Cargando modelo...")
    print("="*70)
    
    vectorizer = joblib.load(vectorizer_path)
    print(f"✓ Vectorizer: {vectorizer_path.name}")
    
    model = joblib.load(model_path)
    print(f"✓ Modelo:     {model_path.name}")
    
    metadata = joblib.load(metadata_path)
    print(f"✓ Metadata:   {metadata_path.name}")
    
    print("="*70)
    print("\nINFORMACIÓN DEL MODELO:")
    print("="*70)
    for key, value in metadata.items():
        print(f"  {key:20s}: {value}")
    print("="*70)
    
    return vectorizer, model, metadata


def list_saved_models(model_name):
    """Lista todos los modelos guardados de un tipo."""
    models_dir = Path('../models')
    model_files = sorted(models_dir.glob(f'{model_name}_model_*.joblib'))
    
    if not model_files:
        print(f"❌ No se encontraron modelos '{model_name}'")
        return
    
    print("="*70)
    print(f"MODELOS GUARDADOS: {model_name}")
    print("="*70)
    
    for model_file in model_files:
        timestamp = model_file.stem.split('_')[-1]
        metadata_file = models_dir / f'{model_name}_metadata_{timestamp}.joblib'
        
        try:
            metadata = joblib.load(metadata_file)
            size_mb = model_file.stat().st_size / (1024 * 1024)
            
            print(f"\n📅 {timestamp}")
            print(f"   Tamaño:    {size_mb:.2f} MB")
            print(f"   Accuracy:  {metadata.get('accuracy', 'N/A'):.4f}")
            print(f"   Train:     {metadata.get('train_samples', 'N/A'):,} samples")
            print(f"   Vocab:     {metadata.get('vocabulary_size', 'N/A'):,} palabras")
        except Exception as e:
            print(f"\n⚠️  {timestamp}: Error al leer metadata ({e})")
    
    print("="*70)

# Mostrar funciones creadas
print("✅ Funciones de carga creadas:")
print("   • load_classifier(model_name, timestamp=None)")
print("   • list_saved_models(model_name)")
```

---

## PASO 4: Nueva sección - PROBAR MODELO CARGADO

### Markdown Cell:
```markdown
## 13. Probar Modelo Cargado

Verificar que el modelo cargado funciona correctamente.
```

### Code Cell 1 - Listar modelos:
```python
# Listar modelos disponibles
# CAMBIAR 'spam_detector' según notebook
list_saved_models('spam_detector')
```

### Code Cell 2 - Cargar y probar:
```python
# Cargar el modelo más reciente
# CAMBIAR 'spam_detector' según notebook
model_name_to_load = 'spam_detector'

print("Cargando modelo...\n")
vectorizer_loaded, model_loaded, metadata_loaded = load_classifier(model_name_to_load)

# Email de prueba
# CAMBIAR según el caso de uso:
#   - SPAM: "WINNER! You won $1000! Click NOW!"
#   - Phishing: "URGENT! Your account suspended. Click to verify!"
test_email = "WINNER! You have won $1000! Click here to claim NOW!"

print("\n" + "="*70)
print("PRUEBA CON MODELO CARGADO")
print("="*70)

# Vectorizar
text_tfidf = vectorizer_loaded.transform([test_email])

# Predecir
prediction = model_loaded.predict(text_tfidf)[0]
probability = model_loaded.predict_proba(text_tfidf)[0]

# Formato de salida
# CAMBIAR etiquetas según notebook:
#   - SPAM: label = 'SPAM' if prediction == 1 else 'HAM'
#   - Phishing: label = '🚨 PHISHING' if prediction == 1 else '✅ LEGÍTIMO'
label = 'SPAM' if prediction == 1 else 'HAM'
confidence = probability[prediction] * 100

print(f"Email: {test_email[:80]}...")
print(f"Predicción: {label}")
print(f"Confianza: {confidence:.2f}%")
print(f"Probabilidades: HAM={probability[0]:.3f}, SPAM={probability[1]:.3f}")
print("="*70)

print("\n✅ Modelo cargado funciona correctamente")
```

---

## Valores a CAMBIAR según notebook:

### Para `02-SPAM`:
```python
model_name = 'spam_detector'
test_email = "WINNER! You won $1000! Click NOW!"
label = 'SPAM' if prediction == 1 else 'HAM'
```

### Para `03-Phishing`:
```python
model_name = 'phishing_detector'
test_email = "URGENT! Account suspended. Verify identity immediately!"
label = '🚨 PHISHING' if prediction == 1 else '✅ LEGÍTIMO'
```

---

## Estructura final de archivos:

```
models/
├── spam_detector_vectorizer_20260105_193000.joblib
├── spam_detector_model_20260105_193000.joblib
├── spam_detector_metadata_20260105_193000.joblib
├── phishing_detector_vectorizer_20260105_194500.joblib
├── phishing_detector_model_20260105_194500.joblib
└── phishing_detector_metadata_20260105_194500.joblib
```

---

## Verificación

Después de añadir y ejecutar:

1. **Verificar que se crean 3 archivos** en `models/`
2. **Ejecutar `list_saved_models()`** - debe listar el modelo
3. **Cargar el modelo** - debe funcionar sin errores
4. **Hacer predicción** - debe dar resultado coherente

---

## Notas importantes:

✅ **joblib** es más eficiente que pickle para sklearn
✅ **Timestamp** permite versioning automático
✅ **Metadata** guarda info útil del entrenamiento
✅ **Separar vectorizer y model** permite reutilización independiente

🚫 **NO commitear** modelos a git (son grandes)
✅ **Añadir** `models/*.joblib` al `.gitignore`
