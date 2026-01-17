# Celdas para Persistencia de Modelos

## Para insertar DESPUÉS del entrenamiento en ambos notebooks

### Celda 1: Imports adicionales (agregar al inicio)
```python
import joblib
from pathlib import Path
from datetime import datetime
```

### Celda 2: Guardar Modelo (después de entrenar)
```python
# ============================================================================
# GUARDAR MODELO Y COMPONENTES
# ============================================================================

# Crear directorio si no existe
models_dir = Path('../models')
models_dir.mkdir(exist_ok=True)

# Timestamp para versioning
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Nombre del modelo (cambiar según notebook)
model_name = 'spam_detector'  # O 'phishing_detector'

# Guardar vectorizador TF-IDF
vectorizer_path = models_dir / f'{model_name}_vectorizer_{timestamp}.joblib'
joblib.dump(vectorizer, vectorizer_path)
print(f"✓ Vectorizer guardado: {vectorizer_path}")

# Guardar modelo entrenado
model_path = models_dir / f'{model_name}_model_{timestamp}.joblib'
joblib.dump(model, model_path)
print(f"✓ Modelo guardado: {model_path}")

# Guardar metadatos
metadata = {
    'model_name': model_name,
    'timestamp': timestamp,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': accuracy,
    'vocabulary_size': len(vectorizer.vocabulary_),
    'max_features': vectorizer.max_features,
    'ngram_range': vectorizer.ngram_range,
}

metadata_path = models_dir / f'{model_name}_metadata_{timestamp}.joblib'
joblib.dump(metadata, metadata_path)
print(f"✓ Metadata guardado: {metadata_path}")

print("\n" + "="*70)
print("ARCHIVOS GUARDADOS:")
print("="*70)
print(f"1. Vectorizer: {vectorizer_path.name}")
print(f"2. Model:      {model_path.name}")
print(f"3. Metadata:   {metadata_path.name}")
print("="*70)
```

### Celda 3: Cargar Modelo (nueva sección al final)
```python
# ============================================================================
# CARGAR MODELO GUARDADO
# ============================================================================

def load_spam_classifier(timestamp=None):
    """
    Carga un clasificador de spam guardado.
    
    Args:
        timestamp: Timestamp específico del modelo. Si None, carga el más reciente.
    
    Returns:
        tuple: (vectorizer, model, metadata)
    """
    models_dir = Path('../models')
    model_name = 'spam_detector'  # O 'phishing_detector'
    
    if timestamp is None:
        # Buscar el modelo más reciente
        model_files = sorted(models_dir.glob(f'{model_name}_model_*.joblib'))
        if not model_files:
            raise FileNotFoundError(f"No se encontraron modelos en {models_dir}")
        model_path = model_files[-1]
        # Extraer timestamp del nombre
        timestamp = model_path.stem.split('_')[-1]
    else:
        model_path = models_dir / f'{model_name}_model_{timestamp}.joblib'
    
    # Cargar componentes
    vectorizer_path = models_dir / f'{model_name}_vectorizer_{timestamp}.joblib'
    metadata_path = models_dir / f'{model_name}_metadata_{timestamp}.joblib'
    
    print("Cargando modelo...")
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    print(f"✓ Modelo cargado: {model_path.name}")
    print(f"✓ Vectorizer cargado: {vectorizer_path.name}")
    print(f"✓ Metadata cargado: {metadata_path.name}")
    print("\n" + "="*70)
    print("INFORMACIÓN DEL MODELO:")
    print("="*70)
    for key, value in metadata.items():
        print(f"{key:20s}: {value}")
    print("="*70)
    
    return vectorizer, model, metadata

# Ejemplo de uso:
# vectorizer_loaded, model_loaded, metadata_loaded = load_spam_classifier()
```

### Celda 4: Probar Modelo Cargado
```python
# ============================================================================
# PROBAR MODELO CARGADO
# ============================================================================

# Cargar el modelo más reciente
vectorizer_loaded, model_loaded, metadata_loaded = load_spam_classifier()

# Probar con un ejemplo
test_email = "WINNER! You have won $1000! Click here to claim your prize NOW!"

# Vectorizar
text_tfidf = vectorizer_loaded.transform([test_email])

# Predecir
prediction = model_loaded.predict(text_tfidf)[0]
probability = model_loaded.predict_proba(text_tfidf)[0]

label = 'SPAM' if prediction == 1 else 'HAM'
confidence = probability[prediction] * 100

print("\n" + "="*70)
print("PRUEBA DEL MODELO CARGADO")
print("="*70)
print(f"Email: {test_email}")
print(f"Predicción: {label}")
print(f"Confianza: {confidence:.2f}%")
print("="*70)
```

### Celda 5: Listar Modelos Guardados
```python
# ============================================================================
# LISTAR MODELOS GUARDADOS
# ============================================================================

def list_saved_models(model_name='spam_detector'):
    """Lista todos los modelos guardados."""
    models_dir = Path('../models')
    model_files = sorted(models_dir.glob(f'{model_name}_model_*.joblib'))
    
    if not model_files:
        print(f"No se encontraron modelos de tipo '{model_name}'")
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
            
            print(f"\nTimestamp: {timestamp}")
            print(f"  Archivo: {model_file.name}")
            print(f"  Tamaño: {size_mb:.2f} MB")
            print(f"  Accuracy: {metadata.get('accuracy', 'N/A'):.4f}")
            print(f"  Train samples: {metadata.get('train_samples', 'N/A'):,}")
            print(f"  Vocabulary: {metadata.get('vocabulary_size', 'N/A'):,} palabras")
        except Exception as e:
            print(f"\nTimestamp: {timestamp}")
            print(f"  ⚠️  Error al leer metadata: {e}")
    
    print("="*70)

# Listar modelos
list_saved_models('spam_detector')  # O 'phishing_detector'
```

## Estructura final de archivos guardados:

```
models/
├── spam_detector_vectorizer_20260105_193045.joblib
├── spam_detector_model_20260105_193045.joblib
├── spam_detector_metadata_20260105_193045.joblib
├── phishing_detector_vectorizer_20260105_194523.joblib
├── phishing_detector_model_20260105_194523.joblib
└── phishing_detector_metadata_20260105_194523.joblib
```

## Metadata incluye:
- model_name
- timestamp
- train_samples
- test_samples
- accuracy
- vocabulary_size
- max_features
- ngram_range
