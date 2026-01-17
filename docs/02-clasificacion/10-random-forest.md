# Random Forest

## 1. ¿Qué es Random Forest?

### Concepto: Sabiduría de la Multitud

```
┌────────────────────────────────────────────────────────────────┐
│  RANDOM FOREST = Conjunto de Árboles de Decisión               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Un solo árbol puede equivocarse fácilmente (overfitting)      │
│                                                                │
│  SOLUCIÓN: Crear MUCHOS árboles y que VOTEN                    │
│                                                                │
│       🌳         🌳         🌳         🌳         🌳            │
│      Árbol 1   Árbol 2   Árbol 3   Árbol 4   Árbol 5          │
│         │         │         │         │         │             │
│         ▼         ▼         ▼         ▼         ▼             │
│       SPAM      SPAM      HAM       SPAM      SPAM            │
│                                                                │
│                        VOTACIÓN                                │
│                           │                                    │
│                           ▼                                    │
│                    4 SPAM vs 1 HAM                             │
│                           │                                    │
│                           ▼                                    │
│                   PREDICCIÓN: SPAM ✓                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Analogía: Diagnóstico Médico

```
Un solo doctor puede equivocarse.
Consultar a 100 doctores especializados y tomar la opinión mayoritaria
es mucho más confiable.

Random Forest hace exactamente esto con árboles de decisión.
```

## 2. Ensemble Learning: El Poder del Conjunto

### Tipos de Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│  ENSEMBLE METHODS                                               │
├─────────────────┬───────────────────────────────────────────────┤
│                 │                                               │
│   BAGGING       │  Entrenar modelos en PARALELO                 │
│   (Bootstrap    │  Cada modelo ve datos diferentes              │
│    Aggregating) │  Votación/promedio final                      │
│                 │  Ejemplo: Random Forest                       │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│                 │                                               │
│   BOOSTING      │  Entrenar modelos en SECUENCIA                │
│                 │  Cada modelo corrige errores del anterior     │
│                 │  Ejemplo: XGBoost, AdaBoost                   │
│                 │                                               │
├─────────────────┼───────────────────────────────────────────────┤
│                 │                                               │
│   STACKING      │  Usar predicciones de modelos base            │
│                 │  como features para un meta-modelo            │
│                 │                                               │
└─────────────────┴───────────────────────────────────────────────┘

Random Forest usa BAGGING + selección aleatoria de features
```

## 3. Cómo Funciona Random Forest

### Paso 1: Bootstrap Sampling

```
DATOS ORIGINALES (N muestras):
┌────┬────────────┬──────────┬───────┐
│ ID │ Feature1   │ Feature2 │ Clase │
├────┼────────────┼──────────┼───────┤
│  1 │    0.5     │   100    │  HAM  │
│  2 │    0.8     │   200    │  SPAM │
│  3 │    0.3     │    50    │  HAM  │
│  4 │    0.9     │   300    │  SPAM │
│  5 │    0.2     │    30    │  HAM  │
└────┴────────────┴──────────┴───────┘

BOOTSTRAP SAMPLE (muestreo CON reemplazo):
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Sample Árbol 1  │  │ Sample Árbol 2  │  │ Sample Árbol 3  │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ 1, 3, 3, 4, 5   │  │ 2, 2, 1, 4, 3   │  │ 5, 1, 4, 4, 2   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
   Notar que algunas muestras se REPITEN
   y otras NO aparecen (Out-of-Bag samples)
```

### Paso 2: Selección Aleatoria de Features

```
FEATURES DISPONIBLES: [F1, F2, F3, F4, F5, F6, F7, F8]

En CADA NODO de CADA árbol, solo consideramos un SUBCONJUNTO:

┌─────────────┬────────────────────────────────────────┐
│   Árbol 1   │                                        │
│   Nodo A    │  Evalúa solo: [F2, F5, F7]            │
│   Nodo B    │  Evalúa solo: [F1, F3, F8]            │
│   Nodo C    │  Evalúa solo: [F4, F6, F7]            │
├─────────────┼────────────────────────────────────────┤
│   Árbol 2   │                                        │
│   Nodo A    │  Evalúa solo: [F1, F4, F6]            │
│   Nodo B    │  Evalúa solo: [F2, F3, F5]            │
│   ...       │  ...                                   │
└─────────────┴────────────────────────────────────────┘

¿Cuántas features evaluar?
  • Clasificación: sqrt(n_features) ≈ √8 ≈ 3
  • Regresión: n_features / 3 ≈ 8/3 ≈ 3
```

### Paso 3: Construcción de Árboles

```
         Datos Bootstrap               Datos Bootstrap
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│      ÁRBOL 1            │   │      ÁRBOL 2            │
│  (usa features F2,F5,F7 │   │  (usa features F1,F4,F6 │
│   en nodo raíz)         │   │   en nodo raíz)         │
│                         │   │                         │
│      [F5 > 0.3?]        │   │      [F1 > 100?]        │
│       /      \          │   │       /      \          │
│     Sí        No        │   │     Sí        No        │
│     /          \        │   │     /          \        │
│ [F2>0.5?]    SPAM       │   │   HAM     [F4>50?]      │
│   /   \                 │   │            /   \        │
│ HAM  SPAM               │   │         SPAM  HAM       │
└─────────────────────────┘   └─────────────────────────┘

Cada árbol es DIFERENTE porque:
  1. Ve diferentes datos (bootstrap)
  2. Considera diferentes features en cada split
```

### Paso 4: Agregación (Voting/Averaging)

```
CLASIFICACIÓN - Votación por mayoría:
┌──────────┬────────────┐
│  Árbol   │ Predicción │
├──────────┼────────────┤
│    1     │    SPAM    │
│    2     │    HAM     │
│    3     │    SPAM    │
│    4     │    SPAM    │
│    5     │    HAM     │
│   ...    │    ...     │
│   100    │    SPAM    │
├──────────┼────────────┤
│  TOTAL   │ 65 SPAM    │
│          │ 35 HAM     │
├──────────┼────────────┤
│  FINAL   │   SPAM ✓   │
└──────────┴────────────┘

REGRESIÓN - Promedio:
┌──────────┬────────────┐
│  Árbol   │ Predicción │
├──────────┼────────────┤
│    1     │    85.2    │
│    2     │    92.1    │
│    3     │    88.7    │
│   ...    │    ...     │
├──────────┼────────────┤
│ PROMEDIO │   88.5     │
└──────────┴────────────┘
```

## 4. Out-of-Bag (OOB) Error

### Concepto

```
┌────────────────────────────────────────────────────────────────┐
│  OUT-OF-BAG (OOB) SAMPLES                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Bootstrap sampling deja ~37% de los datos FUERA de cada árbol │
│                                                                │
│  Datos:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                       │
│                                                                │
│  Árbol 1 usa: [1, 1, 3, 5, 5, 7, 8, 8, 9, 10]                 │
│  OOB para Árbol 1: [2, 4, 6]  ← Estos NO se usaron             │
│                                                                │
│  Árbol 2 usa: [2, 3, 3, 4, 6, 7, 7, 8, 9, 9]                  │
│  OOB para Árbol 2: [1, 5, 10]                                  │
│                                                                │
│  Para cada muestra, calculamos su predicción SOLO              │
│  usando los árboles donde NO participó en el entrenamiento     │
│                                                                │
│  OOB Error ≈ Error de validación cruzada (GRATIS!)            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Ventaja del OOB

```
┌────────────────────────┬─────────────────────────────────────┐
│  Método tradicional    │  Con OOB                            │
├────────────────────────┼─────────────────────────────────────┤
│                        │                                     │
│  1. Split train/test   │  1. Usar TODOS los datos            │
│  2. Entrenar           │  2. Entrenar con bootstrap          │
│  3. Evaluar en test    │  3. OOB error ya calculado          │
│                        │                                     │
│  Pierdes 20-30% datos  │  Usas 100% datos + evaluación       │
│  para test             │  gratuita                           │
│                        │                                     │
└────────────────────────┴─────────────────────────────────────┘
```

## 5. Feature Importance

### Cómo se Calcula

```
┌────────────────────────────────────────────────────────────────┐
│  MÉTODOS DE IMPORTANCIA DE FEATURES                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. MEAN DECREASE IN IMPURITY (MDI)                            │
│     ─────────────────────────────                              │
│     Suma de reducciones de Gini/Entropy al usar esa feature    │
│     promediada sobre todos los árboles                         │
│                                                                │
│     Feature "longitud_email":                                  │
│       Árbol 1: reduce Gini en 0.15                             │
│       Árbol 2: reduce Gini en 0.12                             │
│       ...                                                       │
│       Promedio: 0.13 ← Importancia                             │
│                                                                │
│  2. PERMUTATION IMPORTANCE (más robusto)                       │
│     ───────────────────────────────────                        │
│     a. Medir accuracy base                                     │
│     b. Permutar valores de una feature (desordenar)           │
│     c. Medir nuevo accuracy                                    │
│     d. Importancia = caída en accuracy                         │
│                                                                │
│     Si permutar "longitud_email" baja accuracy de 0.95 a 0.75 │
│     → Importancia = 0.20 (muy importante!)                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización Típica

```
Feature Importance (MDI):
────────────────────────────────────────────────────────────

longitud_email     ████████████████████████████  0.28
num_links          ███████████████████████       0.23
palabras_urgentes  ██████████████████            0.18
tiene_adjunto      ████████████████              0.16
hora_envio         ████████                      0.08
dominio_remitente  ██████                        0.05
num_imagenes       ███                           0.02

────────────────────────────────────────────────────────────

Las 3 features más importantes explican ~70% del modelo
```

## 6. Hiperparámetros Principales

### Tabla de Hiperparámetros

```
┌────────────────────┬───────────┬────────────────────────────────┐
│   Parámetro        │  Default  │  Descripción                   │
├────────────────────┼───────────┼────────────────────────────────┤
│ n_estimators       │    100    │ Número de árboles              │
│                    │           │ Más = mejor, pero más lento    │
├────────────────────┼───────────┼────────────────────────────────┤
│ max_features       │  'sqrt'   │ Features a evaluar por split   │
│                    │           │ 'sqrt', 'log2', int, float     │
├────────────────────┼───────────┼────────────────────────────────┤
│ max_depth          │   None    │ Profundidad máxima por árbol   │
│                    │           │ None = sin límite              │
├────────────────────┼───────────┼────────────────────────────────┤
│ min_samples_split  │     2     │ Mínimo de muestras para split  │
├────────────────────┼───────────┼────────────────────────────────┤
│ min_samples_leaf   │     1     │ Mínimo de muestras en hoja     │
├────────────────────┼───────────┼────────────────────────────────┤
│ bootstrap          │   True    │ Usar bootstrap sampling        │
├────────────────────┼───────────┼────────────────────────────────┤
│ oob_score          │  False    │ Calcular OOB error             │
├────────────────────┼───────────┼────────────────────────────────┤
│ n_jobs             │   None    │ Cores para paralelizar         │
│                    │           │ -1 = todos los cores           │
├────────────────────┼───────────┼────────────────────────────────┤
│ random_state       │   None    │ Semilla para reproducibilidad  │
└────────────────────┴───────────┴────────────────────────────────┘
```

### Guía de Ajuste

```
┌────────────────────────────────────────────────────────────────┐
│  AJUSTE DE HIPERPARÁMETROS                                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  OVERFITTING? (train alto, test bajo)                          │
│  ─────────────────────────────────────                         │
│    • Reducir max_depth (ej: 10, 20)                            │
│    • Aumentar min_samples_split (ej: 5, 10)                    │
│    • Aumentar min_samples_leaf (ej: 2, 5)                      │
│    • Reducir max_features                                      │
│                                                                │
│  UNDERFITTING? (train y test bajos)                            │
│  ──────────────────────────────────                            │
│    • Aumentar n_estimators                                     │
│    • Aumentar max_depth                                        │
│    • Reducir min_samples_split                                 │
│                                                                │
│  MUY LENTO?                                                    │
│  ─────────                                                     │
│    • Usar n_jobs=-1 (paralelizar)                              │
│    • Reducir n_estimators                                      │
│    • Limitar max_depth                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 7. Implementación en Python

### Código Básico

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Datos de ejemplo
X = np.random.randn(1000, 10)  # 1000 muestras, 10 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Clasificación binaria

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar Random Forest
rf = RandomForestClassifier(
    n_estimators=100,        # 100 árboles
    max_depth=10,            # Limitar profundidad
    min_samples_split=5,     # Mínimo para split
    max_features='sqrt',     # sqrt(n_features)
    oob_score=True,          # Calcular OOB error
    n_jobs=-1,               # Usar todos los cores
    random_state=42          # Reproducibilidad
)

rf.fit(X_train, y_train)

# Evaluación
print(f"Accuracy Train: {rf.score(X_train, y_train):.3f}")
print(f"Accuracy Test:  {rf.score(X_test, y_test):.3f}")
print(f"OOB Score:      {rf.oob_score_:.3f}")
```

### Feature Importance

```python
import matplotlib.pyplot as plt

# Obtener importancias
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualizar
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [f"F{i}" for i in indices])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# Tabla de importancias
print("\nFeature Importance Ranking:")
print("=" * 40)
for i in indices:
    print(f"Feature {i}: {importances[i]:.4f}")
```

### Tuning con GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Definir grid de parámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

# Grid Search con Cross-Validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor F1 Score (CV): {grid_search.best_score_:.3f}")

# Evaluar mejor modelo
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

## 8. Random Forest vs Árbol de Decisión

### Comparación

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│      Aspecto        │  Árbol de Decisión  │   Random Forest     │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Overfitting         │      Alto           │      Bajo           │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Varianza            │      Alta           │      Baja           │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Bias                │      Bajo           │      Bajo           │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Interpretabilidad   │      Alta           │      Media          │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Velocidad Train     │      Rápido         │      Medio          │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Velocidad Predict   │      Rápido         │      Medio          │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Manejo de ruido     │      Malo           │      Bueno          │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Estabilidad         │      Baja           │      Alta           │
│ (cambio en datos)   │                     │                     │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

### Trade-off Bias-Variance

```
         Error
            │
            │    \
            │     \      Varianza (Random Forest)
            │      \____________________________
            │
            │    \
            │     \     Varianza (Árbol solo)
            │      \
            │       \
            │        \
            │         \
            └──────────────────────────────── Complejidad del modelo

Random Forest reduce VARIANZA manteniendo BIAS bajo
(promediando muchos árboles de alta varianza)
```

## 9. Ejemplo Práctico: Detección de Intrusiones de Red

### Dataset y Preprocesamiento

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Simular datos de tráfico de red
np.random.seed(42)
n_samples = 5000

# Features de conexiones de red
data = {
    'duracion': np.random.exponential(10, n_samples),
    'bytes_enviados': np.random.exponential(5000, n_samples),
    'bytes_recibidos': np.random.exponential(10000, n_samples),
    'paquetes_enviados': np.random.poisson(50, n_samples),
    'paquetes_recibidos': np.random.poisson(100, n_samples),
    'conexiones_fallidas': np.random.poisson(2, n_samples),
    'num_puertos_destino': np.random.poisson(3, n_samples),
    'flag_syn': np.random.binomial(1, 0.3, n_samples),
    'flag_fin': np.random.binomial(1, 0.2, n_samples),
    'hora_del_dia': np.random.randint(0, 24, n_samples),
}

df = pd.DataFrame(data)

# Crear etiquetas basadas en patrones sospechosos
df['es_ataque'] = (
    (df['conexiones_fallidas'] > 3) |  # Muchos fallos → scan
    (df['num_puertos_destino'] > 5) |   # Port scanning
    ((df['bytes_enviados'] > 20000) & (df['duracion'] < 1)) |  # DDoS
    ((df['flag_syn'] == 1) & (df['flag_fin'] == 0) & (df['duracion'] < 0.1))  # SYN flood
).astype(int)

# Añadir ruido
ruido = np.random.binomial(1, 0.05, n_samples)
df['es_ataque'] = (df['es_ataque'] + ruido) % 2

print(f"Distribución de clases:")
print(df['es_ataque'].value_counts())
print(f"\nRatio ataque: {df['es_ataque'].mean():.2%}")
```

### Entrenamiento y Evaluación

```python
# Preparar datos
X = df.drop('es_ataque', axis=1)
y = df['es_ataque']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar (opcional para RF, pero buena práctica)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Para datos desbalanceados
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# Métricas
print("=" * 60)
print("RESULTADOS DEL MODELO")
print("=" * 60)
print(f"\nOOB Score: {rf.oob_score_:.4f}")
print(f"Train Accuracy: {rf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")

y_pred = rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Normal', 'Ataque']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### Análisis de Features

```python
# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE - INDICADORES DE ATAQUE")
print("=" * 60)
for _, row in feature_importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"{row['feature']:25} {bar} {row['importance']:.4f}")

# Interpretación
print("\n" + "=" * 60)
print("INTERPRETACIÓN PARA CIBERSEGURIDAD")
print("=" * 60)
top_features = feature_importance.head(3)['feature'].tolist()
print(f"""
Los principales indicadores de ataque son:
  1. {top_features[0]}: Valores anómalos sugieren actividad maliciosa
  2. {top_features[1]}: Patrón común en reconocimiento de red
  3. {top_features[2]}: Indicador de técnicas de evasión

Recomendación: Configurar alertas SIEM basadas en umbrales
de estas features para detección temprana.
""")
```

### Output Esperado

```
============================================================
RESULTADOS DEL MODELO
============================================================

OOB Score: 0.9125
Train Accuracy: 0.9875
Test Accuracy: 0.9150

Classification Report:
              precision    recall  f1-score   support

      Normal       0.93      0.95      0.94       720
      Ataque       0.87      0.83      0.85       280

    accuracy                           0.91      1000
   macro avg       0.90      0.89      0.89      1000
weighted avg       0.91      0.91      0.91      1000

Confusion Matrix:
[[684  36]
 [ 49 231]]

============================================================
FEATURE IMPORTANCE - INDICADORES DE ATAQUE
============================================================
conexiones_fallidas       ██████████████████████████ 0.2145
num_puertos_destino       ████████████████████████ 0.1987
bytes_enviados            ██████████████████ 0.1523
flag_syn                  ███████████████ 0.1234
duracion                  ████████████ 0.0987
...
```

## 10. Ventajas y Desventajas

```
┌─────────────────────────────────────────────────────────────────┐
│  VENTAJAS DE RANDOM FOREST                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✓ Muy robusto contra overfitting (vs árbol individual)        │
│  ✓ Maneja bien datos con ruido y outliers                      │
│  ✓ No requiere mucho preprocesamiento (scaling no necesario)   │
│  ✓ Funciona bien con features numéricas y categóricas          │
│  ✓ Proporciona importancia de features                         │
│  ✓ OOB error como estimación gratuita del error                │
│  ✓ Paralelizable (n_jobs=-1)                                   │
│  ✓ Pocos hiperparámetros críticos                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DE RANDOM FOREST                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✗ Menos interpretable que un árbol individual                 │
│  ✗ Puede ser lento con muchos árboles y datos grandes          │
│  ✗ Modelo grande en memoria (todos los árboles)                │
│  ✗ No extrapola bien fuera del rango de entrenamiento          │
│  ✗ Puede tener problemas con datos muy desbalanceados          │
│  ✗ No captura relaciones lineales tan bien como modelos        │
│    lineales                                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 11. Cuándo Usar Random Forest

```
┌────────────────────────────────────────────────────────────────┐
│  CASOS DE USO IDEALES                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Clasificación de malware                                    │
│  ✓ Detección de intrusiones/anomalías                          │
│  ✓ Análisis de fraude                                          │
│  ✓ Predicción de churn                                         │
│  ✓ Diagnóstico médico                                          │
│  ✓ Cuando necesitas importancia de features                    │
│  ✓ Como baseline robusto antes de probar modelos complejos     │
│  ✓ Datos con muchas features (alta dimensionalidad)            │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  EVITAR CUANDO                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Necesitas modelo muy interpretable (usar árbol simple)      │
│  ✗ Latencia crítica en producción (muchos árboles = lento)     │
│  ✗ Datos principalmente texto/secuencias (usar DL)             │
│  ✗ Memoria muy limitada                                        │
│  ✗ Relaciones estrictamente lineales (usar regresión)          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 12. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  RANDOM FOREST - RESUMEN                                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CONCEPTO:                                                     │
│    Ensemble de múltiples árboles de decisión                   │
│    Cada árbol ve datos y features diferentes                   │
│    Predicción final por votación/promedio                      │
│                                                                │
│  COMPONENTES CLAVE:                                            │
│    • Bootstrap Sampling: muestreo con reemplazo                │
│    • Random Feature Selection: subconjunto por nodo            │
│    • Agregación: votación (clasificación) o promedio (reg)     │
│                                                                │
│  HIPERPARÁMETROS PRINCIPALES:                                  │
│    • n_estimators: número de árboles (más = mejor)             │
│    • max_depth: profundidad máxima (controla overfitting)      │
│    • max_features: features por split ('sqrt' default)         │
│                                                                │
│  VENTAJAS:                                                     │
│    • Robusto, pocos hiperparámetros críticos                   │
│    • Feature importance incluida                                │
│    • OOB error como validación gratuita                        │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Detección de intrusiones                                  │
│    • Clasificación de malware                                  │
│    • Análisis de logs                                          │
│    • Detección de phishing                                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Support Vector Machines (SVM)
