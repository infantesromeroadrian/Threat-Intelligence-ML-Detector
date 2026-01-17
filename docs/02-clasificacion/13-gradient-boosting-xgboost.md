# Gradient Boosting y XGBoost

## 1. ¿Qué es Boosting?

### Concepto: Aprender de los Errores

```
┌────────────────────────────────────────────────────────────────┐
│  BOOSTING = Entrenar modelos SECUENCIALMENTE                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Cada modelo nuevo se enfoca en los ERRORES del anterior       │
│                                                                │
│  Modelo 1:  Predice → Comete errores en muestras {3, 7, 12}    │
│       │                                                        │
│       ▼                                                        │
│  Modelo 2:  Se enfoca en {3, 7, 12} → Errores en {7, 15}       │
│       │                                                        │
│       ▼                                                        │
│  Modelo 3:  Se enfoca en {7, 15} → Menos errores               │
│       │                                                        │
│       ▼                                                        │
│      ...                                                       │
│       │                                                        │
│       ▼                                                        │
│  Predicción Final: Combinación ponderada de todos              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Bagging vs Boosting

```
┌─────────────────────────────────────────────────────────────────┐
│  BAGGING (Random Forest)          BOOSTING (Gradient Boosting)  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Modelos en PARALELO              Modelos en SECUENCIA          │
│                                                                 │
│    🌳  🌳  🌳  🌳                    🌳 → 🌳 → 🌳 → 🌳            │
│    │   │   │   │                    │     │     │     │         │
│    ▼   ▼   ▼   ▼                    └──┬──┴──┬──┴──┬──┘         │
│    ────┬────                            │     │     │           │
│        │                          Cada uno corrige              │
│     VOTACIÓN                      los errores del anterior      │
│                                                                 │
│  Reduce VARIANZA                  Reduce BIAS                   │
│  (promediando)                    (aprendiendo errores)         │
│                                                                 │
│  Más robusto                      Mayor accuracy potencial      │
│  Menos riesgo overfitting         Más riesgo overfitting        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Gradient Boosting: La Idea

### Descenso por Gradiente

```
┌────────────────────────────────────────────────────────────────┐
│  GRADIENT BOOSTING                                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  En lugar de ponderar muestras, usamos el GRADIENTE            │
│  de la función de pérdida para guiar el aprendizaje            │
│                                                                │
│  PASO 1: Entrenar modelo inicial F₀                            │
│                                                                │
│  PASO 2: Calcular RESIDUOS (errores)                           │
│          rᵢ = yᵢ - F₀(xᵢ)                                      │
│                                                                │
│  PASO 3: Entrenar nuevo modelo h₁ para predecir RESIDUOS       │
│          h₁ aprende a corregir los errores de F₀               │
│                                                                │
│  PASO 4: Actualizar modelo                                     │
│          F₁(x) = F₀(x) + α·h₁(x)                               │
│          (α = learning rate)                                   │
│                                                                │
│  PASO 5: Repetir pasos 2-4 hasta M modelos                     │
│          F_M(x) = F₀(x) + α·h₁(x) + α·h₂(x) + ... + α·h_M(x)  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización del Proceso

```
Iteración 1: Modelo inicial
────────────────────────────
Datos reales:    ●     ●  ●     ●   ●
Predicción F₀:   ─────────────────────
Residuos:        ↑     ↑  ↑     ↓   ↑   (diferencia real - predicho)


Iteración 2: Corregir residuos
────────────────────────────────
Predicción F₀:   ─────────────────────
+ h₁ (corrección):    ╱╲  ╱╲
= F₁:            ───╱──╲╱──╲──────   (más cerca de los datos)


Iteración N: Ajuste fino
──────────────────────────
F_N:             ──●──●──●──●──●──   (casi perfecto)

Cada iteración reduce el error residual
```

## 3. Algoritmos de Gradient Boosting

### Evolución Histórica

```
┌─────────────────────────────────────────────────────────────────┐
│  EVOLUCIÓN DE GRADIENT BOOSTING                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  2001: Gradient Boosting Machine (GBM)                          │
│        └── Algoritmo original de Friedman                       │
│                                                                 │
│  2014: XGBoost (eXtreme Gradient Boosting)                      │
│        └── Regularización + optimizaciones                      │
│        └── El rey de Kaggle por años                            │
│                                                                 │
│  2017: LightGBM (Microsoft)                                     │
│        └── Crecimiento por hoja (leaf-wise)                     │
│        └── Muy rápido con datos grandes                         │
│                                                                 │
│  2017: CatBoost (Yandex)                                        │
│        └── Manejo nativo de categóricas                         │
│        └── Ordered boosting (reduce overfitting)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Comparación de Implementaciones

```
┌─────────────────┬────────────────┬─────────────┬────────────────┐
│    Aspecto      │   XGBoost      │  LightGBM   │   CatBoost     │
├─────────────────┼────────────────┼─────────────┼────────────────┤
│ Velocidad       │     ★★★☆      │   ★★★★★    │     ★★★★       │
├─────────────────┼────────────────┼─────────────┼────────────────┤
│ Accuracy        │    ★★★★★      │    ★★★★     │    ★★★★★       │
├─────────────────┼────────────────┼─────────────┼────────────────┤
│ Memoria         │     ★★★☆      │   ★★★★★    │     ★★★★       │
├─────────────────┼────────────────┼─────────────┼────────────────┤
│ Categóricas     │   Requiere     │   Básico    │   ★★★★★       │
│                 │   encoding     │             │   (nativo)     │
├─────────────────┼────────────────┼─────────────┼────────────────┤
│ GPU Support     │      ★★★★     │    ★★★★     │    ★★★★★       │
├─────────────────┼────────────────┼─────────────┼────────────────┤
│ Overfitting     │   Necesita     │   Necesita  │   Menos        │
│ control         │   tuning       │   tuning    │   propenso     │
├─────────────────┼────────────────┼─────────────┼────────────────┤
│ Datos pequeños  │    ★★★★★      │    ★★★★     │    ★★★★★       │
├─────────────────┼────────────────┼─────────────┼────────────────┤
│ Datos grandes   │     ★★★★      │   ★★★★★    │     ★★★★       │
└─────────────────┴────────────────┴─────────────┴────────────────┘

RECOMENDACIÓN:
  • Empezar con XGBoost (más documentación, comunidad grande)
  • Datos muy grandes → LightGBM
  • Muchas categóricas → CatBoost
```

## 4. XGBoost: Características Clave

### Regularización

```
┌────────────────────────────────────────────────────────────────┐
│  FUNCIÓN OBJETIVO DE XGBOOST                                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)                                  │
│        ─────────    ─────────                                  │
│        Loss         Regularización                             │
│        (error)      (penalización complejidad)                 │
│                                                                │
│  Donde Ω(f) = γT + ½λ||w||²                                   │
│                                                                │
│    T = número de hojas del árbol                               │
│    w = pesos de las hojas                                      │
│    γ = penalización por número de hojas                        │
│    λ = regularización L2 sobre pesos                           │
│                                                                │
│  GBM tradicional: Solo Loss (sin regularización)               │
│  XGBoost: Loss + Regularización = Mejor generalización         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Optimizaciones de XGBoost

```
┌────────────────────────────────────────────────────────────────┐
│  OPTIMIZACIONES QUE HACEN A XGBOOST RÁPIDO                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. PARALELIZACIÓN                                             │
│     • Construcción de árboles en paralelo (nivel por nivel)    │
│     • Búsqueda de splits en paralelo                           │
│                                                                │
│  2. HISTOGRAMAS                                                │
│     • Agrupa valores continuos en bins                         │
│     • Reduce complejidad de O(n) a O(num_bins)                 │
│                                                                │
│  3. SPARSITY AWARENESS                                         │
│     • Manejo eficiente de valores faltantes                    │
│     • Aprende dirección por defecto para NaN                   │
│                                                                │
│  4. CACHE OPTIMIZATION                                         │
│     • Acceso secuencial a memoria                              │
│     • Block structure para datos grandes                       │
│                                                                │
│  5. OUT-OF-CORE                                                │
│     • Puede procesar datos que no caben en RAM                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 5. Hiperparámetros Principales

### Tabla de Hiperparámetros

```
┌────────────────────┬───────────┬────────────────────────────────┐
│    Parámetro       │  Default  │  Descripción                   │
├────────────────────┼───────────┼────────────────────────────────┤
│                    │           │  CONTROL DE BOOSTING           │
├────────────────────┼───────────┼────────────────────────────────┤
│ n_estimators       │    100    │  Número de árboles             │
│ (num_boost_round)  │           │  Más = más complejo            │
├────────────────────┼───────────┼────────────────────────────────┤
│ learning_rate      │    0.3    │  Peso de cada árbol (η)        │
│ (eta)              │           │  Menor = más robusto           │
├────────────────────┼───────────┼────────────────────────────────┤
│                    │           │  CONTROL DE ÁRBOLES            │
├────────────────────┼───────────┼────────────────────────────────┤
│ max_depth          │     6     │  Profundidad máxima            │
│                    │           │  Menor = menos overfitting     │
├────────────────────┼───────────┼────────────────────────────────┤
│ min_child_weight   │     1     │  Suma mínima de peso en hoja   │
│                    │           │  Mayor = más conservador       │
├────────────────────┼───────────┼────────────────────────────────┤
│ subsample          │    1.0    │  Fracción de muestras por árbol│
│                    │           │  <1 = reduce overfitting       │
├────────────────────┼───────────┼────────────────────────────────┤
│ colsample_bytree   │    1.0    │  Fracción de features por árbol│
├────────────────────┼───────────┼────────────────────────────────┤
│                    │           │  REGULARIZACIÓN                │
├────────────────────┼───────────┼────────────────────────────────┤
│ reg_alpha (α)      │     0     │  Regularización L1             │
├────────────────────┼───────────┼────────────────────────────────┤
│ reg_lambda (λ)     │     1     │  Regularización L2             │
├────────────────────┼───────────┼────────────────────────────────┤
│ gamma (γ)          │     0     │  Mínima reducción de pérdida   │
│                    │           │  para hacer un split           │
└────────────────────┴───────────┴────────────────────────────────┘
```

### Guía de Tuning

```
┌────────────────────────────────────────────────────────────────┐
│  ESTRATEGIA DE TUNING PARA XGBOOST                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  PASO 1: Fijar learning_rate alto (0.1-0.3) y n_estimators    │
│          Ajustar max_depth, min_child_weight                   │
│                                                                │
│  PASO 2: Tune subsample, colsample_bytree                      │
│          Valores típicos: 0.6-0.9                              │
│                                                                │
│  PASO 3: Tune regularización (gamma, reg_alpha, reg_lambda)    │
│                                                                │
│  PASO 4: Reducir learning_rate (0.01-0.1)                      │
│          Aumentar n_estimators proporcionalmente               │
│          Usar early_stopping para encontrar óptimo             │
│                                                                │
└────────────────────────────────────────────────────────────────┘

REGLAS GENERALES:
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  OVERFITTING? (train alto, test bajo)                          │
│    • Reducir max_depth (3-6)                                   │
│    • Reducir learning_rate                                     │
│    • Aumentar min_child_weight                                 │
│    • Reducir subsample (0.6-0.8)                               │
│    • Aumentar reg_alpha, reg_lambda                            │
│                                                                │
│  UNDERFITTING? (train y test bajos)                            │
│    • Aumentar n_estimators                                     │
│    • Aumentar max_depth                                        │
│    • Aumentar learning_rate (temporalmente)                    │
│    • Reducir regularización                                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 6. Implementación en Python

### XGBoost Básico

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Datos de ejemplo
np.random.seed(42)
X = np.random.randn(1000, 10)
y = (X[:, 0] + X[:, 1] * 2 + X[:, 2] ** 2 > 1).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear y entrenar XGBoost
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_clf.fit(X_train, y_train)

# Evaluar
y_pred = xgb_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### Con Early Stopping

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Split incluyendo validation set
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Entrenar con early stopping
xgb_clf = xgb.XGBClassifier(
    n_estimators=1000,  # Número alto, early stopping lo detiene
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50  # Detener si no mejora en 50 rondas
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

print(f"\nMejor iteración: {xgb_clf.best_iteration}")
print(f"Mejor score: {xgb_clf.best_score:.4f}")

# Evaluar en test
y_pred = xgb_clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Grid Search con XGBoost

```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Definir grid de parámetros
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Grid Search
xgb_clf = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

grid_search = GridSearchCV(
    xgb_clf,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor F1 (CV): {grid_search.best_score_:.4f}")

# Evaluar
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Feature Importance

```python
import matplotlib.pyplot as plt
import xgboost as xgb

# Después de entrenar...

# Importancia por gain (más interpretable)
importance = xgb_clf.get_booster().get_score(importance_type='gain')

# Ordenar por importancia
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance (por gain):")
print("=" * 40)
for feature, score in sorted_importance[:10]:
    print(f"{feature}: {score:.2f}")

# Visualización
xgb.plot_importance(xgb_clf, importance_type='gain', max_num_features=10)
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()
```

## 7. LightGBM

### Código Básico

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Entrenar LightGBM
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    num_leaves=31,  # Específico de LightGBM
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='logloss',
    callbacks=[lgb.early_stopping(50)]
)

y_pred = lgb_clf.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Diferencias con XGBoost

```
┌────────────────────────────────────────────────────────────────┐
│  LIGHTGBM vs XGBOOST                                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CRECIMIENTO DEL ÁRBOL:                                        │
│                                                                │
│  XGBoost: Level-wise (nivel por nivel)                         │
│                                                                │
│       [raíz]                                                   │
│      /      \                                                  │
│    [1]      [2]     ← Todo el nivel a la vez                   │
│   /  \     /  \                                                │
│  [3] [4] [5] [6]    ← Siguiente nivel completo                 │
│                                                                │
│  LightGBM: Leaf-wise (por hoja, más profundidad selectiva)     │
│                                                                │
│       [raíz]                                                   │
│      /      \                                                  │
│    [1]      [2]                                                │
│   /  \                                                         │
│  [3] [4]            ← Solo expande hoja con mayor ganancia     │
│   |                                                            │
│  [5]                ← Sigue la mejor hoja                      │
│                                                                │
│  Ventaja: Más rápido, mejor para datos grandes                 │
│  Riesgo: Más propenso a overfitting                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 8. CatBoost

### Código Básico

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# CatBoost maneja categóricas nativamente
cat_clf = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=False
)

# Si tienes features categóricas:
# cat_features = [0, 3, 5]  # índices de columnas categóricas
# cat_clf.fit(X_train, y_train, cat_features=cat_features)

cat_clf.fit(X_train, y_train, eval_set=(X_val, y_val))

y_pred = cat_clf.predict(X_test)
print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

## 9. Ejemplo Práctico: Detección de Malware

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Simular dataset de características de PE (Portable Executable)
np.random.seed(42)
n_samples = 5000

# Features típicas de análisis estático de malware
data = {
    # Características del header PE
    'size_of_code': np.random.exponential(100000, n_samples),
    'size_of_initialized_data': np.random.exponential(50000, n_samples),
    'num_sections': np.random.poisson(5, n_samples),
    'entropy_code': np.random.uniform(4, 8, n_samples),
    'entropy_data': np.random.uniform(3, 8, n_samples),

    # Características de imports
    'num_imports': np.random.poisson(50, n_samples),
    'suspicious_imports': np.random.poisson(2, n_samples),
    'num_dlls': np.random.poisson(10, n_samples),

    # Características de strings
    'num_urls': np.random.poisson(3, n_samples),
    'num_ips': np.random.poisson(1, n_samples),
    'suspicious_strings': np.random.poisson(5, n_samples),

    # Otras
    'packed': np.random.binomial(1, 0.3, n_samples),
    'has_debug_info': np.random.binomial(1, 0.4, n_samples),
    'signed': np.random.binomial(1, 0.6, n_samples),
}

df = pd.DataFrame(data)

# Crear etiquetas basadas en patrones de malware
df['es_malware'] = (
    (df['entropy_code'] > 7.0) |  # Alta entropía = empaquetado
    (df['suspicious_imports'] > 3) |  # Muchos imports sospechosos
    ((df['packed'] == 1) & (df['signed'] == 0)) |  # Empaquetado y sin firmar
    (df['num_urls'] > 5) |  # Muchas URLs
    (df['suspicious_strings'] > 10)  # Muchos strings sospechosos
).astype(int)

# Añadir ruido
ruido = np.random.binomial(1, 0.05, n_samples)
df['es_malware'] = (df['es_malware'] + ruido) % 2

print("Distribución de clases:")
print(df['es_malware'].value_counts())
print(f"Ratio malware: {df['es_malware'].mean():.2%}")

# Preparar datos
X = df.drop('es_malware', axis=1)
y = df['es_malware']

# Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Entrenar XGBoost con early stopping
xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Balance
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    early_stopping_rounds=50
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"\nMejor iteración: {xgb_clf.best_iteration}")

# Evaluar
y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("DETECTOR DE MALWARE - RESULTADOS")
print("=" * 60)

print(f"\nAccuracy: {(y_pred == y_test).mean():.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Benigno', 'Malware']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE - INDICADORES DE MALWARE")
print("=" * 60)

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_clf.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"{row['feature']:25} {bar} {row['importance']:.4f}")

# Interpretación
print("\n" + "=" * 60)
print("INTERPRETACIÓN PARA CIBERSEGURIDAD")
print("=" * 60)
top3 = importance.head(3)['feature'].tolist()
print(f"""
Los principales indicadores de malware identificados son:

1. {top3[0]}: Principal característica discriminante
2. {top3[1]}: Segundo indicador más importante
3. {top3[2]}: Tercer indicador relevante

Recomendaciones:
- Priorizar análisis de archivos con alta {top3[0]}
- Configurar alertas automáticas basadas en estos indicadores
- Considerar sandbox analysis para archivos sospechosos
""")

# Ejemplos de predicción
print("\n" + "=" * 60)
print("EJEMPLOS DE CLASIFICACIÓN")
print("=" * 60)

# Crear ejemplos sintéticos
ejemplos = pd.DataFrame({
    'size_of_code': [50000, 200000, 100000],
    'size_of_initialized_data': [20000, 80000, 40000],
    'num_sections': [4, 8, 5],
    'entropy_code': [5.5, 7.8, 6.2],
    'entropy_data': [4.0, 7.5, 5.0],
    'num_imports': [30, 100, 50],
    'suspicious_imports': [0, 8, 2],
    'num_dlls': [8, 20, 10],
    'num_urls': [0, 10, 2],
    'num_ips': [0, 5, 1],
    'suspicious_strings': [2, 20, 5],
    'packed': [0, 1, 0],
    'has_debug_info': [1, 0, 1],
    'signed': [1, 0, 1]
})

for i, row in ejemplos.iterrows():
    pred = xgb_clf.predict(row.values.reshape(1, -1))[0]
    proba = xgb_clf.predict_proba(row.values.reshape(1, -1))[0]

    resultado = "MALWARE 🚨" if pred == 1 else "BENIGNO ✓"
    confianza = proba[pred]

    print(f"\nEjemplo {i+1}: {resultado} (confianza: {confianza:.1%})")
    print(f"  Entropy código: {row['entropy_code']:.1f}")
    print(f"  Imports sospechosos: {row['suspicious_imports']}")
    print(f"  Empaquetado: {'Sí' if row['packed'] else 'No'}")
    print(f"  Firmado: {'Sí' if row['signed'] else 'No'}")
```

## 10. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────────────┐
│  VENTAJAS DE GRADIENT BOOSTING                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Estado del arte en datos tabulares                          │
│  ✓ Maneja bien features de diferentes tipos                    │
│  ✓ No requiere escalado de features                            │
│  ✓ Proporciona feature importance                              │
│  ✓ Maneja bien valores faltantes (XGBoost, LightGBM)           │
│  ✓ Regularización incorporada                                  │
│  ✓ Early stopping para evitar overfitting                      │
│  ✓ Altamente optimizado y paralelizable                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DE GRADIENT BOOSTING                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Muchos hiperparámetros que ajustar                          │
│  ✗ Propenso a overfitting si no se ajusta bien                 │
│  ✗ Entrenamiento secuencial (no tan paralelizable)             │
│  ✗ Puede ser lento con datasets muy grandes                    │
│  ✗ Menos interpretable que un árbol simple                     │
│  ✗ Sensible a outliers (aunque menos que otros)                │
│  ✗ No extrapola bien fuera del rango de entrenamiento          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 11. Cuándo Usar Gradient Boosting

```
┌────────────────────────────────────────────────────────────────┐
│  CASOS DE USO IDEALES                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Datos tabulares estructurados                               │
│  ✓ Competiciones de ML (Kaggle, etc.)                          │
│  ✓ Clasificación de malware                                    │
│  ✓ Detección de fraude                                         │
│  ✓ Scoring de crédito                                          │
│  ✓ Predicción de churn                                         │
│  ✓ Ranking y recomendación                                     │
│  ✓ Cuando Random Forest no es suficiente                       │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  EVITAR CUANDO                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Datos de imagen/texto/audio (usar Deep Learning)            │
│  ✗ Datos muy pequeños (<100 muestras)                          │
│  ✗ Necesitas interpretabilidad total                           │
│  ✗ Latencia de predicción muy crítica                          │
│  ✗ No tienes tiempo para tuning                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 12. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  GRADIENT BOOSTING - RESUMEN                                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CONCEPTO:                                                     │
│    Entrenar árboles SECUENCIALMENTE                            │
│    Cada árbol corrige los errores del anterior                 │
│    Predicción = suma ponderada de todos los árboles            │
│                                                                │
│  IMPLEMENTACIONES:                                             │
│    • XGBoost: El más popular, buen balance                     │
│    • LightGBM: Más rápido para datos grandes                   │
│    • CatBoost: Mejor para features categóricas                 │
│                                                                │
│  HIPERPARÁMETROS CLAVE:                                        │
│    • n_estimators: número de árboles                           │
│    • learning_rate: peso de cada árbol (0.01-0.3)              │
│    • max_depth: profundidad de árboles (3-10)                  │
│    • subsample/colsample: regularización por muestreo          │
│                                                                │
│  BEST PRACTICES:                                               │
│    • Usar early_stopping                                       │
│    • Cross-validation para tuning                              │
│    • Empezar con defaults, luego optimizar                     │
│    • Monitorear train vs validation para overfitting           │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Detección de malware (análisis estático)                  │
│    • Detección de intrusiones                                  │
│    • Clasificación de amenazas                                 │
│    • Scoring de riesgo                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Con esto completamos los principales modelos de clasificación:**
1. Regresión Logística
2. Árboles de Decisión
3. Random Forest
4. Support Vector Machines
5. Naive Bayes
6. Gradient Boosting / XGBoost
