# Árboles de Decisión

## 1. Concepto Intuitivo

### ¿Qué es un Árbol de Decisión?

```
Un árbol de decisión es una serie de PREGUNTAS que
llevan a una DECISIÓN final.

Ejemplo cotidiano: "¿Es este email SPAM?"

                    ┌─────────────────────┐
                    │ ¿Contiene "FREE"?   │
                    └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
             SÍ                    NO
              │                     │
              ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐
    │ ¿Tiene enlaces? │   │ ¿Remitente      │
    └────────┬────────┘   │  conocido?      │
             │            └────────┬────────┘
      ┌──────┴──────┐           ┌──┴──┐
     SÍ            NO          SÍ    NO
      │             │           │     │
      ▼             ▼           ▼     ▼
   [SPAM]        [SPAM]      [HAM]  [REVISAR]
   (95%)         (70%)       (99%)  (50%)
```

### Componentes del Árbol

```
┌────────────────────────────────────────────────────────┐
│  ANATOMÍA DE UN ÁRBOL DE DECISIÓN                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  NODO RAÍZ:     Primera pregunta (arriba del todo)    │
│  NODOS INTERNOS: Preguntas intermedias                │
│  RAMAS:         Respuestas posibles (sí/no, </>)     │
│  HOJAS:         Decisiones finales (clases)           │
│                                                        │
│  PROFUNDIDAD:   Número de niveles del árbol           │
│                                                        │
│         [Raíz]  ← Profundidad 0                       │
│          / \                                           │
│       [A]  [B]  ← Profundidad 1                       │
│       /\    /\                                         │
│     [C][D][E][F] ← Profundidad 2 (hojas)              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 2. ¿Cómo Decide el Árbol qué Preguntar?

### El Problema

```
Tenemos datos de emails:

┌───────────┬──────────┬────────────┬─────────┐
│ Tiene URL │ Longitud │ Mayúsculas │  Clase  │
├───────────┼──────────┼────────────┼─────────┤
│    Sí     │  Corto   │    Alta    │  SPAM   │
│    No     │  Largo   │    Baja    │  HAM    │
│    Sí     │  Corto   │    Alta    │  SPAM   │
│    No     │  Corto   │    Baja    │  HAM    │
│    Sí     │  Largo   │    Baja    │  HAM    │
│    Sí     │  Corto   │    Alta    │  SPAM   │
└───────────┴──────────┴────────────┴─────────┘

¿Por qué feature empezamos?
  → Elegimos la que MEJOR separe las clases
```

### Criterios de División

```
┌────────────────────────────────────────────────────────┐
│  CRITERIOS PARA ELEGIR LA MEJOR PREGUNTA               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  GINI IMPURITY (Impureza de Gini):                    │
│    Mide qué tan "mezcladas" están las clases          │
│    Gini = 1 - Σ(pᵢ²)                                  │
│    Gini = 0 → Perfectamente puro (una sola clase)     │
│    Gini = 0.5 → Máxima impureza (50/50)              │
│                                                        │
│  ENTROPY (Entropía):                                   │
│    Mide el "desorden" o incertidumbre                 │
│    Entropy = -Σ(pᵢ · log₂(pᵢ))                       │
│    Entropy = 0 → Perfectamente puro                   │
│    Entropy = 1 → Máxima incertidumbre (50/50)        │
│                                                        │
│  INFORMATION GAIN (Ganancia de Información):          │
│    Reducción de entropía al hacer la división         │
│    IG = Entropy(padre) - Σ(peso × Entropy(hijo))     │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Ejemplo: Cálculo de Gini

```
Datos originales: 6 emails (3 SPAM, 3 HAM)
Gini original = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5

Opción A: Dividir por "Tiene URL"
  URL=Sí: 4 emails (3 SPAM, 1 HAM) → Gini = 1-(0.75²+0.25²) = 0.375
  URL=No: 2 emails (0 SPAM, 2 HAM) → Gini = 1-(0²+1²) = 0.0 (puro!)

  Gini ponderado = (4/6)×0.375 + (2/6)×0.0 = 0.25

Opción B: Dividir por "Longitud"
  Corto: 4 emails (3 SPAM, 1 HAM) → Gini = 0.375
  Largo: 2 emails (0 SPAM, 2 HAM) → Gini = 0.0

  Gini ponderado = (4/6)×0.375 + (2/6)×0.0 = 0.25

Ambas igual de buenas. Elegimos una y continuamos.
```

## 3. Algoritmo de Construcción

### Proceso Recursivo

```
┌────────────────────────────────────────────────────────┐
│  ALGORITMO DE CONSTRUCCIÓN (CART)                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  function construir_arbol(datos):                      │
│                                                        │
│    1. SI todos los datos son de una clase:            │
│       → Crear HOJA con esa clase                      │
│       → RETURN                                         │
│                                                        │
│    2. SI no quedan features o max_depth alcanzado:    │
│       → Crear HOJA con clase mayoritaria              │
│       → RETURN                                         │
│                                                        │
│    3. PARA cada feature:                              │
│       → Calcular Gini/Entropía de cada división       │
│       → Guardar la mejor                              │
│                                                        │
│    4. DIVIDIR datos según mejor feature               │
│                                                        │
│    5. RECURSIVAMENTE:                                  │
│       → construir_arbol(datos_izquierda)              │
│       → construir_arbol(datos_derecha)                │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Visualización del Proceso

```
Paso 1: Datos iniciales
┌─────────────────────┐
│ 3 SPAM, 3 HAM       │  Gini = 0.5
│ (mezclados)         │
└──────────┬──────────┘
           │
           ▼ Mejor split: "Tiene URL"

Paso 2: Primera división
           ┌─────────────────┐
           │  ¿Tiene URL?    │
           └────────┬────────┘
                    │
         ┌──────────┴──────────┐
        SÍ                    NO
         │                     │
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│ 3 SPAM, 1 HAM   │   │ 0 SPAM, 2 HAM   │
│ Gini = 0.375    │   │ Gini = 0 (puro!)│
└────────┬────────┘   └─────────────────┘
         │                     │
         ▼                     ▼
   Seguir dividiendo        [HAM] ✓

Paso 3: Segunda división (rama izquierda)
┌─────────────────┐
│ ¿Mayúsculas?    │
└────────┬────────┘
         │
    ┌────┴────┐
  Alta       Baja
    │         │
    ▼         ▼
 [SPAM]     [HAM]
  (3/3)     (1/1)
```

## 4. Predicción con el Árbol

### Cómo Clasificar un Nuevo Email

```
Nuevo email: {URL: Sí, Longitud: Corto, Mayúsculas: Alta}

                ┌─────────────────┐
                │  ¿Tiene URL?    │
                └────────┬────────┘
                         │
              SÍ ←───────┘
              │
              ▼
        ┌─────────────────┐
        │  ¿Mayúsculas?   │
        └────────┬────────┘
                 │
      Alta ←─────┘
      │
      ▼
   [SPAM] ← Predicción final

El email sigue el camino: URL=Sí → Mayúsculas=Alta → SPAM
```

## 5. Implementación en Python

### Código Básico

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Datos de ejemplo
df = pd.DataFrame({
    'tiene_url': [1, 0, 1, 0, 1, 1, 0, 1],
    'longitud': [100, 500, 80, 400, 600, 90, 300, 120],
    'mayusculas_pct': [30, 5, 25, 8, 10, 35, 3, 28],
    'clase': [1, 0, 1, 0, 0, 1, 0, 1]  # 1=SPAM, 0=HAM
})

X = df[['tiene_url', 'longitud', 'mayusculas_pct']]
y = df['clase']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Crear y entrenar árbol
arbol = DecisionTreeClassifier(
    criterion='gini',      # o 'entropy'
    max_depth=3,           # Limitar profundidad
    min_samples_split=2,   # Mínimo para dividir
    min_samples_leaf=1,    # Mínimo en hojas
    random_state=42
)

arbol.fit(X_train, y_train)

# Predecir
y_pred = arbol.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Visualizar el Árbol

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(
    arbol,
    feature_names=['tiene_url', 'longitud', 'mayusculas_pct'],
    class_names=['HAM', 'SPAM'],
    filled=True,           # Colorear nodos
    rounded=True,          # Bordes redondeados
    fontsize=12
)
plt.title('Árbol de Decisión - Detector de SPAM')
plt.tight_layout()
plt.savefig('arbol_spam.png', dpi=150)
plt.show()
```

### Ver Importancia de Features

```python
# Importancia de cada feature
importancias = pd.DataFrame({
    'feature': X.columns,
    'importancia': arbol.feature_importances_
}).sort_values('importancia', ascending=False)

print("Importancia de Features:")
print(importancias)

# Visualizar
importancias.plot(kind='barh', x='feature', y='importancia')
plt.title('Importancia de Features')
plt.xlabel('Importancia')
plt.show()
```

## 6. Hiperparámetros Importantes

### Control de Complejidad

```
┌─────────────────────┬────────────────────────────────────┐
│   HIPERPARÁMETRO    │           EFECTO                   │
├─────────────────────┼────────────────────────────────────┤
│                     │                                    │
│   max_depth         │  Profundidad máxima del árbol     │
│                     │  Bajo → Underfitting              │
│                     │  Alto → Overfitting               │
│                     │                                    │
├─────────────────────┼────────────────────────────────────┤
│                     │                                    │
│   min_samples_split │  Mínimo de muestras para dividir  │
│                     │  un nodo interno                   │
│                     │  Alto → Árbol más simple          │
│                     │                                    │
├─────────────────────┼────────────────────────────────────┤
│                     │                                    │
│   min_samples_leaf  │  Mínimo de muestras en una hoja   │
│                     │  Alto → Hojas más generales       │
│                     │                                    │
├─────────────────────┼────────────────────────────────────┤
│                     │                                    │
│   max_features      │  Número de features a considerar  │
│                     │  en cada división                  │
│                     │                                    │
├─────────────────────┼────────────────────────────────────┤
│                     │                                    │
│   criterion         │  'gini' o 'entropy'               │
│                     │  Generalmente similar rendimiento │
│                     │                                    │
└─────────────────────┴────────────────────────────────────┘
```

### Visualización del Efecto de max_depth

```
max_depth = 1 (muy simple):
       ┌───────┐
       │ URL?  │
       └───┬───┘
         ┌─┴─┐
      [SPAM] [HAM]

→ Underfitting: No captura complejidad


max_depth = 10 (muy profundo):
       ┌───────┐
       │ URL?  │
       └───┬───┘
      ┌────┴────┐
    ┌─┴─┐     ┌─┴─┐
   ...  ...  ...  ...
  (muchos niveles más)

→ Overfitting: Memoriza datos de training


max_depth = 3-5 (balanceado):
       ┌───────┐
       │ URL?  │
       └───┬───┘
      ┌────┴────┐
    ┌─┴─┐     ┌─┴─┐
   [S] ┌┴┐   [H] ┌┴┐
      [S][H]    [H][S]

→ Generaliza bien
```

## 7. Overfitting en Árboles

### El Problema

```
Sin restricciones, el árbol puede crecer hasta que
cada hoja tenga UN SOLO ejemplo → Memorización total

Training Accuracy: 100% (perfecto)
Test Accuracy: 60% (terrible)

El árbol ha MEMORIZADO los datos de training,
incluyendo el ruido.
```

### Soluciones: Poda (Pruning)

```
┌────────────────────────────────────────────────────────┐
│  TÉCNICAS DE PODA                                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  PRE-PRUNING (Poda temprana):                         │
│    • Limitar max_depth                                │
│    • Establecer min_samples_split alto                │
│    • Establecer min_samples_leaf alto                 │
│    → Detener el crecimiento ANTES de que ocurra      │
│                                                        │
│  POST-PRUNING (Poda posterior):                       │
│    • Construir árbol completo                         │
│    • Eliminar ramas que no mejoran validación         │
│    • ccp_alpha en sklearn (Cost Complexity Pruning)   │
│    → Podar DESPUÉS de construir                       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Ejemplo de Cost Complexity Pruning

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Encontrar el mejor alpha para poda
path = arbol.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Probar diferentes alphas
scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    scores.append(score)

# Elegir el mejor alpha
mejor_alpha = ccp_alphas[np.argmax(scores)]
print(f"Mejor alpha: {mejor_alpha:.4f}")

# Entrenar árbol podado
arbol_podado = DecisionTreeClassifier(
    ccp_alpha=mejor_alpha,
    random_state=42
)
arbol_podado.fit(X_train, y_train)
```

## 8. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────┐
│  VENTAJAS                                              │
├────────────────────────────────────────────────────────┤
│  ✓ Fácil de interpretar y visualizar                  │
│  ✓ No requiere normalización de datos                 │
│  ✓ Maneja features numéricas y categóricas            │
│  ✓ Captura relaciones no lineales                     │
│  ✓ Selección automática de features                   │
│  ✓ Rápido de entrenar y predecir                      │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  DESVENTAJAS                                           │
├────────────────────────────────────────────────────────┤
│  ✗ Tendencia al overfitting                           │
│  ✗ Inestable: pequeños cambios → árbol diferente     │
│  ✗ Sesgo hacia features con muchos valores            │
│  ✗ No captura bien relaciones lineales               │
│  ✗ Puede crear fronteras de decisión "escalonadas"   │
└────────────────────────────────────────────────────────┘
```

## 9. Aplicación en Ciberseguridad

### Ejemplo: Detección de Intrusiones

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Simular datos de red
np.random.seed(42)
n = 1000

data = {
    'bytes_sent': np.random.exponential(1000, n),
    'bytes_recv': np.random.exponential(2000, n),
    'packets': np.random.poisson(100, n),
    'duration': np.random.exponential(10, n),
    'failed_logins': np.random.poisson(0.5, n),
    'port_scan_count': np.random.poisson(0.2, n),
}

# Generar labels (ataque si cumple ciertas condiciones)
df = pd.DataFrame(data)
df['ataque'] = (
    (df['failed_logins'] > 2) |
    (df['port_scan_count'] > 1) |
    ((df['bytes_sent'] > 3000) & (df['duration'] < 2))
).astype(int)

# Preparar datos
X = df.drop('ataque', axis=1)
y = df['ataque']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar árbol
arbol = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)
arbol.fit(X_train, y_train)

# Evaluar
y_pred = arbol.predict(X_test)
print(classification_report(y_test, y_pred,
                           target_names=['Normal', 'Ataque']))

# Ver reglas aprendidas
print("\nFeatures más importantes:")
for feat, imp in sorted(zip(X.columns, arbol.feature_importances_),
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.3f}")
```

## 10. Resumen

```
┌───────────────────────────────────────────────────────────────┐
│  ÁRBOLES DE DECISIÓN - RESUMEN                                │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  CONCEPTO:                                                    │
│    Serie de preguntas if/else que llevan a una decisión      │
│                                                               │
│  CONSTRUCCIÓN:                                                │
│    • Elegir feature que mejor separa clases (Gini/Entropy)   │
│    • Dividir datos recursivamente                            │
│    • Parar cuando se cumple criterio                         │
│                                                               │
│  HIPERPARÁMETROS CLAVE:                                       │
│    • max_depth: controla profundidad                         │
│    • min_samples_split: mínimo para dividir                  │
│    • min_samples_leaf: mínimo en hojas                       │
│                                                               │
│  OVERFITTING:                                                 │
│    • Problema común en árboles                               │
│    • Solución: Poda (pre o post)                             │
│                                                               │
│  VENTAJA PRINCIPAL:                                           │
│    Interpretabilidad - puedes "leer" las reglas              │
│                                                               │
│  DESVENTAJA PRINCIPAL:                                        │
│    Inestable y propenso a overfitting                        │
│    → Solución: Random Forest (próximo tema)                  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Random Forest - Ensemble de árboles para mayor estabilidad
