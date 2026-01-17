# Introducción a Machine Learning: Regresión Lineal

## 1. Clasificación de Técnicas de Machine Learning

```
                        MACHINE LEARNING
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        SUPERVISADO     NO SUPERVISADO   REFUERZO
              │               │               │
        ┌─────┴─────┐   ┌────┴────┐    (Q-Learning,
        │           │   │         │     Policy Gradient)
   REGRESIÓN  CLASIFICACIÓN  Clustering
        │           │       Reducción
   (Predecir   (Predecir    Dimensionalidad
    valores     categorías)
   continuos)
```

### Aprendizaje Supervisado
- **Input:** Datos etiquetados (X, y)
- **Objetivo:** Aprender función f: X → y
- **Ejemplos:**
  - Regresión: predecir precio de casa, temperatura, coste
  - Clasificación: spam/no-spam, gato/perro, fraude/legítimo

### Aprendizaje No Supervisado
- **Input:** Datos sin etiquetas (solo X)
- **Objetivo:** Encontrar patrones ocultos
- **Ejemplos:**
  - Clustering: segmentación de clientes
  - PCA: reducción de dimensionalidad

## 2. Regresión Lineal: Carta de Presentación

### Características del Algoritmo

| Propiedad | Valor |
|-----------|-------|
| **Tipo de aprendizaje** | Supervisado |
| **Paradigma** | Basado en modelos |
| **Tipo de predicción** | Regresión (valores continuos) |
| **Función hipótesis** | Lineal: h(x) = w₀ + w₁x₁ + ... + wₙxₙ |
| **Complejidad** | Baja (algoritmo sencillo) |

### ¿Cuándo Usar Regresión Lineal?

**✅ Usar cuando:**
- Los datos muestran tendencia lineal
- Relación entre variables es aproximadamente proporcional
- Necesitas interpretabilidad (coeficientes = importancia)

**❌ NO usar cuando:**
- Relación es claramente no lineal (exponencial, cuadrática, etc.)
- Datos con muchos outliers que distorsionan la línea
- Variables categóricas sin codificar

## 3. Ejemplo Práctico: Coste de Incidentes de Seguridad

### Problema
Predecir el **coste** de un incidente de seguridad basado en el **número de sistemas afectados**.

### Visualización de los Datos

```
Coste (€)
   │
25k│                             ●
   │
20k│                    ●
   │              ●
15k│         ●              
   │    ●                      Línea de regresión
10k│  ●                        ajustada: y = mx + b
   │ ●                                /
 5k│●                               /
   │                              /
   │_____________________________/____________
   0   500  1000  1500  2000        Sistemas afectados
```

### La Función Hipótesis

El algoritmo construye automáticamente una **línea recta** que mejor se ajusta a los datos:

```
h(x) = w₀ + w₁·x

Donde:
  - x  = número de sistemas afectados (feature)
  - w₀ = bias/intercepto (valor cuando x=0)
  - w₁ = peso/pendiente (cuánto aumenta y por cada unidad de x)
  - h(x) = predicción del coste
```

## 4. Notación Matemática

### Conjunto de Datos (Dataset)

```
┌─────────────┬──────────────────────┬────────────┐
│  Incidente  │  X₁ (Sistemas)      │  y (Coste) │
├─────────────┼──────────────────────┼────────────┤
│      1      │       1000           │   10,000€  │  ← Ejemplo de entrenamiento
│      2      │       1500           │   20,000€  │
│      3      │        500           │    5,500€  │
│     ...     │        ...           │     ...    │
│      m      │        200           │    2,500€  │
└─────────────┴──────────────────────┴────────────┘
       ↑              ↑                    ↑
   Índice de      Feature            Target/Label
   ejemplo       (entrada)           (salida)
```

### Símbolos y Definiciones

| Símbolo | Significado | Ejemplo |
|---------|-------------|---------|
| **m** | Número de ejemplos de entrenamiento | m = 1000 incidentes |
| **n** | Número de features/características | n = 1 (solo sistemas afectados) |
| **X** | Variables de entrada (features) | X₁ = sistemas afectados |
| **y** | Variable de salida (target) | y = coste del incidente |
| **(X, y)** | Un ejemplo de entrenamiento | (1000, 10000€) |
| **(Xⁱ, yⁱ)** | El i-ésimo ejemplo | (X³, y³) = (500, 5500€) |
| **h(X)** | Función hipótesis (predicción) | h(1200) = 15000€ |

### Caso con Múltiples Features

Si extraemos más características:

```
X₁ = Sistemas afectados
X₂ = Número de vulnerabilidades
X₃ = Tiempo de resolución (horas)
X₄ = Nivel de criticidad (1-10)

Ejemplo de entrenamiento:
(X, y) = ([1000, 5, 48, 8], 10000€)
          ↑────────────↑    ↑
         Vector X       Target y
```

La función hipótesis se expande:

```
h(X) = w₀ + w₁·X₁ + w₂·X₂ + w₃·X₃ + w₄·X₄

O en forma vectorial:
h(X) = w₀ + Σ(wᵢ·Xᵢ)   para i=1 hasta n
```

## 5. Implementación en Código

### Representación con Pandas

En código Python usaremos **DataFrames de Pandas**:

```python
import pandas as pd

# Dataset se representa como DataFrame
df = pd.DataFrame({
    'sistemas_afectados': [1000, 1500, 500, 800, 1200],  # Features (X)
    'vulnerabilidades': [5, 8, 2, 3, 6],
    'tiempo_horas': [48, 72, 24, 36, 60],
    'coste': [10000, 20000, 5500, 8000, 15000]          # Target (y)
})

# Convención común:
# - df: DataFrame completo
# - X: Features (columnas de entrada)
# - y: Target (columna de salida)

X = df[['sistemas_afectados', 'vulnerabilidades', 'tiempo_horas']]
y = df['coste']
```

### Flujo del Algoritmo

```
┌──────────────────┐
│  Dataset (X, y)  │
│  Etiquetado      │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│  Algoritmo de           │
│  Regresión Lineal       │
│  (entrenamiento)        │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Modelo h(X) = w₀+w₁X₁+..│  ← Función hipótesis
│  (línea ajustada)        │     ajustada
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Nuevos datos (X_nuevo)  │
│  Sin etiqueta            │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Predicción: h(X_nuevo)  │
│  = coste estimado        │
└──────────────────────────┘
```

## 6. Conceptos Clave para Entender

### Suma Ponderada

La regresión lineal calcula una **suma ponderada** de las features:

```
Predicción = (peso₁ × feature₁) + (peso₂ × feature₂) + ... + bias

Ejemplo:
h(X) = 5 + (10 × sistemas) + (500 × vulnerabilidades)
                ↑                    ↑
             pesos/weights       bias/intercepto
```

**Los pesos indican la importancia de cada feature.**

### Aprendizaje Basado en Modelos

El algoritmo **construye un modelo** (la función h) que luego se usa para predicciones:

- **Fase de entrenamiento:** Ajustar pesos w₀, w₁, ..., wₙ
- **Fase de predicción:** Usar h(X) con los pesos aprendidos

### Valores Continuos

La salida es un **número real en un rango continuo**, no una categoría discreta:

```
Regresión:      →  2500€, 10000€, 15473.23€  (infinitos valores posibles)
Clasificación:  →  Spam/No-spam, A/B/C       (valores discretos)
```

## 7. Próximos Pasos

En las siguientes secciones veremos:

1. **Construcción del modelo:** Cómo calcular los pesos óptimos
2. **Función de coste:** Cómo medir qué tan buena es nuestra línea
3. **Optimización:** Algoritmos para encontrar los mejores pesos (Gradient Descent)
4. **Implementación práctica:** Código completo con scikit-learn y desde cero
5. **Evaluación:** Métricas para medir el rendimiento (MSE, R², RMSE)

---

## Resumen Rápido

```
┌───────────────────────────────────────────────────────────┐
│ REGRESIÓN LINEAL                                          │
├───────────────────────────────────────────────────────────┤
│ • Supervisado (necesita datos etiquetados)                │
│ • Basado en modelos (construye función h)                 │
│ • Predice valores continuos (regresión)                   │
│ • Función lineal: h(X) = w₀ + w₁X₁ + ... + wₙXₙ          │
│ • Ajusta una línea a datos con tendencia lineal           │
│ • Interpretable: pesos = importancia de features          │
└───────────────────────────────────────────────────────────┘
```

**Ecuación fundamental:**
```
h(X) = w₀ + Σ(wᵢ·Xᵢ)    donde i = 1 hasta n
```

**Objetivo del algoritmo:**
Encontrar los valores de w₀, w₁, ..., wₙ que mejor ajusten la línea a los datos de entrenamiento.
