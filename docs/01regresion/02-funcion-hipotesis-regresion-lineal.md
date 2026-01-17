# Función Hipótesis en Regresión Lineal

## 1. Esquema General de Aprendizaje Supervisado

Todas las técnicas de aprendizaje supervisado siguen un patrón común:

```
┌─────────────────────┐
│  CONJUNTO DE DATOS  │
│  (Experiencia)      │
│  (X, y) etiquetado  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│   FUNCIÓN HIPÓTESIS h(X)        │
│   (Función matemática con       │
│    parámetros θ₀, θ₁, ..., θₙ)  │
└──────────┬──────────────────────┘
           │
           │ Ajuste de parámetros
           │ con los datos
           ▼
┌─────────────────────────────────┐
│        MODELO                   │
│   h(X) con parámetros           │
│   óptimos ajustados             │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│      PREDICCIONES               │
│   h(X_nuevo) = ŷ                │
└─────────────────────────────────┘
```

**Componentes clave:**
1. **Conjunto de datos:** La experiencia pasada etiquetada
2. **Función hipótesis:** La función matemática con parámetros ajustables
3. **Modelo:** La función hipótesis después del entrenamiento
4. **Predicciones:** Usar el modelo para datos nuevos

## 2. La Función Hipótesis de Regresión Lineal

### Ecuación de la Recta (recordatorio)

La función hipótesis de la regresión lineal es la **ecuación de la recta**:

```
y = mx + b

Donde:
  - m = pendiente (inclinación de la recta)
  - b = intercepto (punto de corte con eje Y)
  - x = variable independiente
  - y = variable dependiente
```

### Notación en Machine Learning

En ML usamos notación con **parámetros theta (θ)**:

```
h_θ(x) = θ₀ + θ₁·x

Donde:
  - h_θ(x) = función hipótesis (predicción)
  - θ₀ = parámetro bias/intercepto (equivalente a 'b')
  - θ₁ = parámetro peso/pendiente (equivalente a 'm')
  - x = feature/característica de entrada
```

**Correspondencia:**
```
y = mx + b     (Notación matemática clásica)
  ‖   ‖   ‖
h_θ(x) = θ₁·x + θ₀   (Notación ML)
```

## 3. Regresión Lineal Univariable (Una Variable)

### Definición

**Regresión lineal univariable** o **regresión lineal simple**: cuando tenemos una **única característica de entrada**.

```
h_θ(x) = θ₀ + θ₁·x₁

Componentes:
  - x₁: única característica de entrada
  - θ₀: intercepto (valor cuando x₁ = 0)
  - θ₁: pendiente (cuánto cambia h por cada unidad de x₁)
  - Total: 2 parámetros a ajustar
```

### Visualización Gráfica

```
Coste (y)
    │
    │    h_θ₃(x)
    │      /
30k │     /              Diferentes rectas posibles
    │    /●              según valores de θ₀ y θ₁
    │   / 
25k │  /  ●   h_θ₁(x)  
    │ /      /
20k │/  ●  /              ● = datos de entrenamiento
    │   ● /               / = posibles funciones hipótesis
15k │ ●  /
    │   /  h_θ₂(x)
10k │● /      /
    │ /      /
 5k │/  ●   /
    │      /
    └───────────────────────────── Sistemas afectados (x)
    0    500   1000  1500  2000

h_θ₁(x): θ₀ = 2000, θ₁ = 12  →  Línea con pendiente suave
h_θ₂(x): θ₀ =    0, θ₁ = 15  →  Línea que pasa por origen
h_θ₃(x): θ₀ = 5000, θ₁ = 10  →  Línea con intercepto alto
```

### ¿Qué Determinan los Parámetros?

| Parámetro | Función | Efecto Visual |
|-----------|---------|---------------|
| **θ₀** | Intercepto | Dónde corta la línea el eje Y |
| **θ₁** | Pendiente | Inclinación de la recta |

```
Efecto de θ₀:
    │
    │    θ₀ = 10000
    │       /
    │      /
    │     /  θ₀ = 5000
    │    /  /
    │   /  /
    │  /  /   θ₀ = 0
    │ /  /   /
    │/  /   /
    └───────────────
    Mueve la línea arriba/abajo

Efecto de θ₁:
    │      /
    │     / θ₁ grande (pendiente alta)
    │    /
    │   /
    │  /  θ₁ pequeño (pendiente baja)
    │ /  /
    │/  /
    └───────────────
    Cambia la inclinación
```

### Ejemplo Práctico

**Problema:** Predecir coste de incidente según sistemas afectados

```python
# Datos de entrenamiento
X₁ (sistemas): [500, 1000, 1500, 2000]
y (coste):     [6000, 11000, 16000, 21000]

# Función hipótesis después del entrenamiento
h_θ(x) = 1000 + 10·x₁

# Parámetros aprendidos:
θ₀ = 1000  →  Coste base (cuando x₁ = 0)
θ₁ = 10    →  Cada sistema añade 10€ al coste

# Predicción para 1200 sistemas:
h_θ(1200) = 1000 + 10·1200 = 13,000€
```

## 4. Regresión Lineal Multivariable (Múltiples Variables)

### Definición

**Regresión lineal multivariable**: cuando tenemos **n características de entrada**.

```
h_θ(X) = θ₀ + θ₁·X₁ + θ₂·X₂ + ... + θₙ·Xₙ

Componentes:
  - X₁, X₂, ..., Xₙ: n características de entrada
  - θ₀: intercepto/bias
  - θ₁, θ₂, ..., θₙ: pesos/pendientes de cada característica
  - Total: (n + 1) parámetros a ajustar
```

**Forma compacta (notación sumatoria):**
```
         n
h_θ(X) = θ₀ + Σ θᵢ·Xᵢ
        i=1
```

### Ejemplo con 3 Features

```
Predecir coste de incidente con 3 características:

X₁ = Sistemas afectados
X₂ = Número de vulnerabilidades
X₃ = Tiempo de resolución (horas)

Función hipótesis:
h_θ(X) = θ₀ + θ₁·X₁ + θ₂·X₂ + θ₃·X₃

Ejemplo con valores aprendidos:
h_θ(X) = 500 + 8·X₁ + 1200·X₂ + 50·X₃

Parámetros:
  θ₀ = 500    →  Coste base
  θ₁ = 8      →  Cada sistema añade 8€
  θ₂ = 1200   →  Cada vulnerabilidad añade 1200€
  θ₃ = 50     →  Cada hora añade 50€
```

### Predicción con Multivariable

```python
# Nuevo incidente sin etiquetar:
X_nuevo = [1000 sistemas, 5 vulnerabilidades, 48 horas]

# Predicción:
h_θ(X_nuevo) = 500 + 8·1000 + 1200·5 + 50·48
             = 500 + 8000 + 6000 + 2400
             = 16,900€
```

### Tabla Comparativa

| Aspecto | Univariable | Multivariable |
|---------|-------------|---------------|
| **Features** | 1 (X₁) | n (X₁, X₂, ..., Xₙ) |
| **Función** | θ₀ + θ₁·X₁ | θ₀ + Σθᵢ·Xᵢ |
| **Parámetros** | 2 (θ₀, θ₁) | n+1 (θ₀, θ₁, ..., θₙ) |
| **Visualización** | Línea en 2D | Hiperplano en (n+1)D |
| **Ejemplo** | h(x) = 1000 + 10·x₁ | h(X) = 500 + 8·X₁ + 1200·X₂ |

## 5. Suma Ponderada de Entradas

### ¿Por qué "Suma Ponderada"?

La regresión lineal es una **suma ponderada** porque:

```
h_θ(X) = θ₀ + θ₁·X₁ + θ₂·X₂ + θ₃·X₃ + ... + θₙ·Xₙ
         ↑    ↑────────────────────────────────↑
       bias        suma ponderada de inputs
```

**Ponderada** significa que cada entrada Xᵢ se **multiplica por un peso θᵢ** antes de sumarla.

### Componentes

```
┌─────────────────────────────────────────┐
│  SUMA PONDERADA                         │
├─────────────────────────────────────────┤
│                                         │
│  θ₁·X₁  +  θ₂·X₂  +  ...  +  θₙ·Xₙ     │
│   ↑  ↑      ↑  ↑             ↑  ↑      │
│  peso input peso input      peso input │
│                                         │
├─────────────────────────────────────────┤
│  BIAS (término independiente)           │
├─────────────────────────────────────────┤
│                                         │
│  θ₀  →  Constante añadida               │
│                                         │
└─────────────────────────────────────────┘

Predicción final = BIAS + SUMA PONDERADA
```

### Interpretación de los Pesos

```
h_θ(X) = 500 + 8·X₁ + 1200·X₂ + 50·X₃

Análisis de importancia:
  θ₂ = 1200  →  Vulnerabilidades es el factor MÁS importante
  θ₃ = 50    →  Tiempo tiene impacto medio
  θ₁ = 8     →  Sistemas tiene menor impacto relativo
  θ₀ = 500   →  Coste base mínimo
```

**Magnitud del peso = Importancia de la característica**

## 6. El Proceso de Entrenamiento

### Objetivo del Entrenamiento

```
ANTES DEL ENTRENAMIENTO:
  h_θ(X) = θ₀ + θ₁·X₁ + θ₂·X₂ + θ₃·X₃
  
  θ₀ = ?   →  Valores desconocidos
  θ₁ = ?       o inicializados
  θ₂ = ?       aleatoriamente
  θ₃ = ?

DESPUÉS DEL ENTRENAMIENTO:
  h_θ(X) = 500 + 8·X₁ + 1200·X₂ + 50·X₃
  
  θ₀ = 500   →  Valores óptimos
  θ₁ = 8         ajustados a la
  θ₂ = 1200      tendencia de
  θ₃ = 50        los datos
```

### ¿Cómo se Ajustan los Parámetros?

El **entrenamiento** es el proceso de encontrar los valores óptimos de θ₀, θ₁, ..., θₙ que:

1. **Minimicen el error** entre predicciones y valores reales
2. **Ajusten la línea** lo mejor posible a la tendencia de los datos

```
Datos de entrenamiento:
    │
    │         ●  ←  y_real = 20000
    │        /
    │       / ●  ←  y_real = 15000
    │      /
    │  ●  /      
    │    /       Objetivo: encontrar θ₀ y θ₁
    │   /        que hagan que la línea
    │● /         pase lo más cerca posible
    │ /          de todos los puntos
    └─────────────────────

La línea óptima minimiza la distancia vertical
entre predicciones h_θ(x) y valores reales y
```

### Próximo Paso: Función de Coste

Para entrenar el modelo necesitamos:
1. **Función de coste (error):** Medir qué tan mal está ajustada la línea
2. **Algoritmo de optimización:** Ajustar θ₀, θ₁, ..., θₙ para minimizar el error

Esto lo veremos en la siguiente sección.

## 7. Resumen

```
┌───────────────────────────────────────────────────────────────┐
│ FUNCIÓN HIPÓTESIS - REGRESIÓN LINEAL                          │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  UNIVARIABLE (n=1):                                           │
│    h_θ(x) = θ₀ + θ₁·x₁                                        │
│    • 2 parámetros (θ₀, θ₁)                                    │
│    • Línea recta en 2D                                        │
│                                                               │
│  MULTIVARIABLE (n>1):                                         │
│    h_θ(X) = θ₀ + θ₁·X₁ + θ₂·X₂ + ... + θₙ·Xₙ                  │
│    • (n+1) parámetros                                         │
│    • Hiperplano en (n+1)D                                     │
│                                                               │
│  SUMA PONDERADA:                                              │
│    Predicción = BIAS + Σ(peso_i × feature_i)                  │
│                                                               │
│  OBJETIVO DEL ENTRENAMIENTO:                                  │
│    Encontrar θ₀, θ₁, ..., θₙ óptimos que ajusten              │
│    la función a la tendencia de los datos                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Conceptos clave:**
- θ₀ controla el **intercepto** (dónde corta eje Y)
- θ₁, θ₂, ..., θₙ controlan las **pendientes** (inclinación)
- Los **pesos** indican la **importancia** de cada feature
- El **entrenamiento** ajusta automáticamente todos los θᵢ

---

**Siguiente:** Función de coste y algoritmo de optimización (Gradient Descent)
