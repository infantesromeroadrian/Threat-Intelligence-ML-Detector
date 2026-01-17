# Vectorización de Texto: TF-IDF

## 1. El Problema: Texto → Números

### ¿Por qué Necesitamos Vectorizar?

Los algoritmos de ML trabajan con **números**, no con texto.

```
┌─────────────────────────────────────────────────────────┐
│  PROBLEMA                                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Email: "FREE iPhone! Click NOW to WIN!"                │
│                                                         │
│  Regresión Logística necesita:                          │
│  h_θ(x) = sigmoid(θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ)      │
│                                                         │
│  ¿Cómo convertimos texto en x₁, x₂, ..., xₙ?           │
│                                                         │
└─────────────────────────────────────────────────────────┘

Solución: VECTORIZACIÓN
  Texto → Vector de números de dimensión fija
```

### Pipeline de Texto a Vector

```
┌───────────────────────┐
│  "FREE iPhone! WIN!"  │  ← Texto original
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  1. PREPROCESAMIENTO  │
│  • Lowercase          │
│  • Eliminar puntuación│
│  "free iphone win"    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  2. TOKENIZACIÓN      │
│  ["free", "iphone",   │
│   "win"]              │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  3. VECTORIZACIÓN     │
│  [0.0, 0.52, 0.0,     │
│   0.61, 0.0, 0.59]    │  ← Vector numérico
└───────────────────────┘
```

## 2. Bag of Words (BoW)

### Concepto Básico

**Bag of Words:** Representar texto como frecuencia de palabras.

```
Vocabulario (todas las palabras únicas del corpus):
["buy", "cheap", "click", "free", "hello", "home", "iphone", "mom", "win"]

Documento 1: "free iphone win"
Vector BoW:  [0, 0, 0, 1, 0, 0, 1, 0, 1]
              ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
           buy cheap click free hello home iphone mom win

Documento 2: "hello mom"
Vector BoW:  [0, 0, 0, 0, 1, 0, 0, 1, 0]
```

### Limitaciones de BoW

```
┌────────────────────────────────────────────────────────┐
│  PROBLEMAS DE BAG OF WORDS                             │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1. IGNORA FRECUENCIA RELATIVA:                        │
│     "the" aparece 1000 veces → peso = 1000             │
│     "malware" aparece 5 veces → peso = 5              │
│     Pero "malware" es más informativo!                 │
│                                                        │
│  2. NO CONSIDERA IMPORTANCIA:                          │
│     Palabras comunes ("the", "is", "a") tienen         │
│     el mismo peso que palabras importantes             │
│                                                        │
│  3. VECTORES MUY DISPERSOS (sparse):                   │
│     Vocabulario de 10,000 palabras                     │
│     → Vector de 10,000 dimensiones                     │
│     → La mayoría son 0s                                │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 3. TF-IDF: La Solución

### ¿Qué es TF-IDF?

**TF-IDF = Term Frequency × Inverse Document Frequency**

```
┌────────────────────────────────────────────────────────┐
│  TF-IDF                                                │
├────────────────────────────────────────────────────────┤
│                                                        │
│  TF-IDF(t, d) = TF(t, d) × IDF(t)                     │
│                                                        │
│  Donde:                                                │
│    t = término (palabra)                               │
│    d = documento                                       │
│    TF = frecuencia del término en el documento         │
│    IDF = rareza del término en el corpus               │
│                                                        │
└────────────────────────────────────────────────────────┘

Intuición:
  • TF alto: La palabra aparece mucho en ESTE documento
  • IDF alto: La palabra es RARA en el corpus general
  • TF-IDF alto: Palabra frecuente aquí pero rara globalmente
                 → Probablemente IMPORTANTE para este documento
```

### Fórmula TF (Term Frequency)

```
                    Veces que 't' aparece en documento 'd'
TF(t, d) = ───────────────────────────────────────────────────
            Total de términos en documento 'd'


Ejemplo:
  Documento: "click here click now click"

  TF("click") = 3/5 = 0.6
  TF("here")  = 1/5 = 0.2
  TF("now")   = 1/5 = 0.2
```

### Fórmula IDF (Inverse Document Frequency)

```
                    Total de documentos (N)
IDF(t) = log ────────────────────────────────────
              Documentos que contienen 't' + 1

(El +1 evita división por cero)


Ejemplo con N = 1000 documentos:

┌─────────────┬──────────────────────┬─────────┐
│   Palabra   │ Docs que la contienen│  IDF    │
├─────────────┼──────────────────────┼─────────┤
│    "the"    │         950          │  0.05   │  ← Muy común, IDF bajo
│   "click"   │         100          │  2.30   │  ← Moderado
│  "malware"  │          10          │  4.61   │  ← Raro, IDF alto
│   "trojan"  │           5          │  5.30   │  ← Muy raro, IDF muy alto
└─────────────┴──────────────────────┴─────────┘

IDF penaliza palabras comunes y premia palabras raras
```

### Cálculo Completo TF-IDF

```
Ejemplo:
  Corpus: 1000 documentos
  Documento actual: "click here to claim your free prize click"
  Palabra objetivo: "click"

  TF("click") = 2/8 = 0.25

  "click" aparece en 100 de 1000 documentos
  IDF("click") = log(1000/101) = log(9.9) ≈ 2.29

  TF-IDF("click") = 0.25 × 2.29 = 0.57


Comparación:
┌───────────┬────────┬────────┬──────────┐
│  Palabra  │   TF   │  IDF   │  TF-IDF  │
├───────────┼────────┼────────┼──────────┤
│  "the"    │  0.10  │  0.05  │   0.005  │  ← Peso bajo (común)
│  "click"  │  0.25  │  2.29  │   0.57   │  ← Peso alto
│  "prize"  │  0.125 │  3.91  │   0.49   │  ← Peso alto (raro)
└───────────┴────────┴────────┴──────────┘
```

## 4. Visualización del Proceso

### De Texto a Vector TF-IDF

```
CORPUS (3 documentos):
  D1: "free money click here"
  D2: "hello mom coming home"
  D3: "free prize click now win"

VOCABULARIO:
  ["click", "coming", "free", "hello", "here", "home",
   "mom", "money", "now", "prize", "win"]

MATRIZ TF-IDF:
                click  coming  free  hello  here  home  mom  money  now  prize  win
         ┌─────────────────────────────────────────────────────────────────────────┐
    D1   │  0.41    0.0   0.41   0.0   0.58   0.0  0.0   0.58  0.0   0.0   0.0   │
    D2   │  0.0    0.50   0.0   0.50   0.0   0.50 0.50   0.0   0.0   0.0   0.0   │
    D3   │  0.35    0.0   0.35   0.0   0.0    0.0  0.0   0.0  0.44  0.44  0.44  │
         └─────────────────────────────────────────────────────────────────────────┘

Observaciones:
  • "free" y "click" aparecen en D1 y D3 → IDF moderado
  • "money" solo en D1 → IDF alto → peso alto en D1
  • "mom", "home" solo en D2 → IDF alto → representan bien D2
```

### Representación en Espacio Vectorial

```
        click
          │
          │    ● D3 ("free prize click now win")
          │   /
          │  /
          │ /
          │/
──────────┼─────────────────── free
         /│
        / │
       /  │
      ●   │
     D1   │
          │
          │           ● D2 ("hello mom coming home")
          │             (perpendicular, diferente tema)
          │
          ▼
         mom

Los documentos similares están CERCA en el espacio vectorial
Los documentos diferentes están LEJOS
```

## 5. Parámetros Importantes de TF-IDF

### max_features

```
max_features = N

Limita el vocabulario a las N palabras más frecuentes.

┌───────────────────────────────────────────────────────┐
│  EFECTO DE max_features                               │
├───────────────────────────────────────────────────────┤
│                                                       │
│  max_features = 100                                   │
│    ✓ Vectores pequeños (100 dimensiones)             │
│    ✓ Rápido de procesar                              │
│    ✗ Puede perder palabras importantes               │
│                                                       │
│  max_features = 10000                                 │
│    ✓ Captura más vocabulario                         │
│    ✗ Vectores grandes                                │
│    ✗ Más lento, más memoria                          │
│                                                       │
│  Recomendación: 1000-5000 para textos cortos         │
│                 5000-20000 para textos largos        │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### ngram_range

```
ngram_range = (min_n, max_n)

Incluye secuencias de N palabras consecutivas.

┌───────────────────────────────────────────────────────┐
│  EJEMPLOS DE N-GRAMAS                                 │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Texto: "click here now"                              │
│                                                       │
│  ngram_range = (1, 1)  →  Unigramas                  │
│    ["click", "here", "now"]                          │
│                                                       │
│  ngram_range = (1, 2)  →  Uni + Bigramas             │
│    ["click", "here", "now",                          │
│     "click here", "here now"]                        │
│                                                       │
│  ngram_range = (1, 3)  →  Uni + Bi + Trigramas       │
│    ["click", "here", "now",                          │
│     "click here", "here now",                        │
│     "click here now"]                                │
│                                                       │
└───────────────────────────────────────────────────────┘

Ventaja de bigramas:
  "free" + "shipping" como palabras separadas ≠ "free shipping"
  El bigrama "free shipping" captura el significado conjunto
```

### stop_words

```
stop_words = 'english'

Elimina palabras muy comunes sin valor semántico.

┌───────────────────────────────────────────────────────┐
│  STOP WORDS (ejemplos en inglés)                      │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Artículos: the, a, an                               │
│  Preposiciones: in, on, at, to, for, with            │
│  Conjunciones: and, or, but                          │
│  Pronombres: I, you, he, she, it, we, they           │
│  Verbos auxiliares: is, are, was, were, be, been     │
│  Otros: this, that, these, those                     │
│                                                       │
└───────────────────────────────────────────────────────┘

Texto original: "Click on the link to win a free prize"
Sin stop words: "Click link win free prize"
```

## 6. Implementación en Python

### Código Básico

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Crear vectorizador
vectorizer = TfidfVectorizer(
    max_features=3000,      # Top 3000 palabras
    stop_words='english',   # Eliminar stop words
    lowercase=True,         # Todo a minúsculas
    ngram_range=(1, 2)      # Unigramas y bigramas
)

# Corpus de ejemplo
corpus = [
    "free money click here",
    "hello mom coming home",
    "free prize click now win"
]

# Ajustar y transformar
X = vectorizer.fit_transform(corpus)

print(f"Shape: {X.shape}")  # (3, N) donde N ≤ 3000
print(f"Vocabulario: {len(vectorizer.vocabulary_)} términos")
```

### Regla de Oro: Fit en Train, Transform en Test

```python
# ⚠️ CRÍTICO: Evitar data leakage

# INCORRECTO ❌
vectorizer.fit_transform(X_train + X_test)  # Leakage!

# CORRECTO ✓
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit + Transform
X_test_tfidf = vectorizer.transform(X_test)        # Solo Transform

┌────────────────────────────────────────────────────────┐
│  ¿POR QUÉ?                                             │
├────────────────────────────────────────────────────────┤
│                                                        │
│  fit() aprende:                                        │
│    • El vocabulario (qué palabras existen)            │
│    • Los valores IDF (rareza de cada palabra)         │
│                                                        │
│  Si hacemos fit() en test:                             │
│    • El modelo "ve" datos del futuro                  │
│    • Los IDF incluyen información del test            │
│    • Métricas infladas artificialmente                │
│                                                        │
│  El test debe ser COMPLETAMENTE desconocido           │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 7. Ejemplo Completo: Pipeline SPAM

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Cargar datos
df = pd.read_csv('email.csv')
X = df['Message']
y = df['Category'].map({'ham': 0, 'spam': 1})

# 2. Split PRIMERO
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Vectorizar
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit aquí
X_test_tfidf = vectorizer.transform(X_test)        # Solo transform

# 4. Entrenar modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Evaluar
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # ~97%
```

## 8. Interpretación de Coeficientes

### Palabras más Importantes

```python
# Obtener nombres de features y coeficientes
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]

# Top palabras que indican SPAM (θ positivo)
spam_indices = coef.argsort()[-10:][::-1]
print("TOP 10 PALABRAS → SPAM:")
for idx in spam_indices:
    print(f"  {feature_names[idx]:20s} θ = {coef[idx]:+.3f}")

# Top palabras que indican HAM (θ negativo)
ham_indices = coef.argsort()[:10]
print("\nTOP 10 PALABRAS → HAM:")
for idx in ham_indices:
    print(f"  {feature_names[idx]:20s} θ = {coef[idx]:+.3f}")
```

### Output Típico

```
TOP 10 PALABRAS → SPAM:
  txt                  θ = +4.381
  claim                θ = +3.388
  free                 θ = +2.816
  prize                θ = +2.546
  win                  θ = +2.220
  urgent               θ = +2.121
  mobile               θ = +3.493
  www                  θ = +3.270
  reply                θ = +3.174
  150p                 θ = +2.903

TOP 10 PALABRAS → HAM:
  ok                   θ = -1.789
  home                 θ = -1.348
  sorry                θ = -1.226
  later                θ = -1.155
  good                 θ = -1.138
  going                θ = -1.123
  hey                  θ = -1.194
  got                  θ = -1.285
  come                 θ = -1.367
  ll                   θ = -1.609
```

## 9. Resumen

```
┌───────────────────────────────────────────────────────────────┐
│  TF-IDF - RESUMEN                                             │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  OBJETIVO: Convertir texto en vectores numéricos              │
│                                                               │
│  FÓRMULA:                                                     │
│    TF-IDF(t, d) = TF(t, d) × IDF(t)                          │
│                                                               │
│  COMPONENTES:                                                 │
│    • TF: Frecuencia del término en el documento              │
│    • IDF: Rareza del término en el corpus                    │
│                                                               │
│  PARÁMETROS CLAVE:                                            │
│    • max_features: Limitar vocabulario                        │
│    • ngram_range: Incluir secuencias de palabras              │
│    • stop_words: Eliminar palabras comunes                    │
│                                                               │
│  REGLA DE ORO:                                                │
│    fit_transform(train) → transform(test)                     │
│    NUNCA fit en test (data leakage)                          │
│                                                               │
│  VENTAJAS:                                                    │
│    ✓ Pondera importancia de palabras                          │
│    ✓ Penaliza palabras muy comunes                            │
│    ✓ Simple y efectivo                                        │
│                                                               │
│  LIMITACIONES:                                                │
│    ✗ No captura semántica (sinónimos)                        │
│    ✗ No considera orden de palabras (más allá de n-gramas)   │
│    ✗ Vectores dispersos (sparse)                              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Métricas de evaluación para clasificadores (Precision, Recall, F1)
