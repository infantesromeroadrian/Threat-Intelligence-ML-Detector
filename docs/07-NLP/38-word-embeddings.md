# Word Embeddings: Word2Vec, GloVe y FastText

## 1. El Problema con BoW y TF-IDF

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  LIMITACIONES DE BOW/TF-IDF                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. NO CAPTURAN SEMÁNTICA                                                   │
│     "rey" y "monarca" son vectores completamente diferentes                 │
│     aunque significan lo mismo                                              │
│                                                                              │
│  2. NO CAPTURAN RELACIONES                                                  │
│     No saben que "perro" está relacionado con "gato"                        │
│     más que con "avión"                                                     │
│                                                                              │
│  3. VECTORES DISPERSOS (SPARSE)                                             │
│     Vocabulario de 50,000 palabras → vector de 50,000 dimensiones           │
│     La mayoría son 0s                                                       │
│                                                                              │
│  4. DIMENSIONALIDAD CRECE CON VOCABULARIO                                   │
│     Más palabras únicas = vectores más grandes                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

                              TF-IDF                    Word Embeddings
                         (sparse, grande)              (denso, pequeño)

  "king"           [0, 0, 0.3, 0, 0, ..., 0]         [0.2, -0.5, 0.8, 0.1]
                   ←───── 50,000 dims ─────→          ←─── 300 dims ───→

  "queen"          [0, 0, 0, 0.4, 0, ..., 0]         [0.3, -0.4, 0.7, 0.2]
                   (totalmente diferente)              (muy similar!)
```

---

## 2. ¿Qué son los Word Embeddings?

**Word Embeddings**: Representaciones vectoriales densas que capturan significado semántico.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORD EMBEDDINGS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  IDEA CENTRAL: "Las palabras se conocen por la compañía que tienen"         │
│                (Distributional Hypothesis - J.R. Firth, 1957)               │
│                                                                              │
│  Si dos palabras aparecen en contextos similares,                           │
│  probablemente tienen significados similares.                               │
│                                                                              │
│  "El _____ ladró fuerte"     → perro, can, cachorro                         │
│  "El _____ maulló"           → gato, felino, minino                         │
│                                                                              │
│  Perro y gato aparecen en contextos similares → vectores cercanos           │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PROPIEDADES MÁGICAS:                                                        │
│                                                                              │
│    vector("rey") - vector("hombre") + vector("mujer") ≈ vector("reina")     │
│                                                                              │
│    vector("París") - vector("Francia") + vector("España") ≈ vector("Madrid")│
│                                                                              │
│  Los embeddings capturan relaciones semánticas y analógicas!                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Visualización en 2D (después de PCA/t-SNE)

```
        ▲
        │     • queen
        │            • king
        │
        │  • woman
        │         • man
   ─────┼────────────────────────────────▶
        │
        │           • cat  • kitten
        │      • dog  • puppy
        │
        │
        │                    • car  • truck
        │                         • vehicle
        ▼

Palabras semánticamente similares están CERCA en el espacio vectorial
```

---

## 3. Word2Vec

**Word2Vec** (Google, 2013): El algoritmo que popularizó los word embeddings.

### 3.1 Arquitecturas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WORD2VEC ARQUITECTURAS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐         │
│  │       CBOW                  │    │       SKIP-GRAM              │         │
│  │  (Continuous Bag of Words)  │    │                              │         │
│  ├─────────────────────────────┤    ├─────────────────────────────┤         │
│  │                             │    │                              │         │
│  │  Contexto → Palabra         │    │  Palabra → Contexto         │         │
│  │                             │    │                              │         │
│  │  Input: palabras alrededor  │    │  Input: palabra central     │         │
│  │  Output: palabra central    │    │  Output: palabras alrededor │         │
│  │                             │    │                              │         │
│  │  "el gato ___ en casa"     │    │  "el gato duerme en casa"   │         │
│  │       ↓                     │    │         ↓                    │         │
│  │    "duerme"                 │    │  "el", "gato", "en", "casa" │         │
│  │                             │    │                              │         │
│  │  ✓ Rápido                   │    │  ✓ Mejor con palabras raras │         │
│  │  ✓ Bueno con datos grandes  │    │  ✓ Más usado en práctica    │         │
│  │                             │    │                              │         │
│  └─────────────────────────────┘    └─────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Skip-gram en Detalle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SKIP-GRAM                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Oración: "The quick brown fox jumps"                                       │
│  Window size = 2                                                             │
│  Target: "brown"                                                             │
│                                                                              │
│  Pares de entrenamiento:                                                     │
│    (brown, the)                                                              │
│    (brown, quick)                                                            │
│    (brown, fox)                                                              │
│    (brown, jumps)                                                            │
│                                                                              │
│  Arquitectura de red:                                                        │
│                                                                              │
│   Input          Hidden          Output                                      │
│  (one-hot)      (embedding)     (softmax)                                   │
│                                                                              │
│  [0,0,1,0,0]      W₁            [p(the), p(quick), ...]                     │
│       │       ┌───────┐                                                      │
│       └──────▶│ 300d  │──────▶  Predicción de palabras contexto             │
│               │       │     W₂                                               │
│               └───────┘                                                      │
│                  ↑                                                           │
│              Embedding                                                       │
│              de "brown"                                                      │
│                                                                              │
│  W₁ = Matriz de embeddings de input (lo que nos interesa)                   │
│  W₂ = Matriz de embeddings de output                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Negative Sampling

El softmax sobre todo el vocabulario es muy costoso. Negative sampling lo optimiza.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NEGATIVE SAMPLING                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  En lugar de predecir sobre TODO el vocabulario (50,000+ palabras):         │
│                                                                              │
│  1. Par positivo: (brown, fox) → label = 1                                  │
│                                                                              │
│  2. Muestrear K palabras negativas aleatorias:                              │
│     (brown, random₁) → label = 0                                            │
│     (brown, random₂) → label = 0                                            │
│     ...                                                                      │
│     (brown, randomₖ) → label = 0                                            │
│                                                                              │
│  3. Convertir en clasificación binaria:                                     │
│     ¿Es (word, context) un par real o fake?                                 │
│                                                                              │
│  Típicamente K = 5-20 para datasets pequeños                                │
│              K = 2-5 para datasets grandes                                  │
│                                                                              │
│  Esto reduce el costo de O(V) a O(K) por ejemplo de entrenamiento!         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Implementación con Gensim

```python
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
from typing import List

class Word2VecTrainer:
    """
    Entrenador de Word2Vec con Gensim.
    """

    def __init__(self,
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 5,
                 sg: int = 1,  # 1=skip-gram, 0=CBOW
                 negative: int = 5,
                 epochs: int = 5,
                 workers: int = 4):
        """
        Args:
            vector_size: Dimensión de los embeddings
            window: Tamaño de la ventana de contexto
            min_count: Ignorar palabras con frecuencia < min_count
            sg: 1 para Skip-gram, 0 para CBOW
            negative: Número de negative samples
            epochs: Épocas de entrenamiento
        """
        self.params = {
            'vector_size': vector_size,
            'window': window,
            'min_count': min_count,
            'sg': sg,
            'negative': negative,
            'epochs': epochs,
            'workers': workers,
        }
        self.model = None

    def train(self, sentences: List[List[str]]) -> 'Word2VecTrainer':
        """
        Entrena el modelo.

        Args:
            sentences: Lista de oraciones tokenizadas
                       [["word1", "word2"], ["word3", "word4"], ...]
        """
        self.model = Word2Vec(sentences, **self.params)
        return self

    def get_vector(self, word: str) -> np.ndarray:
        """Obtiene el embedding de una palabra."""
        return self.model.wv[word]

    def most_similar(self, word: str, topn: int = 10) -> List[tuple]:
        """Encuentra palabras más similares."""
        return self.model.wv.most_similar(word, topn=topn)

    def analogy(self, positive: List[str], negative: List[str],
                topn: int = 5) -> List[tuple]:
        """
        Resuelve analogías: king - man + woman = ?

        Args:
            positive: Palabras a sumar ["king", "woman"]
            negative: Palabras a restar ["man"]
        """
        return self.model.wv.most_similar(
            positive=positive,
            negative=negative,
            topn=topn
        )

    def similarity(self, word1: str, word2: str) -> float:
        """Calcula similitud coseno entre dos palabras."""
        return self.model.wv.similarity(word1, word2)

    def save(self, path: str):
        """Guarda el modelo."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> 'Word2VecTrainer':
        """Carga un modelo guardado."""
        trainer = cls()
        trainer.model = Word2Vec.load(path)
        return trainer


# Ejemplo de entrenamiento
sentences = [
    ["the", "malware", "infected", "the", "system"],
    ["the", "virus", "infected", "multiple", "computers"],
    ["ransomware", "encrypted", "all", "files"],
    ["the", "trojan", "installed", "a", "backdoor"],
    ["malware", "communicated", "with", "command", "and", "control"],
    ["the", "antivirus", "detected", "the", "malware"],
    ["firewall", "blocked", "malicious", "traffic"],
    # ... más oraciones de corpus de seguridad
]

# Entrenar
trainer = Word2VecTrainer(vector_size=50, window=3, min_count=1, epochs=100)
trainer.train(sentences)

# Usar
print("Palabras similares a 'malware':")
print(trainer.most_similar('malware', topn=5))
```

### 3.5 Usar Embeddings Pre-entrenados

```python
import gensim.downloader as api

# Descargar modelo pre-entrenado (puede tardar)
# word2vec-google-news-300: 3 millones de palabras, 300 dimensiones
# Entrenado en Google News (~100 mil millones de palabras)

print("Modelos disponibles:")
print(list(api.info()['models'].keys())[:10])

# Cargar modelo (primera vez descarga ~1.5GB)
# model = api.load('word2vec-google-news-300')

# Para pruebas rápidas, usar modelo más pequeño
model = api.load('glove-wiki-gigaword-50')  # 50 dimensiones

# Ejemplos
print("\nSimilares a 'malware':")
try:
    print(model.most_similar('malware', topn=5))
except KeyError:
    print("'malware' no está en el vocabulario")

print("\nSimilares a 'attack':")
print(model.most_similar('attack', topn=5))

# Analogía: king - man + woman = ?
print("\nAnalogía: king - man + woman = ?")
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
print(result)

# Similitud
print(f"\nSimilitud 'virus' - 'malware': {model.similarity('virus', 'computer'):.4f}")
```

---

## 4. GloVe (Global Vectors)

**GloVe** (Stanford, 2014): Combina estadísticas globales con aprendizaje local.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GloVe                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DIFERENCIA CON WORD2VEC:                                                   │
│                                                                              │
│  Word2Vec: Aprende de ventanas locales (predicción)                         │
│  GloVe: Usa matriz de co-ocurrencia global (factorización)                  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MATRIZ DE CO-OCURRENCIA X:                                                 │
│                                                                              │
│              ice    steam   solid   gas    water   ...                      │
│         ┌─────────────────────────────────────────────┐                     │
│  ice    │   0      2       5       0       8      ...│                      │
│  steam  │   2      0       0       4       6      ...│                      │
│  solid  │   5      0       0       0       3      ...│                      │
│  gas    │   0      4       0       0       2      ...│                      │
│  water  │   8      6       3       2       0      ...│                      │
│  ...    │  ...    ...     ...     ...     ...    ...│                      │
│         └─────────────────────────────────────────────┘                     │
│                                                                              │
│  X[i,j] = veces que palabra i aparece cerca de palabra j                    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  OBJETIVO: Encontrar vectores wᵢ, wⱼ tal que:                               │
│                                                                              │
│     wᵢᵀ · wⱼ + bᵢ + bⱼ = log(Xᵢⱼ)                                          │
│                                                                              │
│  Los vectores deben "explicar" las co-ocurrencias                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Usar GloVe Pre-entrenado

```python
import numpy as np
from typing import Dict, List
import gensim.downloader as api

class GloVeEmbeddings:
    """
    Wrapper para usar embeddings GloVe.
    """

    def __init__(self, model_name: str = 'glove-wiki-gigaword-100'):
        """
        Args:
            model_name: Nombre del modelo GloVe
                - glove-wiki-gigaword-50
                - glove-wiki-gigaword-100
                - glove-wiki-gigaword-200
                - glove-wiki-gigaword-300
                - glove-twitter-25
                - glove-twitter-50
                - glove-twitter-100
                - glove-twitter-200
        """
        print(f"Cargando {model_name}...")
        self.model = api.load(model_name)
        self.vector_size = self.model.vector_size
        print(f"Cargado. Vocabulario: {len(self.model)} palabras")

    def get_vector(self, word: str) -> np.ndarray:
        """Obtiene embedding de una palabra."""
        try:
            return self.model[word]
        except KeyError:
            return None

    def get_sentence_embedding(self, tokens: List[str],
                               method: str = 'mean') -> np.ndarray:
        """
        Obtiene embedding de una oración.

        Args:
            tokens: Lista de tokens
            method: 'mean' o 'sum'
        """
        vectors = []
        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)

        if not vectors:
            return np.zeros(self.vector_size)

        vectors = np.array(vectors)

        if method == 'mean':
            return vectors.mean(axis=0)
        else:
            return vectors.sum(axis=0)

    def most_similar(self, word: str, topn: int = 10) -> List[tuple]:
        """Encuentra palabras similares."""
        return self.model.most_similar(word, topn=topn)

    def similarity(self, word1: str, word2: str) -> float:
        """Similitud entre dos palabras."""
        return self.model.similarity(word1, word2)


# Ejemplo
glove = GloVeEmbeddings('glove-wiki-gigaword-50')

# Embedding de oración
sentence = ["the", "hacker", "exploited", "a", "vulnerability"]
sent_vec = glove.get_sentence_embedding(sentence)
print(f"Embedding de oración shape: {sent_vec.shape}")

# Similares
print("\nPalabras similares a 'hacker':")
print(glove.most_similar('hacker', topn=5))
```

---

## 5. FastText

**FastText** (Facebook, 2016): Embeddings a nivel de subpalabras (character n-grams).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FastText                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INNOVACIÓN: Representa palabras como suma de n-gramas de caracteres        │
│                                                                              │
│  Palabra: "where" con n=3                                                   │
│                                                                              │
│  N-gramas: <wh, whe, her, ere, re>, <where>                                 │
│            ↑                              ↑                                  │
│       subpalabras               palabra completa                            │
│                                                                              │
│  vector("where") = vector("<wh") + vector("whe") + ... + vector("<where>")  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  VENTAJAS SOBRE WORD2VEC:                                                   │
│                                                                              │
│  1. MANEJA PALABRAS FUERA DE VOCABULARIO (OOV)                              │
│     Word2Vec: "unfairness" no en vocab → error                              │
│     FastText: "unfairness" → construir de n-gramas                          │
│                              <un, unf, nfa, fai, air, irn, rne, nes, ess>   │
│                              Comparte con "unfair", "fairness", etc.        │
│                                                                              │
│  2. MEJOR PARA PALABRAS RARAS                                               │
│     Palabras poco frecuentes comparten n-gramas con palabras comunes        │
│                                                                              │
│  3. MEJOR PARA IDIOMAS CON MORFOLOGÍA RICA                                  │
│     Español, Alemán, Finés... donde hay muchas variantes de palabras        │
│                                                                              │
│  4. ROBUSTO A TYPOS                                                          │
│     "hackr" comparte n-gramas con "hacker"                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementación con Gensim

```python
from gensim.models import FastText
from typing import List
import numpy as np

class FastTextTrainer:
    """
    Entrenador de FastText.
    """

    def __init__(self,
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 5,
                 min_n: int = 3,  # Mínimo tamaño de n-grama
                 max_n: int = 6,  # Máximo tamaño de n-grama
                 epochs: int = 5,
                 sg: int = 1):
        """
        Args:
            min_n: Mínimo tamaño de character n-gram
            max_n: Máximo tamaño de character n-gram
        """
        self.params = {
            'vector_size': vector_size,
            'window': window,
            'min_count': min_count,
            'min_n': min_n,
            'max_n': max_n,
            'epochs': epochs,
            'sg': sg,
        }
        self.model = None

    def train(self, sentences: List[List[str]]) -> 'FastTextTrainer':
        """Entrena el modelo."""
        self.model = FastText(sentences, **self.params)
        return self

    def get_vector(self, word: str) -> np.ndarray:
        """
        Obtiene embedding de una palabra.
        FUNCIONA INCLUSO SI LA PALABRA NO ESTÁ EN EL VOCABULARIO!
        """
        return self.model.wv[word]

    def most_similar(self, word: str, topn: int = 10) -> List[tuple]:
        """Encuentra palabras similares."""
        return self.model.wv.most_similar(word, topn=topn)


# Ejemplo mostrando ventaja con OOV
sentences = [
    ["malware", "infected", "the", "system"],
    ["ransomware", "encrypted", "files"],
    ["trojan", "installed", "backdoor"],
    ["virus", "spread", "quickly"],
    ["worm", "replicated", "across", "network"],
]

trainer = FastTextTrainer(vector_size=50, min_count=1, epochs=50)
trainer.train(sentences)

# Probar con palabra OOV (no en el corpus)
oov_word = "malwares"  # Plural, no visto en entrenamiento

print(f"Vector de '{oov_word}' (OOV):")
vec = trainer.get_vector(oov_word)
print(f"  Shape: {vec.shape}")
print(f"  Funciona gracias a n-gramas compartidos con 'malware'")

# Probar con typo
typo_word = "mawlare"  # Typo de 'malware'
print(f"\nVector de '{typo_word}' (typo):")
vec_typo = trainer.get_vector(typo_word)
print(f"  Shape: {vec_typo.shape}")
```

### FastText Pre-entrenado

```python
import fasttext.util
import fasttext

# Descargar modelo pre-entrenado (primera vez)
# fasttext.util.download_model('en', if_exists='ignore')

# Cargar modelo
# ft = fasttext.load_model('cc.en.300.bin')

# O usar gensim para modelos más pequeños
import gensim.downloader as api

# Modelos FastText en gensim
# 'fasttext-wiki-news-subwords-300'

# Ejemplo con API de gensim
ft_model = api.load('fasttext-wiki-news-subwords-300')

# Funciona con OOV
oov_examples = ['cyberattck', 'phishng', 'malwar3', 'h4ck3r']

print("Embeddings para palabras OOV/typos:")
for word in oov_examples:
    try:
        vec = ft_model[word]
        print(f"  {word}: vector de {vec.shape[0]} dimensiones")
    except KeyError:
        print(f"  {word}: no disponible")
```

---

## 6. Comparación de Métodos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   COMPARACIÓN DE EMBEDDINGS                                  │
├───────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│   Aspecto     │    Word2Vec     │     GloVe       │      FastText           │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ Método        │ Predicción      │ Factorización   │ Predicción + n-gramas   │
│               │ (ventana local) │ (global)        │                         │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ OOV           │ ✗ No maneja     │ ✗ No maneja     │ ✓ Maneja bien           │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ Typos         │ ✗ No            │ ✗ No            │ ✓ Robusto               │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ Palabras raras│ Regular         │ Regular         │ ✓ Mejor                 │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ Velocidad     │ ✓ Rápido        │ ✓ Rápido        │ Más lento               │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ Memoria       │ Normal          │ Normal          │ Mayor (n-gramas)        │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ Morfología    │ Regular         │ Regular         │ ✓ Excelente             │
├───────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│ Uso típico    │ General         │ General         │ Idiomas morfológicos,   │
│               │                 │                 │ dominios con OOV        │
└───────────────┴─────────────────┴─────────────────┴─────────────────────────┘
```

---

## 7. Usando Embeddings para Clasificación

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import gensim.downloader as api
from typing import List, Tuple

class EmbeddingClassifier:
    """
    Clasificador de texto usando word embeddings.
    """

    def __init__(self, embedding_model=None):
        """
        Args:
            embedding_model: Modelo de embeddings pre-cargado
        """
        if embedding_model is None:
            print("Cargando embeddings...")
            self.embeddings = api.load('glove-wiki-gigaword-50')
        else:
            self.embeddings = embedding_model

        self.vector_size = self.embeddings.vector_size
        self.classifier = LogisticRegression(max_iter=1000)

    def text_to_vector(self, tokens: List[str],
                       method: str = 'mean') -> np.ndarray:
        """
        Convierte texto tokenizado a vector.

        Methods:
            'mean': Promedio de embeddings (más común)
            'sum': Suma de embeddings
            'max': Max pooling por dimensión
        """
        vectors = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.embeddings:
                vectors.append(self.embeddings[token_lower])

        if not vectors:
            return np.zeros(self.vector_size)

        vectors = np.array(vectors)

        if method == 'mean':
            return vectors.mean(axis=0)
        elif method == 'sum':
            return vectors.sum(axis=0)
        elif method == 'max':
            return vectors.max(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def prepare_data(self, texts: List[List[str]],
                     labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento.

        Args:
            texts: Lista de textos tokenizados
            labels: Lista de etiquetas
        """
        X = np.array([self.text_to_vector(text) for text in texts])
        y = np.array(labels)
        return X, y

    def train(self, X: np.ndarray, y: np.ndarray):
        """Entrena el clasificador."""
        self.classifier.fit(X, y)

    def predict(self, texts: List[List[str]]) -> np.ndarray:
        """Predice etiquetas para textos."""
        X = np.array([self.text_to_vector(text) for text in texts])
        return self.classifier.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 target_names: List[str] = None):
        """Evalúa el modelo."""
        y_pred = self.classifier.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=target_names))


# Ejemplo: Clasificación de emails (spam vs ham)
# Datos de ejemplo
texts = [
    # Spam
    ["free", "money", "click", "here", "now"],
    ["win", "prize", "lottery", "claim"],
    ["urgent", "action", "required", "account"],
    ["congratulations", "winner", "selected"],
    ["discount", "offer", "limited", "time"],
    # Ham
    ["meeting", "tomorrow", "office", "discuss"],
    ["project", "update", "schedule", "review"],
    ["thank", "you", "help", "yesterday"],
    ["dinner", "tonight", "family", "home"],
    ["report", "attached", "please", "review"],
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=spam, 0=ham

# Entrenar y evaluar
classifier = EmbeddingClassifier()
X, y = classifier.prepare_data(texts, labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

classifier.train(X_train, y_train)
classifier.evaluate(X_test, y_test, target_names=['ham', 'spam'])

# Predecir nuevo texto
new_text = ["free", "iphone", "click", "link"]
prediction = classifier.predict([new_text])
print(f"\nTexto: {new_text}")
print(f"Predicción: {'spam' if prediction[0] == 1 else 'ham'}")
```

---

## 8. Embeddings para Seguridad

### Entrenar Embeddings de Dominio Específico

```python
from gensim.models import Word2Vec
from typing import List
import re

class SecurityEmbeddings:
    """
    Word embeddings entrenados en corpus de seguridad.

    Corpus recomendados:
    - CVE descriptions
    - Security advisories
    - Threat reports (APT reports)
    - Malware analysis reports
    - Security blogs/articles
    """

    def __init__(self, vector_size: int = 100):
        self.vector_size = vector_size
        self.model = None

    def preprocess_security_text(self, text: str) -> List[str]:
        """
        Preprocesamiento específico para textos de seguridad.
        Preserva elementos importantes.
        """
        text = text.lower()

        # Preservar CVE IDs
        text = re.sub(r'cve-(\d{4})-(\d+)', r'CVE_\1_\2', text)

        # Preservar CWE IDs
        text = re.sub(r'cwe-(\d+)', r'CWE_\1', text)

        # Preservar hashes (simplificados)
        text = re.sub(r'\b[a-f0-9]{32}\b', 'HASH_MD5', text)
        text = re.sub(r'\b[a-f0-9]{64}\b', 'HASH_SHA256', text)

        # Preservar IPs
        text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP_ADDR', text)

        # Tokenizar
        tokens = re.findall(r'\b[\w_]+\b', text)

        # Filtrar muy cortos
        tokens = [t for t in tokens if len(t) > 2]

        return tokens

    def train(self, documents: List[str], epochs: int = 10):
        """
        Entrena embeddings en corpus de seguridad.

        Args:
            documents: Lista de documentos de texto
        """
        # Preprocesar
        sentences = [self.preprocess_security_text(doc) for doc in documents]

        # Entrenar Word2Vec
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=5,
            min_count=2,
            sg=1,  # Skip-gram
            epochs=epochs,
            workers=4
        )

        print(f"Modelo entrenado. Vocabulario: {len(self.model.wv)} palabras")

    def find_related_terms(self, term: str, topn: int = 10) -> List[tuple]:
        """Encuentra términos relacionados."""
        if term not in self.model.wv:
            print(f"'{term}' no está en el vocabulario")
            return []
        return self.model.wv.most_similar(term, topn=topn)

    def term_similarity(self, term1: str, term2: str) -> float:
        """Similitud entre dos términos."""
        try:
            return self.model.wv.similarity(term1, term2)
        except KeyError:
            return 0.0


# Ejemplo con corpus de seguridad simulado
security_corpus = [
    "A buffer overflow vulnerability in OpenSSL allows remote code execution",
    "The CVE-2021-44228 Log4j vulnerability enables remote code execution via JNDI injection",
    "SQL injection vulnerability discovered in login form allows authentication bypass",
    "Cross-site scripting XSS flaw in web application allows cookie theft",
    "Remote code execution vulnerability in Apache Struts allows server compromise",
    "Memory corruption bug in kernel driver leads to privilege escalation",
    "The malware uses DLL injection to evade detection and persist on the system",
    "Ransomware encrypts files using AES-256 and demands Bitcoin payment",
    "The trojan establishes connection to command and control server",
    "Phishing campaign targets financial institutions with credential harvesting",
    "APT group exploits zero-day vulnerability in enterprise software",
    "Buffer overflow in image parser allows arbitrary code execution",
    # ... más documentos de CVEs, reportes, etc.
]

# Entrenar
sec_embeddings = SecurityEmbeddings(vector_size=50)
sec_embeddings.train(security_corpus, epochs=50)

# Explorar relaciones
print("Términos relacionados con 'vulnerability':")
for term, score in sec_embeddings.find_related_terms('vulnerability', 5):
    print(f"  {term}: {score:.4f}")

print("\nTérminos relacionados con 'malware':")
for term, score in sec_embeddings.find_related_terms('malware', 5):
    print(f"  {term}: {score:.4f}")

# Similitudes
pairs = [
    ('vulnerability', 'exploit'),
    ('malware', 'trojan'),
    ('buffer', 'overflow'),
    ('injection', 'xss'),
]

print("\nSimilitudes:")
for t1, t2 in pairs:
    sim = sec_embeddings.term_similarity(t1, t2)
    print(f"  {t1} - {t2}: {sim:.4f}")
```

---

## 9. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WORD EMBEDDINGS - RESUMEN                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CONCEPTO: Representar palabras como vectores densos que capturan           │
│            significado semántico                                            │
│                                                                              │
│  MÉTODOS PRINCIPALES:                                                        │
│    • Word2Vec: Skip-gram o CBOW, aprende de contexto local                  │
│    • GloVe: Factoriza matriz de co-ocurrencia global                        │
│    • FastText: Usa n-gramas de caracteres, maneja OOV                       │
│                                                                              │
│  PROPIEDADES:                                                                │
│    • Palabras similares → vectores cercanos                                 │
│    • Capturan analogías: king - man + woman ≈ queen                         │
│    • Vectores densos (100-300 dims) vs sparse (10,000+ dims)               │
│                                                                              │
│  USO TÍPICO:                                                                 │
│    1. Usar embeddings pre-entrenados (GloVe, FastText)                      │
│    2. Fine-tune en tu dominio si es muy específico                          │
│    3. Promediar embeddings de palabras para representar documentos          │
│    4. Usar como input para modelos de clasificación/NER/etc.                │
│                                                                              │
│  CUÁNDO USAR CADA UNO:                                                       │
│    • Texto general: Word2Vec o GloVe pre-entrenado                          │
│    • Muchos OOV/typos: FastText                                             │
│    • Dominio muy específico: Entrenar propio + fine-tune                    │
│    • Mejor rendimiento: Usar como input para BERT (siguiente tema)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
