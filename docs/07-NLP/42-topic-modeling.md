# Topic Modeling: LDA y LSA

## 1. ¿Qué es Topic Modeling?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOPIC MODELING                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  OBJETIVO: Descubrir temas ocultos en una colección de documentos           │
│            de forma NO supervisada (sin etiquetas)                          │
│                                                                              │
│  INPUT:  Colección de documentos (corpus)                                   │
│  OUTPUT: K temas, cada uno definido por palabras características            │
│                                                                              │
│  EJEMPLO:                                                                    │
│                                                                              │
│  Documentos:                                                                 │
│    Doc1: "The new iPhone has amazing camera features"                       │
│    Doc2: "Stock market crashed due to inflation concerns"                   │
│    Doc3: "Apple announced record quarterly earnings"                        │
│    Doc4: "Interest rates impact housing market"                             │
│                                                                              │
│  Temas descubiertos:                                                         │
│    Topic 1 (Tecnología): iPhone, camera, features, Apple, announced         │
│    Topic 2 (Finanzas):   market, stock, inflation, earnings, rates          │
│                                                                              │
│  Cada documento es una MEZCLA de temas:                                     │
│    Doc1: 95% Tecnología, 5% Finanzas                                        │
│    Doc3: 60% Tecnología, 40% Finanzas  ← Apple = tech + finance             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Aplicaciones

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      APLICACIONES DE TOPIC MODELING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GENERAL:                                                                    │
│    • Organización automática de documentos                                  │
│    • Sistemas de recomendación basados en contenido                         │
│    • Análisis de tendencias en redes sociales                               │
│    • Exploración de grandes corpus de texto                                 │
│                                                                              │
│  EN SEGURIDAD:                                                               │
│    • Categorización automática de CVEs por tipo                             │
│    • Análisis de reportes de threat intelligence                            │
│    • Clustering de muestras de malware por comportamiento                   │
│    • Detección de temas emergentes en logs/eventos                          │
│    • Análisis de phishing emails por campaña                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. LSA (Latent Semantic Analysis)

### 2.1 Concepto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LSA                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LSA = TF-IDF + SVD (Singular Value Decomposition)                          │
│                                                                              │
│  PROCESO:                                                                    │
│                                                                              │
│  1. Crear matriz documento-término (TF-IDF)                                 │
│                                                                              │
│            word₁  word₂  word₃  ...  wordₙ                                  │
│     doc₁ [  0.1    0.5    0.0   ...   0.2  ]                                │
│     doc₂ [  0.0    0.3    0.4   ...   0.0  ]                                │
│     ...                                                                      │
│     docₘ [  0.2    0.0    0.1   ...   0.3  ]                                │
│                                                                              │
│     Shape: (m documentos, n palabras)                                       │
│                                                                              │
│  2. Aplicar SVD:  A = U · Σ · Vᵀ                                           │
│                                                                              │
│     U: Documentos en espacio de temas   (m × k)                             │
│     Σ: Valores singulares (importancia) (k × k)                             │
│     Vᵀ: Palabras en espacio de temas    (k × n)                             │
│                                                                              │
│  3. Truncar a k dimensiones (temas)                                         │
│                                                                              │
│     Los k componentes principales = k "temas"                               │
│     Cada tema es una combinación lineal de palabras                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementación

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from typing import List, Tuple

class LSAModel:
    """
    Latent Semantic Analysis para Topic Modeling.
    """

    def __init__(self,
                 n_topics: int = 10,
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2)):
        self.n_topics = n_topics
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range
        )

        self.svd = TruncatedSVD(
            n_components=n_topics,
            random_state=42
        )

    def fit(self, documents: List[str]) -> 'LSAModel':
        """Entrena el modelo."""
        # TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        # SVD
        self.doc_topics = self.svd.fit_transform(self.tfidf_matrix)

        # Vocabulario
        self.feature_names = self.vectorizer.get_feature_names_out()

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transforma nuevos documentos a espacio de temas."""
        tfidf = self.vectorizer.transform(documents)
        return self.svd.transform(tfidf)

    def get_topics(self, n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Retorna las palabras más importantes por tema.
        """
        topics = []

        for topic_idx, topic in enumerate(self.svd.components_):
            # Ordenar palabras por peso en el tema
            top_indices = topic.argsort()[:-n_words-1:-1]
            top_words = [(self.feature_names[i], topic[i])
                        for i in top_indices]
            topics.append(top_words)

        return topics

    def print_topics(self, n_words: int = 10):
        """Imprime los temas descubiertos."""
        topics = self.get_topics(n_words)

        for idx, topic in enumerate(topics):
            words = [word for word, _ in topic]
            print(f"Topic {idx}: {', '.join(words)}")

    def get_document_topics(self, doc_idx: int) -> List[Tuple[int, float]]:
        """
        Retorna distribución de temas para un documento.
        """
        doc_topic_dist = self.doc_topics[doc_idx]

        # Normalizar
        doc_topic_dist = np.abs(doc_topic_dist)
        doc_topic_dist = doc_topic_dist / doc_topic_dist.sum()

        return [(i, score) for i, score in enumerate(doc_topic_dist)]


# Ejemplo
documents = [
    "The vulnerability allows remote code execution through buffer overflow",
    "Ransomware encrypted all files and demanded bitcoin payment",
    "SQL injection attack bypassed authentication mechanism",
    "Phishing email contained malicious attachment with trojan",
    "Buffer overflow in kernel driver leads to privilege escalation",
    "Ransomware gang demands million dollar ransom from hospital",
    "Cross-site scripting vulnerability found in web application",
    "Malware communicates with command and control server",
]

lsa = LSAModel(n_topics=3)
lsa.fit(documents)

print("=== TEMAS DESCUBIERTOS (LSA) ===\n")
lsa.print_topics(n_words=5)

print("\n=== DISTRIBUCIÓN DE TEMAS POR DOCUMENTO ===\n")
for i, doc in enumerate(documents[:4]):
    topics = lsa.get_document_topics(i)
    print(f"Doc {i}: {doc[:50]}...")
    for topic_id, score in sorted(topics, key=lambda x: -x[1])[:2]:
        print(f"  Topic {topic_id}: {score:.2f}")
```

---

## 3. LDA (Latent Dirichlet Allocation)

### 3.1 Concepto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LDA                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LDA es un MODELO PROBABILÍSTICO GENERATIVO                                 │
│                                                                              │
│  ASUMPCIONES:                                                                │
│    1. Cada documento es una mezcla de temas                                 │
│    2. Cada tema es una distribución sobre palabras                          │
│                                                                              │
│  PROCESO GENERATIVO (cómo LDA imagina que se escriben documentos):          │
│                                                                              │
│  Para cada documento d:                                                      │
│    1. Elegir mezcla de temas θd ~ Dirichlet(α)                              │
│       ej: θ = [0.6, 0.3, 0.1] → 60% topic1, 30% topic2, 10% topic3          │
│                                                                              │
│    2. Para cada palabra w en el documento:                                  │
│       a. Elegir un tema z ~ Multinomial(θd)                                 │
│          ej: z = topic1 (con prob 0.6)                                      │
│                                                                              │
│       b. Elegir palabra w ~ Multinomial(φz)                                 │
│          ej: w = "malware" (palabra probable en topic1)                     │
│                                                                              │
│  ENTRENAMIENTO:                                                              │
│    Invertir el proceso: dado los documentos,                                │
│    descubrir θ (mezcla de temas) y φ (distribución de palabras por tema)   │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LSA vs LDA:                                                                 │
│                                                                              │
│  LSA:                                                                        │
│    - Algebra lineal (SVD)                                                   │
│    - Componentes pueden ser negativos                                       │
│    - Más rápido                                                             │
│                                                                              │
│  LDA:                                                                        │
│    - Modelo probabilístico                                                  │
│    - Distribuciones (suman a 1, no negativos)                               │
│    - Más interpretable                                                       │
│    - Puede generar nuevos documentos                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Parámetros

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PARÁMETROS DE LDA                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  K (n_topics):                                                               │
│    Número de temas a descubrir                                              │
│    - Muy pocos: temas demasiado generales                                   │
│    - Muchos: temas redundantes o sin sentido                                │
│    - Típico: 10-100 dependiendo del corpus                                  │
│                                                                              │
│  α (alpha) - Prior sobre distribución documento-tema:                       │
│    - α alto: documentos cubren muchos temas (mezcla uniforme)              │
│    - α bajo: documentos se enfocan en pocos temas                          │
│    - Típico: 50/K o "auto" para aprender                                   │
│                                                                              │
│  β (beta/eta) - Prior sobre distribución tema-palabra:                      │
│    - β alto: temas usan muchas palabras (generales)                        │
│    - β bajo: temas usan pocas palabras (específicos)                       │
│    - Típico: 0.01 o "auto"                                                 │
│                                                                              │
│  Iteraciones:                                                                │
│    - LDA usa Gibbs Sampling o Variational Inference                        │
│    - Típico: 100-500 iteraciones                                           │
│    - Más iteraciones = mejor convergencia pero más lento                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Implementación con Gensim

```python
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    remove_stopwords,
    strip_short
)
import numpy as np
from typing import List, Tuple
import pyLDAvis.gensim_models

class LDAModel:
    """
    LDA con Gensim para Topic Modeling.
    """

    def __init__(self,
                 n_topics: int = 10,
                 alpha: str = 'auto',
                 eta: str = 'auto',
                 passes: int = 10,
                 iterations: int = 100):
        self.n_topics = n_topics
        self.alpha = alpha
        self.eta = eta
        self.passes = passes
        self.iterations = iterations

        # Preprocesamiento
        self.filters = [
            lambda x: x.lower(),
            strip_punctuation,
            strip_numeric,
            remove_stopwords,
            strip_short,
        ]

    def preprocess(self, documents: List[str]) -> List[List[str]]:
        """Preprocesa documentos."""
        processed = []
        for doc in documents:
            tokens = preprocess_string(doc, self.filters)
            processed.append(tokens)
        return processed

    def fit(self, documents: List[str]) -> 'LDAModel':
        """Entrena el modelo LDA."""
        # Preprocesar
        self.processed_docs = self.preprocess(documents)

        # Crear diccionario
        self.dictionary = corpora.Dictionary(self.processed_docs)

        # Filtrar extremos
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)

        # Crear corpus (bag of words)
        self.corpus = [self.dictionary.doc2bow(doc)
                      for doc in self.processed_docs]

        # Entrenar LDA
        self.model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            alpha=self.alpha,
            eta=self.eta,
            passes=self.passes,
            iterations=self.iterations,
            random_state=42,
            workers=4
        )

        return self

    def get_topics(self, n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """Retorna palabras top por tema."""
        topics = []
        for topic_id in range(self.n_topics):
            topic_words = self.model.show_topic(topic_id, n_words)
            topics.append(topic_words)
        return topics

    def print_topics(self, n_words: int = 10):
        """Imprime los temas."""
        for idx, topic in enumerate(self.get_topics(n_words)):
            words = [f"{word}({prob:.3f})" for word, prob in topic]
            print(f"Topic {idx}: {', '.join(words)}")

    def get_document_topics(self, document: str) -> List[Tuple[int, float]]:
        """Obtiene distribución de temas para un documento."""
        tokens = preprocess_string(document, self.filters)
        bow = self.dictionary.doc2bow(tokens)
        return self.model.get_document_topics(bow)

    def compute_coherence(self, coherence: str = 'c_v') -> float:
        """
        Calcula coherencia del modelo.
        Valores más altos = mejores temas.
        """
        coherence_model = CoherenceModel(
            model=self.model,
            texts=self.processed_docs,
            dictionary=self.dictionary,
            coherence=coherence
        )
        return coherence_model.get_coherence()

    def visualize(self):
        """Genera visualización interactiva."""
        vis_data = pyLDAvis.gensim_models.prepare(
            self.model,
            self.corpus,
            self.dictionary
        )
        return vis_data


# Ejemplo
documents = [
    "The vulnerability allows remote code execution through buffer overflow in the kernel",
    "Ransomware encrypted all files demanding bitcoin payment from the victim",
    "SQL injection attack bypassed authentication mechanism in the web application",
    "Phishing email contained malicious attachment delivering trojan malware",
    "Buffer overflow vulnerability in network driver causes system crash",
    "Ransomware gang demands million dollar ransom cryptocurrency payment",
    "Cross-site scripting XSS vulnerability found in JavaScript application",
    "Malware establishes persistence and communicates with command control server",
    "Remote code execution vulnerability exploited in Apache Struts framework",
    "Banking trojan steals credentials through keylogger and screen capture",
]

lda = LDAModel(n_topics=3, passes=20)
lda.fit(documents)

print("=== TEMAS DESCUBIERTOS (LDA) ===\n")
lda.print_topics(n_words=5)

print(f"\nCoherencia del modelo: {lda.compute_coherence():.4f}")

print("\n=== TEMAS POR DOCUMENTO ===\n")
for doc in documents[:3]:
    print(f"Doc: {doc[:60]}...")
    topics = lda.get_document_topics(doc)
    for topic_id, prob in sorted(topics, key=lambda x: -x[1]):
        if prob > 0.1:
            print(f"  Topic {topic_id}: {prob:.2f}")
```

### 3.4 Selección del Número de Temas

```python
import matplotlib.pyplot as plt
from typing import List, Tuple

def find_optimal_topics(documents: List[str],
                        topic_range: range = range(2, 15),
                        passes: int = 10) -> Tuple[int, List[float]]:
    """
    Encuentra el número óptimo de temas usando coherencia.
    """
    coherence_scores = []

    for n_topics in topic_range:
        print(f"Probando {n_topics} temas...", end=" ")

        model = LDAModel(n_topics=n_topics, passes=passes)
        model.fit(documents)
        coherence = model.compute_coherence()
        coherence_scores.append(coherence)

        print(f"Coherencia: {coherence:.4f}")

    # Mejor número de temas
    best_idx = np.argmax(coherence_scores)
    best_n_topics = list(topic_range)[best_idx]

    # Visualizar
    plt.figure(figsize=(10, 6))
    plt.plot(list(topic_range), coherence_scores, 'bo-')
    plt.axvline(x=best_n_topics, color='r', linestyle='--',
                label=f'Óptimo: {best_n_topics} temas')
    plt.xlabel('Número de Temas')
    plt.ylabel('Coherencia (c_v)')
    plt.title('Selección del Número de Temas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return best_n_topics, coherence_scores


# Uso
# best_n, scores = find_optimal_topics(documents, range(2, 10))
```

---

## 4. BERTopic (Modern Topic Modeling)

### 4.1 Concepto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BERTopic                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BERTopic = BERT Embeddings + UMAP + HDBSCAN + c-TF-IDF                     │
│                                                                              │
│  PIPELINE:                                                                   │
│                                                                              │
│  1. Embeddings con BERT/Sentence-Transformers                               │
│     Cada documento → vector denso (768d)                                    │
│                                                                              │
│  2. Reducción con UMAP                                                       │
│     768d → 5d (más fácil de clusterizar)                                    │
│                                                                              │
│  3. Clustering con HDBSCAN                                                   │
│     Encuentra clusters densos (= temas)                                     │
│     No requiere especificar K                                               │
│                                                                              │
│  4. Representación con c-TF-IDF                                             │
│     Para cada cluster, extraer palabras representativas                     │
│     c-TF-IDF: TF-IDF a nivel de cluster                                     │
│                                                                              │
│  VENTAJAS sobre LDA:                                                         │
│    • Usa embeddings contextuales (mejor semántica)                          │
│    • No requiere especificar número de temas                                │
│    • Maneja mejor textos cortos                                             │
│    • Resultados más coherentes                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Implementación

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import List

class BERTopicModel:
    """
    Topic Modeling moderno usando BERTopic.
    """

    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 min_topic_size: int = 5,
                 n_gram_range: tuple = (1, 2)):
        """
        Args:
            embedding_model: Modelo de sentence transformers
            min_topic_size: Mínimo documentos por tema
        """
        self.embedding_model = SentenceTransformer(embedding_model)

        self.vectorizer = CountVectorizer(
            ngram_range=n_gram_range,
            stop_words='english'
        )

        self.model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer,
            min_topic_size=min_topic_size,
            verbose=True
        )

    def fit(self, documents: List[str]) -> 'BERTopicModel':
        """Entrena el modelo."""
        self.topics, self.probs = self.model.fit_transform(documents)
        self.documents = documents
        return self

    def get_topics(self) -> dict:
        """Retorna información de todos los temas."""
        return self.model.get_topics()

    def get_topic_info(self):
        """Retorna DataFrame con info de temas."""
        return self.model.get_topic_info()

    def get_document_topics(self, doc_idx: int) -> tuple:
        """Retorna tema y probabilidad de un documento."""
        return self.topics[doc_idx], self.probs[doc_idx]

    def find_topics(self, search_term: str, top_n: int = 5):
        """Busca temas relacionados con un término."""
        return self.model.find_topics(search_term, top_n=top_n)

    def visualize_topics(self):
        """Visualización interactiva de temas."""
        return self.model.visualize_topics()

    def visualize_documents(self):
        """Visualización de documentos en espacio 2D."""
        return self.model.visualize_documents(self.documents)

    def visualize_barchart(self, top_n_topics: int = 10):
        """Barchart de palabras por tema."""
        return self.model.visualize_barchart(top_n_topics=top_n_topics)


# Ejemplo
documents = [
    "Critical buffer overflow vulnerability in OpenSSL allows RCE",
    "New ransomware variant encrypts files with AES-256",
    "SQL injection discovered in popular WordPress plugin",
    "Phishing campaign targets banking customers with fake emails",
    "Kernel vulnerability leads to privilege escalation on Linux",
    "Ransomware gang demands 2 million in Bitcoin",
    "XSS vulnerability found in React application",
    "Trojan malware steals credentials from browsers",
    "Remote code execution via deserialization bug",
    "Cryptominer malware abuses cloud resources",
    "Authentication bypass in REST API endpoint",
    "Botnet spreads through unpatched IoT devices",
]

# Crear y entrenar modelo
model = BERTopicModel(min_topic_size=2)
model.fit(documents)

# Ver temas
print("=== TEMAS DESCUBIERTOS (BERTopic) ===\n")
topic_info = model.get_topic_info()
print(topic_info)

# Temas por documento
print("\n=== TEMA POR DOCUMENTO ===")
for i, doc in enumerate(documents[:5]):
    topic, prob = model.get_document_topics(i)
    print(f"Topic {topic} ({prob:.2f}): {doc[:50]}...")
```

---

## 5. Aplicaciones en Seguridad

### 5.1 Categorización de CVEs

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from typing import List, Dict

class CVETopicAnalyzer:
    """
    Analiza CVEs usando Topic Modeling para categorización automática.
    """

    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )

    def fit(self, cve_descriptions: List[str]) -> 'CVETopicAnalyzer':
        """Entrena con descripciones de CVEs."""
        self.tfidf = self.vectorizer.fit_transform(cve_descriptions)
        self.doc_topics = self.lda.fit_transform(self.tfidf)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def get_topic_labels(self) -> Dict[int, str]:
        """
        Intenta asignar etiquetas semánticas a los temas
        basándose en las palabras clave.
        """
        # Palabras clave por categoría de vulnerabilidad
        category_keywords = {
            'Buffer Overflow': ['buffer', 'overflow', 'stack', 'heap', 'memory'],
            'SQL Injection': ['sql', 'injection', 'database', 'query'],
            'XSS': ['xss', 'script', 'cross', 'site', 'javascript'],
            'RCE': ['remote', 'code', 'execution', 'command'],
            'DoS': ['denial', 'service', 'crash', 'dos', 'resource'],
            'Auth Bypass': ['authentication', 'bypass', 'authorization', 'access'],
            'Info Disclosure': ['information', 'disclosure', 'leak', 'sensitive'],
            'Privilege Escalation': ['privilege', 'escalation', 'root', 'admin'],
            'Path Traversal': ['path', 'traversal', 'directory', 'file'],
            'CSRF': ['csrf', 'request', 'forgery', 'cross'],
        }

        topic_labels = {}

        for topic_idx in range(self.n_topics):
            topic_words = self._get_topic_words(topic_idx, n=20)
            topic_word_set = set([w for w, _ in topic_words])

            # Encontrar mejor match
            best_category = 'Other'
            best_score = 0

            for category, keywords in category_keywords.items():
                score = len(topic_word_set.intersection(keywords))
                if score > best_score:
                    best_score = score
                    best_category = category

            topic_labels[topic_idx] = best_category

        return topic_labels

    def _get_topic_words(self, topic_idx: int, n: int = 10):
        """Obtiene palabras top de un tema."""
        topic = self.lda.components_[topic_idx]
        top_indices = topic.argsort()[:-n-1:-1]
        return [(self.feature_names[i], topic[i]) for i in top_indices]

    def categorize(self, cve_description: str) -> Dict:
        """Categoriza una nueva descripción de CVE."""
        tfidf = self.vectorizer.transform([cve_description])
        topic_dist = self.lda.transform(tfidf)[0]

        top_topic = topic_dist.argmax()
        labels = self.get_topic_labels()

        return {
            'top_topic': top_topic,
            'category': labels[top_topic],
            'confidence': topic_dist[top_topic],
            'topic_distribution': {
                labels.get(i, f'Topic_{i}'): prob
                for i, prob in enumerate(topic_dist)
                if prob > 0.1
            }
        }


# Ejemplo
cve_descriptions = [
    "A stack-based buffer overflow in the handling of HTTP headers allows attackers to execute arbitrary code",
    "The application does not properly sanitize user input, allowing SQL injection attacks",
    "A vulnerability in the authentication mechanism allows attackers to bypass login",
    "Cross-site scripting (XSS) vulnerability in the search functionality",
    "Remote code execution via crafted serialized object",
    "Denial of service via malformed network packet",
    "Information disclosure through error messages",
    "Privilege escalation through symbolic link following",
    "Directory traversal allows reading arbitrary files",
    "Buffer overflow in image parsing library",
]

analyzer = CVETopicAnalyzer(n_topics=5)
analyzer.fit(cve_descriptions)

# Categorizar nueva CVE
new_cve = "A heap-based buffer overflow in the PDF parser allows remote code execution"
result = analyzer.categorize(new_cve)

print(f"CVE: {new_cve}")
print(f"Categoría: {result['category']}")
print(f"Confianza: {result['confidence']:.2f}")
print(f"Distribución: {result['topic_distribution']}")
```

### 5.2 Análisis de Logs con Topics

```python
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List
from collections import defaultdict

class LogTopicAnalyzer:
    """
    Analiza logs de seguridad para detectar patrones/temas.

    Útil para:
    - Agrupar eventos similares
    - Detectar comportamientos anómalos
    - Reducir ruido en alertas
    """

    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics

        # NMF es bueno para logs (non-negative)
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            token_pattern=r'[a-zA-Z0-9_]+',
            stop_words=['the', 'a', 'an', 'and', 'or', 'to', 'from']
        )

        self.nmf = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=200
        )

    def fit(self, logs: List[str]) -> 'LogTopicAnalyzer':
        """Entrena con logs."""
        self.tfidf = self.vectorizer.fit_transform(logs)
        self.doc_topics = self.nmf.fit_transform(self.tfidf)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def get_topics(self, n_words: int = 10) -> List[List[str]]:
        """Retorna palabras por tema."""
        topics = []
        for topic_idx in range(self.n_topics):
            top_indices = self.nmf.components_[topic_idx].argsort()[:-n_words-1:-1]
            words = [self.feature_names[i] for i in top_indices]
            topics.append(words)
        return topics

    def assign_topic(self, log: str) -> int:
        """Asigna tema a un log."""
        tfidf = self.vectorizer.transform([log])
        topic_dist = self.nmf.transform(tfidf)
        return topic_dist.argmax()

    def group_logs_by_topic(self, logs: List[str]) -> dict:
        """Agrupa logs por tema dominante."""
        groups = defaultdict(list)

        for log in logs:
            topic = self.assign_topic(log)
            groups[topic].append(log)

        return dict(groups)

    def detect_anomalies(self, logs: List[str],
                        threshold: float = 0.1) -> List[int]:
        """
        Detecta logs anómalos (bajo score en todos los temas).
        """
        anomalies = []

        for i, log in enumerate(logs):
            tfidf = self.vectorizer.transform([log])
            topic_dist = self.nmf.transform(tfidf)

            # Si ningún tema tiene score alto, es anómalo
            if topic_dist.max() < threshold:
                anomalies.append(i)

        return anomalies


# Ejemplo
logs = [
    "Failed password for root from 192.168.1.100 port 22",
    "Failed password for admin from 192.168.1.100 port 22",
    "Failed password for test from 192.168.1.100 port 22",
    "Accepted publickey for user1 from 10.0.0.1 port 22",
    "Accepted password for user2 from 10.0.0.2 port 22",
    "Connection closed by 192.168.1.100 port 22",
    "HTTP GET /admin/config.php 403 Forbidden",
    "HTTP POST /login.php 200 OK",
    "SQL syntax error near 'OR 1=1'",
    "Firewall DROP from 203.0.113.50 to port 445",
    "Firewall DROP from 203.0.113.50 to port 3389",
]

analyzer = LogTopicAnalyzer(n_topics=4)
analyzer.fit(logs)

print("=== TEMAS EN LOGS ===\n")
for i, words in enumerate(analyzer.get_topics(5)):
    print(f"Topic {i}: {', '.join(words)}")

print("\n=== LOGS AGRUPADOS ===\n")
groups = analyzer.group_logs_by_topic(logs)
for topic, topic_logs in groups.items():
    print(f"Topic {topic}:")
    for log in topic_logs[:2]:
        print(f"  - {log}")
```

---

## 6. Evaluación y Mejores Prácticas

### 6.1 Métricas de Evaluación

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MÉTRICAS PARA TOPIC MODELING                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  COHERENCIA:                                                                 │
│    Mide qué tan bien las palabras de un tema "van juntas"                   │
│                                                                              │
│    • C_V: Basada en sliding window y similitud de palabra                   │
│    • C_UCI: Basada en PMI (pointwise mutual information)                    │
│    • C_NPMI: PMI normalizado                                                │
│    • U_MASS: Basada en co-ocurrencia en documentos                          │
│                                                                              │
│    Valores típicos: 0.3-0.5 (bueno), >0.5 (excelente)                       │
│                                                                              │
│  PERPLEJIDAD:                                                                │
│    Mide qué tan bien el modelo predice held-out data                        │
│    Menor = mejor, pero no siempre correlaciona con calidad                  │
│                                                                              │
│  EVALUACIÓN HUMANA:                                                          │
│    • Word intrusion: ¿Puedes identificar la palabra que no pertenece?       │
│    • Topic intrusion: ¿Puedes identificar el tema que no corresponde?       │
│    Más costoso pero más confiable                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Mejores Prácticas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MEJORES PRÁCTICAS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PREPROCESAMIENTO:                                                           │
│    ✓ Eliminar stop words                                                    │
│    ✓ Lemmatización (opcional pero ayuda)                                    │
│    ✓ Filtrar palabras muy frecuentes (>90% docs)                           │
│    ✓ Filtrar palabras muy raras (<2-5 docs)                                │
│    ✓ Usar bigramas/trigramas para capturar frases                          │
│                                                                              │
│  SELECCIÓN DE K (número de temas):                                          │
│    ✓ Probar rango de valores (5-50)                                        │
│    ✓ Usar coherencia para seleccionar                                       │
│    ✓ Validar manualmente que temas tengan sentido                          │
│    ✓ Considerar interpretabilidad vs granularidad                          │
│                                                                              │
│  INTERPRETACIÓN:                                                             │
│    ✓ Nombrar temas basándose en palabras top                                │
│    ✓ Revisar documentos representativos de cada tema                        │
│    ✓ Visualizar (pyLDAvis, BERTopic)                                        │
│    ✓ Iterar: ajustar preprocessing si temas no son claros                  │
│                                                                              │
│  EN PRODUCCIÓN:                                                              │
│    ✓ Monitorear distribución de temas en nuevos documentos                 │
│    ✓ Re-entrenar periódicamente si corpus cambia                           │
│    ✓ Considerar modelos online (incremental updates)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TOPIC MODELING - RESUMEN                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CONCEPTO:                                                                   │
│    Descubrir temas ocultos en colección de documentos (no supervisado)      │
│                                                                              │
│  MÉTODOS:                                                                    │
│    • LSA: SVD sobre TF-IDF, rápido, menos interpretable                     │
│    • LDA: Modelo probabilístico, distribuciones, más interpretable          │
│    • NMF: Similar a LSA pero non-negative                                   │
│    • BERTopic: BERT + clustering, moderno, mejor calidad                    │
│                                                                              │
│  CUÁNDO USAR CADA UNO:                                                       │
│    • LSA: Reducción de dimensionalidad rápida                               │
│    • LDA: Topic modeling clásico, documentos largos                         │
│    • NMF: Cuando necesitas interpretabilidad con non-negative               │
│    • BERTopic: Mejores resultados, textos cortos, no necesitas K            │
│                                                                              │
│  EN SEGURIDAD:                                                               │
│    • Categorización automática de CVEs                                      │
│    • Análisis de reportes de threat intelligence                            │
│    • Agrupación de logs/eventos similares                                   │
│    • Detección de campañas de phishing                                      │
│                                                                              │
│  EVALUACIÓN:                                                                 │
│    • Coherencia (C_V, NPMI): objetivo                                       │
│    • Evaluación humana: subjetivo pero importante                           │
│    • Perplejidad: menos confiable                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
