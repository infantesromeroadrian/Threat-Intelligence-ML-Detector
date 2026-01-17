# Naive Bayes

## 1. ¿Qué es Naive Bayes?

### Concepto Básico

```
┌────────────────────────────────────────────────────────────────┐
│  NAIVE BAYES = Clasificador basado en el Teorema de Bayes      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  PREGUNTA: Dado un email con ciertas palabras,                 │
│            ¿cuál es la PROBABILIDAD de que sea SPAM?           │
│                                                                │
│  Email contiene: "urgente", "premio", "gratis"                 │
│                                                                │
│       P(SPAM | email) = ?                                      │
│                                                                │
│  Naive Bayes calcula esta probabilidad usando                  │
│  las probabilidades de cada palabra en emails SPAM vs HAM      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### ¿Por qué "Naive" (Ingenuo)?

```
┌────────────────────────────────────────────────────────────────┐
│  LA SUPOSICIÓN "NAIVE"                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ASUME: Las features son INDEPENDIENTES entre sí               │
│                                                                │
│  Ejemplo con email:                                            │
│                                                                │
│    P("urgente", "premio" | SPAM)                               │
│        = P("urgente" | SPAM) × P("premio" | SPAM)              │
│                                                                │
│  EN REALIDAD: "urgente" y "premio" están CORRELACIONADAS       │
│               en emails de spam (suelen aparecer juntas)       │
│                                                                │
│  ¿Por qué funciona aunque la suposición sea falsa?             │
│    • La clasificación depende del RANKING, no del valor exacto │
│    • Los errores tienden a cancelarse                          │
│    • En la práctica funciona sorprendentemente bien            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 2. Teorema de Bayes

### La Fórmula

```
┌────────────────────────────────────────────────────────────────┐
│  TEOREMA DE BAYES                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│                    P(B|A) · P(A)                               │
│       P(A|B) = ─────────────────                               │
│                      P(B)                                      │
│                                                                │
│  En clasificación:                                             │
│                                                                │
│                      P(X|clase) · P(clase)                     │
│   P(clase|X) = ────────────────────────────                    │
│                          P(X)                                  │
│                                                                │
│  Donde:                                                        │
│    P(clase|X)  = Posterior (lo que queremos)                   │
│    P(X|clase)  = Likelihood (probabilidad de ver X en clase)   │
│    P(clase)    = Prior (probabilidad a priori de la clase)     │
│    P(X)        = Evidence (constante de normalización)         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Ejemplo Intuitivo

```
PROBLEMA: ¿Este email es SPAM?

Email: "Has ganado un premio urgente"

DATOS HISTÓRICOS:
┌─────────────────────────────────────────────────────────────┐
│  Palabra     │ P(palabra|SPAM) │ P(palabra|HAM)            │
├──────────────┼─────────────────┼─────────────────────────── │
│  "ganado"    │      0.15       │      0.02                 │
│  "premio"    │      0.20       │      0.01                 │
│  "urgente"   │      0.25       │      0.05                 │
└──────────────┴─────────────────┴───────────────────────────┘

P(SPAM) = 0.30  (30% del correo histórico era spam)
P(HAM)  = 0.70

CÁLCULO (simplificado, asumiendo independencia):

P(SPAM|email) ∝ P("ganado"|SPAM) × P("premio"|SPAM) × P("urgente"|SPAM) × P(SPAM)
              = 0.15 × 0.20 × 0.25 × 0.30
              = 0.00225

P(HAM|email) ∝ P("ganado"|HAM) × P("premio"|HAM) × P("urgente"|HAM) × P(HAM)
             = 0.02 × 0.01 × 0.05 × 0.70
             = 0.000007

P(SPAM|email) >> P(HAM|email)  →  CLASIFICAR COMO SPAM ✓
```

## 3. Variantes de Naive Bayes

### Tabla Comparativa

```
┌─────────────────────┬──────────────────────────────────────────┐
│  Variante           │  Uso y características                   │
├─────────────────────┼──────────────────────────────────────────┤
│                     │                                          │
│  Gaussian NB        │  Features continuas (numéricas)          │
│                     │  Asume distribución normal               │
│                     │  Ej: altura, peso, temperatura           │
│                     │                                          │
├─────────────────────┼──────────────────────────────────────────┤
│                     │                                          │
│  Multinomial NB     │  Conteos discretos                       │
│                     │  Ideal para TEXTO (frecuencia palabras)  │
│                     │  Ej: clasificación de documentos, spam   │
│                     │                                          │
├─────────────────────┼──────────────────────────────────────────┤
│                     │                                          │
│  Bernoulli NB       │  Features binarias (0/1)                 │
│                     │  Presencia/ausencia de feature           │
│                     │  Ej: ¿contiene palabra X? Sí/No          │
│                     │                                          │
├─────────────────────┼──────────────────────────────────────────┤
│                     │                                          │
│  Complement NB      │  Para clases desbalanceadas              │
│                     │  Mejora Multinomial en casos extremos    │
│                     │                                          │
└─────────────────────┴──────────────────────────────────────────┘
```

### Gaussian Naive Bayes

```
┌────────────────────────────────────────────────────────────────┐
│  GAUSSIAN NAIVE BAYES                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Asume que cada feature sigue una distribución NORMAL          │
│  (campana de Gauss) dentro de cada clase                       │
│                                                                │
│                    1                 (x - μ)²                  │
│  P(x|clase) = ─────────── · exp( - ───────── )                │
│               √(2πσ²)                 2σ²                      │
│                                                                │
│  Para cada clase, calculamos:                                  │
│    μ = media de la feature                                     │
│    σ = desviación estándar                                     │
│                                                                │
│  Feature: "bytes_enviados"                                     │
│                                                                │
│      Tráfico Normal          Tráfico Malicioso                │
│           ╭─╮                      ╭─╮                         │
│          ╱   ╲                    ╱   ╲                        │
│         ╱     ╲                  ╱     ╲                       │
│        ╱       ╲                ╱       ╲                      │
│    ───┴─────────┴───        ───┴─────────┴───                  │
│        μ=1000                    μ=50000                       │
│        σ=200                     σ=10000                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Multinomial Naive Bayes

```
┌────────────────────────────────────────────────────────────────┐
│  MULTINOMIAL NAIVE BAYES                                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Ideal para CLASIFICACIÓN DE TEXTO                             │
│  Trabaja con CONTEOS de palabras                               │
│                                                                │
│  Email 1: "urgente premio gratis premio"                       │
│                                                                │
│  Vector de conteo:                                             │
│  ┌──────────┬──────────┬────────┬─────────┬─────────┐         │
│  │ urgente  │  premio  │ gratis │  hola   │  reunion│         │
│  ├──────────┼──────────┼────────┼─────────┼─────────┤         │
│  │    1     │    2     │   1    │    0    │    0    │         │
│  └──────────┴──────────┴────────┴─────────┴─────────┘         │
│                                                                │
│                    (n_palabra + α)                             │
│  P(palabra|clase) = ────────────────────                       │
│                     (N_clase + α × |V|)                        │
│                                                                │
│  Donde:                                                        │
│    n_palabra = conteo de la palabra en documentos de la clase │
│    N_clase = total de palabras en documentos de la clase       │
│    α = suavizado Laplace (evita probabilidad 0)                │
│    |V| = tamaño del vocabulario                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 4. Suavizado de Laplace

### El Problema

```
┌────────────────────────────────────────────────────────────────┐
│  PROBLEMA: Palabras nunca vistas                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Palabra "bitcoin" nunca apareció en emails HAM del training   │
│                                                                │
│  P("bitcoin" | HAM) = 0                                        │
│                                                                │
│  DESASTRE:                                                     │
│    P(HAM | email_con_bitcoin)                                  │
│    = ... × P("bitcoin"|HAM) × ...                              │
│    = ... × 0 × ...                                             │
│    = 0                                                         │
│                                                                │
│  ¡Una sola palabra ANULA toda la probabilidad!                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### La Solución

```
┌────────────────────────────────────────────────────────────────┐
│  SUAVIZADO DE LAPLACE (Additive Smoothing)                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IDEA: Añadir α (usualmente 1) a todos los conteos             │
│                                                                │
│  Sin suavizado:                                                │
│                     conteo(palabra, clase)                     │
│  P(palabra|clase) = ──────────────────────                     │
│                     conteo_total(clase)                        │
│                                                                │
│  Con suavizado (α = 1):                                        │
│                     conteo(palabra, clase) + 1                 │
│  P(palabra|clase) = ─────────────────────────────              │
│                     conteo_total(clase) + |vocabulario|        │
│                                                                │
│  Ejemplo:                                                      │
│    "bitcoin" en HAM: 0 apariciones, vocabulario=10000          │
│    Sin suavizado: P = 0/1000 = 0                               │
│    Con suavizado: P = (0+1)/(1000+10000) = 0.00009             │
│                                                                │
│  Ya no es cero, pero sigue siendo muy pequeño (correcto)       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 5. Implementación en Python

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Datos numéricos (features continuas)
np.random.seed(42)
# Clase 0: valores bajos, Clase 1: valores altos
X = np.vstack([
    np.random.normal(0, 1, (500, 4)),    # Clase 0
    np.random.normal(3, 1.5, (500, 4))   # Clase 1
])
y = np.array([0]*500 + [1]*500)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar Gaussian NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Evaluar
print(f"Accuracy: {gnb.score(X_test, y_test):.3f}")
print(classification_report(y_test, gnb.predict(X_test)))

# Ver parámetros aprendidos
print("\nParámetros por clase:")
for i, clase in enumerate(gnb.classes_):
    print(f"Clase {clase}:")
    print(f"  Medias:      {gnb.theta_[i]}")
    print(f"  Varianzas:   {gnb.var_[i]}")
```

### Multinomial Naive Bayes (para Texto)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Datos de texto
emails = [
    "urgente ganaste premio gratis click aqui",
    "reunion mañana oficina proyecto",
    "felicidades ganador premio especial",
    "informe trimestral adjunto revisar",
    "oferta limitada descuento exclusivo",
    "conferencia viernes sala reuniones",
    "gratis regalo promocion especial ahora",
    "presupuesto aprobado siguiente fase",
    # ... más ejemplos
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=SPAM, 0=HAM

# Vectorizar texto
vectorizer = CountVectorizer()  # o TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# Entrenar Multinomial NB
mnb = MultinomialNB(alpha=1.0)  # alpha = suavizado Laplace
mnb.fit(X_train, y_train)

# Evaluar
print(f"Accuracy: {mnb.score(X_test, y_test):.3f}")

# Ver palabras más indicativas de cada clase
feature_names = vectorizer.get_feature_names_out()
for i, clase in enumerate(['HAM', 'SPAM']):
    top10_idx = mnb.feature_log_prob_[i].argsort()[-10:][::-1]
    top_words = [feature_names[j] for j in top10_idx]
    print(f"\nTop palabras para {clase}: {', '.join(top_words)}")
```

### Pipeline Completo para Clasificación de Texto

```python
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

# Pipeline: Vectorización + Clasificación
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)  # unigrams y bigrams
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

# Cross-validation
scores = cross_val_score(pipeline, emails, labels, cv=5, scoring='f1')
print(f"F1 Score (CV): {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Entrenar modelo final
pipeline.fit(emails, labels)

# Predecir nuevos emails
nuevos_emails = [
    "felicidades has ganado un premio increible",
    "la reunion de mañana se cancela"
]
predicciones = pipeline.predict(nuevos_emails)
probabilidades = pipeline.predict_proba(nuevos_emails)

for email, pred, prob in zip(nuevos_emails, predicciones, probabilidades):
    clase = "SPAM" if pred == 1 else "HAM"
    confianza = max(prob)
    print(f"'{email[:40]}...' → {clase} ({confianza:.2%})")
```

## 6. Hiperparámetros

```
┌─────────────────┬─────────────────┬────────────────────────────┐
│  Parámetro      │  Default        │  Descripción               │
├─────────────────┼─────────────────┼────────────────────────────┤
│                 │                 │                            │
│  alpha          │  1.0            │  Suavizado Laplace         │
│  (Multinomial/  │                 │  0 = sin suavizado         │
│   Bernoulli)    │                 │  1 = suavizado estándar    │
│                 │                 │  <1 = menos suavizado      │
│                 │                 │                            │
├─────────────────┼─────────────────┼────────────────────────────┤
│                 │                 │                            │
│  fit_prior      │  True           │  Aprender P(clase)         │
│                 │                 │  False = clases uniformes  │
│                 │                 │                            │
├─────────────────┼─────────────────┼────────────────────────────┤
│                 │                 │                            │
│  class_prior    │  None           │  Probabilidades a priori   │
│                 │                 │  personalizadas            │
│                 │                 │                            │
├─────────────────┼─────────────────┼────────────────────────────┤
│                 │                 │                            │
│  var_smoothing  │  1e-9           │  Para GaussianNB           │
│  (Gaussian)     │                 │  Estabilidad numérica      │
│                 │                 │                            │
└─────────────────┴─────────────────┴────────────────────────────┘
```

## 7. Ejemplo Práctico: Detector de Spam

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re

# Simular dataset de emails
np.random.seed(42)

# Palabras típicas de spam y ham
spam_words = ['urgente', 'premio', 'ganador', 'gratis', 'oferta',
              'descuento', 'click', 'promocion', 'exclusivo', 'ahora',
              'dinero', 'millones', 'felicidades', 'increible', 'limitado']

ham_words = ['reunion', 'proyecto', 'informe', 'cliente', 'oficina',
             'presupuesto', 'equipo', 'revisar', 'adjunto', 'saludos',
             'agenda', 'confirmar', 'disponibilidad', 'propuesta', 'gracias']

def generar_email(es_spam, n_palabras=15):
    """Genera un email sintético"""
    if es_spam:
        palabras = np.random.choice(spam_words, n_palabras, replace=True)
        # Añadir algunas palabras ham para realismo
        palabras = np.append(palabras, np.random.choice(ham_words, 3))
    else:
        palabras = np.random.choice(ham_words, n_palabras, replace=True)
        # Añadir algunas palabras spam ocasionales
        if np.random.random() < 0.1:
            palabras = np.append(palabras, np.random.choice(spam_words, 1))
    np.random.shuffle(palabras)
    return ' '.join(palabras)

# Generar dataset
n_spam = 500
n_ham = 1500  # Desbalanceado como en la realidad

emails = []
labels = []

for _ in range(n_spam):
    emails.append(generar_email(True))
    labels.append(1)

for _ in range(n_ham):
    emails.append(generar_email(False))
    labels.append(0)

emails = np.array(emails)
labels = np.array(labels)

# Shuffle
idx = np.random.permutation(len(emails))
emails, labels = emails[idx], labels[idx]

print(f"Total emails: {len(emails)}")
print(f"SPAM: {sum(labels)} ({sum(labels)/len(labels):.1%})")
print(f"HAM: {len(labels)-sum(labels)} ({1-sum(labels)/len(labels):.1%})")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.2, random_state=42, stratify=labels
)

# Vectorizar
tfidf = TfidfVectorizer(max_features=100)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Entrenar Naive Bayes
nb = MultinomialNB(alpha=0.5)
nb.fit(X_train_tfidf, y_train)

# Evaluar
y_pred = nb.predict(X_test_tfidf)
y_proba = nb.predict_proba(X_test_tfidf)

print("\n" + "=" * 60)
print("RESULTADOS DEL DETECTOR DE SPAM")
print("=" * 60)
print(f"\nAccuracy: {nb.score(X_test_tfidf, y_test):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Analizar palabras más indicativas
print("\n" + "=" * 60)
print("PALABRAS MÁS INDICATIVAS")
print("=" * 60)

feature_names = tfidf.get_feature_names_out()
log_probs_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]

# Más indicativas de SPAM
spam_indicators = np.argsort(log_probs_diff)[-10:][::-1]
print("\nTop palabras SPAM:")
for idx in spam_indicators:
    print(f"  {feature_names[idx]:15} (log-ratio: {log_probs_diff[idx]:.2f})")

# Más indicativas de HAM
ham_indicators = np.argsort(log_probs_diff)[:10]
print("\nTop palabras HAM:")
for idx in ham_indicators:
    print(f"  {feature_names[idx]:15} (log-ratio: {log_probs_diff[idx]:.2f})")

# Probar con emails nuevos
print("\n" + "=" * 60)
print("CLASIFICACIÓN DE NUEVOS EMAILS")
print("=" * 60)

nuevos_emails = [
    "urgente premio exclusivo ganador felicidades",
    "reunion proyecto equipo revisar informe",
    "oferta gratis click ahora promocion descuento",
    "propuesta cliente presupuesto confirmar"
]

for email in nuevos_emails:
    X_nuevo = tfidf.transform([email])
    pred = nb.predict(X_nuevo)[0]
    proba = nb.predict_proba(X_nuevo)[0]

    clase = "SPAM" if pred == 1 else "HAM"
    confianza = proba[pred]

    print(f"\n'{email}'")
    print(f"  → {clase} (confianza: {confianza:.1%})")
    print(f"  → Probabilidades: HAM={proba[0]:.1%}, SPAM={proba[1]:.1%}")
```

## 8. Naive Bayes en Ciberseguridad

### Casos de Uso

```
┌────────────────────────────────────────────────────────────────┐
│  APLICACIONES EN CIBERSEGURIDAD                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. DETECCIÓN DE SPAM/PHISHING                                 │
│     • Clasificar emails por contenido                          │
│     • Identificar URLs maliciosas                              │
│     • Filtrar mensajes sospechosos                             │
│                                                                │
│  2. CLASIFICACIÓN DE MALWARE                                   │
│     • Análisis de strings en binarios                          │
│     • Categorización por familia de malware                    │
│     • Detección de scripts maliciosos                          │
│                                                                │
│  3. DETECCIÓN DE INTRUSIONES (IDS)                             │
│     • Clasificar paquetes de red                               │
│     • Identificar patrones de ataque                           │
│     • Análisis de logs de sistema                              │
│                                                                │
│  4. ANÁLISIS DE AMENAZAS                                       │
│     • Clasificar reportes de incidentes                        │
│     • Categorizar IOCs (Indicators of Compromise)              │
│     • Priorizar alertas de seguridad                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Ejemplo: Clasificador de Logs de Seguridad

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Simular logs de seguridad
np.random.seed(42)

# Patrones de logs por categoría
log_patterns = {
    'brute_force': [
        "failed login attempt user {} from {}",
        "authentication failure for user {} ip {}",
        "invalid password for {} from port {}",
        "maximum login attempts exceeded user {}",
        "account locked after failed attempts {}"
    ],
    'sql_injection': [
        "SQL syntax error near '{}' in query",
        "invalid SQL statement: {} OR 1=1",
        "detected UNION SELECT in parameter {}",
        "possible SQL injection in field {}",
        "blocked query with suspicious pattern {}"
    ],
    'normal': [
        "user {} logged in successfully",
        "session {} started for user {}",
        "file {} accessed by user {}",
        "user {} logged out",
        "connection established from {}"
    ]
}

def generar_log(categoria, n_logs):
    logs = []
    patterns = log_patterns[categoria]
    for _ in range(n_logs):
        pattern = np.random.choice(patterns)
        # Rellenar placeholders con datos aleatorios
        filled = pattern.format(
            f"user{np.random.randint(100)}",
            f"192.168.{np.random.randint(255)}.{np.random.randint(255)}"
        )
        logs.append(filled)
    return logs

# Generar dataset
logs = []
labels = []

for cat_id, categoria in enumerate(['normal', 'brute_force', 'sql_injection']):
    cat_logs = generar_log(categoria, 300)
    logs.extend(cat_logs)
    labels.extend([cat_id] * len(cat_logs))

logs = np.array(logs)
labels = np.array(labels)

# Shuffle
idx = np.random.permutation(len(logs))
logs, labels = logs[idx], labels[idx]

print("Distribución de logs:")
for i, cat in enumerate(['normal', 'brute_force', 'sql_injection']):
    print(f"  {cat}: {sum(labels == i)}")

# Split y entrenar
X_train, X_test, y_train, y_test = train_test_split(
    logs, labels, test_size=0.2, random_state=42, stratify=labels
)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=200)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_tfidf, y_train)

print("\n" + "=" * 60)
print("CLASIFICADOR DE LOGS DE SEGURIDAD")
print("=" * 60)
print(classification_report(
    y_test, nb.predict(X_test_tfidf),
    target_names=['normal', 'brute_force', 'sql_injection']
))

# Clasificar nuevos logs
print("\nClasificación de nuevos logs:")
nuevos_logs = [
    "failed login attempt user admin from 192.168.1.100",
    "detected UNION SELECT in parameter id",
    "user john logged in successfully"
]

for log in nuevos_logs:
    X_nuevo = tfidf.transform([log])
    pred = nb.predict(X_nuevo)[0]
    proba = nb.predict_proba(X_nuevo)[0]

    categorias = ['normal', 'brute_force', 'sql_injection']
    print(f"\n'{log}'")
    print(f"  → {categorias[pred].upper()} ({proba[pred]:.1%})")
```

## 9. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────────────┐
│  VENTAJAS DE NAIVE BAYES                                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ MUY RÁPIDO (entrenamiento y predicción)                     │
│  ✓ Funciona bien con pocos datos de entrenamiento              │
│  ✓ No requiere mucho tuning de hiperparámetros                 │
│  ✓ Escalable a datasets grandes                                │
│  ✓ Maneja bien alta dimensionalidad (texto)                    │
│  ✓ Proporciona probabilidades calibradas                       │
│  ✓ Robusto contra features irrelevantes                        │
│  ✓ Actualizable incrementalmente                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DE NAIVE BAYES                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Asunción de independencia casi siempre es FALSA             │
│  ✗ No captura interacciones entre features                     │
│  ✗ Probabilidades pueden estar mal calibradas                  │
│  ✗ Zero-frequency problem (requiere suavizado)                 │
│  ✗ Sensible a features con diferentes escalas (Gaussian)       │
│  ✗ No funciona bien con datos muy correlacionados              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 10. Naive Bayes vs Otros Clasificadores

```
┌─────────────────────┬───────────────────────────────────────────┐
│  Aspecto            │  Naive Bayes                              │
├─────────────────────┼───────────────────────────────────────────┤
│ Velocidad Train     │  ★★★★★  Muy rápido (una pasada)          │
├─────────────────────┼───────────────────────────────────────────┤
│ Velocidad Predict   │  ★★★★★  Muy rápido                       │
├─────────────────────┼───────────────────────────────────────────┤
│ Accuracy            │  ★★★☆☆  Bueno, pero no el mejor          │
├─────────────────────┼───────────────────────────────────────────┤
│ Con pocos datos     │  ★★★★★  Excelente                        │
├─────────────────────┼───────────────────────────────────────────┤
│ Alta dimensión      │  ★★★★★  Excelente (texto)                │
├─────────────────────┼───────────────────────────────────────────┤
│ Interpretabilidad   │  ★★★★☆  Bueno (log-probabilities)        │
├─────────────────────┼───────────────────────────────────────────┤
│ Tuning requerido    │  ★★★★★  Mínimo (solo alpha)              │
└─────────────────────┴───────────────────────────────────────────┘

COMPARACIÓN PARA CLASIFICACIÓN DE TEXTO:

           │  Accuracy  │ Velocidad │ Interpretable │  Pocos datos
───────────┼────────────┼───────────┼───────────────┼─────────────
Naive Bayes│   Bueno    │ Muy Alta  │     Alta      │  Excelente
───────────┼────────────┼───────────┼───────────────┼─────────────
Logistic   │   Bueno    │   Alta    │     Alta      │    Bueno
───────────┼────────────┼───────────┼───────────────┼─────────────
SVM        │   Alto     │   Media   │     Baja      │    Bueno
───────────┼────────────┼───────────┼───────────────┼─────────────
Random     │   Alto     │   Media   │     Media     │    Medio
Forest     │            │           │               │
───────────┼────────────┼───────────┼───────────────┼─────────────
Deep       │  Muy Alto  │   Baja    │     Baja      │    Malo
Learning   │            │           │               │
```

## 11. Cuándo Usar Naive Bayes

```
┌────────────────────────────────────────────────────────────────┐
│  CASOS DE USO IDEALES                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Clasificación de texto (spam, sentimiento, categorías)      │
│  ✓ Sistemas en tiempo real (predicción muy rápida)             │
│  ✓ Como baseline antes de probar modelos más complejos         │
│  ✓ Cuando tienes POCOS datos de entrenamiento                  │
│  ✓ Filtrado de emails/mensajes                                 │
│  ✓ Sistemas de recomendación simples                           │
│  ✓ Clasificación de documentos                                 │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  EVITAR CUANDO                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Features están muy correlacionadas                          │
│  ✗ Necesitas el máximo accuracy posible                        │
│  ✗ Las interacciones entre features son importantes            │
│  ✗ Datos numéricos que no siguen distribución normal           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 12. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  NAIVE BAYES - RESUMEN                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CONCEPTO:                                                     │
│    Clasificador probabilístico basado en Teorema de Bayes      │
│    Asume independencia entre features (naive = ingenuo)        │
│                                                                │
│  VARIANTES:                                                    │
│    • GaussianNB: features continuas (distribución normal)      │
│    • MultinomialNB: conteos (ideal para texto)                 │
│    • BernoulliNB: features binarias (presencia/ausencia)       │
│                                                                │
│  FÓRMULA:                                                      │
│    P(clase|X) ∝ P(X|clase) × P(clase)                          │
│    P(X|clase) = Π P(xᵢ|clase)  [independencia]                │
│                                                                │
│  HIPERPARÁMETRO CLAVE:                                         │
│    alpha: suavizado Laplace (evita probabilidad 0)             │
│                                                                │
│  FORTALEZAS:                                                   │
│    • Muy rápido                                                │
│    • Funciona con pocos datos                                  │
│    • Excelente para texto                                      │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Detección de spam/phishing                                │
│    • Clasificación de logs                                     │
│    • Análisis de malware                                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Gradient Boosting y XGBoost
