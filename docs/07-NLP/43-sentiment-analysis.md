# Sentiment Analysis (Análisis de Sentimiento)

## 1. ¿Qué es Sentiment Analysis?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SENTIMENT ANALYSIS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  OBJETIVO: Determinar la actitud, opinión o emoción expresada en un texto  │
│                                                                              │
│  TIPOS DE ANÁLISIS:                                                          │
│                                                                              │
│  1. POLARIDAD (más común):                                                  │
│     Positivo / Negativo / Neutro                                            │
│     "This product is amazing!" → Positivo                                   │
│     "Terrible service, never again" → Negativo                              │
│     "The package arrived today" → Neutro                                    │
│                                                                              │
│  2. ESCALA DE INTENSIDAD:                                                   │
│     Muy negativo / Negativo / Neutro / Positivo / Muy positivo             │
│     1 ★ ★ ★ ★ ★ 5                                                          │
│                                                                              │
│  3. DETECCIÓN DE EMOCIONES:                                                 │
│     Alegría, tristeza, ira, miedo, sorpresa, disgusto                       │
│     "I can't believe they did this!" → Ira/Sorpresa                        │
│                                                                              │
│  4. ANÁLISIS DE ASPECTOS:                                                   │
│     Sentimiento por característica específica                               │
│     "Great camera, but terrible battery life"                               │
│     → Camera: Positivo, Battery: Negativo                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Aplicaciones

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        APLICACIONES                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GENERAL:                                                                    │
│    • Monitoreo de marca/reputación                                          │
│    • Análisis de reviews de productos                                       │
│    • Social media monitoring                                                 │
│    • Customer service automation                                            │
│    • Market research                                                         │
│                                                                              │
│  EN SEGURIDAD:                                                               │
│    • Detección de contenido tóxico/hate speech                              │
│    • Análisis de comunicaciones sospechosas                                 │
│    • Detección de insider threats (empleados descontentos)                  │
│    • Monitoreo de dark web/foros                                            │
│    • Análisis de phishing (urgencia, miedo)                                 │
│    • Threat intelligence (sentimiento hacia targets)                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Enfoques para Sentiment Analysis

### 2.1 Enfoque Basado en Léxico

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LEXICON-BASED APPROACH                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  IDEA: Usar diccionarios de palabras con puntuaciones de sentimiento       │
│                                                                              │
│  LÉXICOS POPULARES:                                                          │
│    • VADER: Específico para social media, maneja emojis                     │
│    • SentiWordNet: Basado en WordNet                                        │
│    • AFINN: Puntuaciones de -5 a +5                                         │
│    • TextBlob: Simple y fácil de usar                                       │
│                                                                              │
│  PROCESO:                                                                    │
│    1. Tokenizar texto                                                        │
│    2. Buscar cada palabra en el léxico                                      │
│    3. Agregar puntuaciones (suma, promedio)                                 │
│    4. Considerar modificadores (negación, intensificadores)                 │
│                                                                              │
│  EJEMPLO:                                                                    │
│    "This is not a good product"                                             │
│    good = +3                                                                │
│    not (negación) → invierte → -3                                          │
│    Score total: -3 → Negativo                                               │
│                                                                              │
│  VENTAJAS:                           DESVENTAJAS:                            │
│    ✓ No requiere entrenamiento       ✗ No captura contexto                  │
│    ✓ Interpretable                   ✗ Vocabulario limitado                 │
│    ✓ Rápido                          ✗ No maneja sarcasmo/ironía            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementación con VADER

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List

class VADERSentiment:
    """
    Análisis de sentimiento con VADER.
    Especialmente bueno para texto informal, social media, emojis.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict:
        """
        Analiza sentimiento de un texto.

        Returns:
            Dict con scores:
            - neg: Score negativo (0-1)
            - neu: Score neutro (0-1)
            - pos: Score positivo (0-1)
            - compound: Score compuesto (-1 a 1)
        """
        return self.analyzer.polarity_scores(text)

    def classify(self, text: str, threshold: float = 0.05) -> str:
        """
        Clasifica texto en positivo/negativo/neutro.

        Args:
            threshold: Umbral para compound score
        """
        scores = self.analyze(text)
        compound = scores['compound']

        if compound >= threshold:
            return 'positive'
        elif compound <= -threshold:
            return 'negative'
        else:
            return 'neutral'

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analiza múltiples textos."""
        return [self.analyze(text) for text in texts]


# Ejemplo
analyzer = VADERSentiment()

texts = [
    "This product is absolutely amazing! Best purchase ever! 😍",
    "Terrible service. I'm so disappointed and angry.",
    "The package arrived today.",
    "I love this! 💕 But the price is a bit high...",
    "Not bad, not great. It's okay I guess.",
    "WORST EXPERIENCE EVER!!! Never buying again! 😡",
]

print("=== ANÁLISIS DE SENTIMIENTO (VADER) ===\n")
for text in texts:
    scores = analyzer.analyze(text)
    classification = analyzer.classify(text)
    print(f"Text: {text[:50]}...")
    print(f"  Scores: pos={scores['pos']:.2f}, neg={scores['neg']:.2f}, "
          f"neu={scores['neu']:.2f}, compound={scores['compound']:.2f}")
    print(f"  Classification: {classification}\n")
```

### 2.3 Implementación con TextBlob

```python
from textblob import TextBlob
from typing import Tuple

class TextBlobSentiment:
    """
    Análisis de sentimiento con TextBlob.
    Simple y fácil de usar.
    """

    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analiza sentimiento.

        Returns:
            (polarity, subjectivity)
            - polarity: -1 (negativo) a 1 (positivo)
            - subjectivity: 0 (objetivo) a 1 (subjetivo)
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def classify(self, text: str) -> str:
        """Clasifica en positivo/negativo/neutro."""
        polarity, _ = self.analyze(text)

        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'


# Ejemplo
analyzer = TextBlobSentiment()

texts = [
    "I absolutely love this product!",
    "This is the worst service I've ever experienced.",
    "The sky is blue.",
]

for text in texts:
    polarity, subjectivity = analyzer.analyze(text)
    classification = analyzer.classify(text)
    print(f"Text: {text}")
    print(f"  Polarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")
    print(f"  Class: {classification}\n")
```

---

## 3. Machine Learning para Sentiment Analysis

### 3.1 Clasificador con TF-IDF + Logistic Regression

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Tuple

class MLSentimentClassifier:
    """
    Clasificador de sentimiento con Machine Learning tradicional.
    """

    def __init__(self,
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        )

    def fit(self, texts: List[str], labels: List[int]):
        """
        Entrena el clasificador.

        Args:
            texts: Lista de textos
            labels: 0=negativo, 1=positivo (o más clases)
        """
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predice etiquetas."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predice probabilidades."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def get_important_features(self, class_idx: int = 1,
                               top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Retorna features más importantes para una clase.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.classifier.coef_[0] if len(self.classifier.classes_) == 2 \
                else self.classifier.coef_[class_idx]

        # Top positivos (indicadores de la clase)
        top_positive = np.argsort(coefs)[-top_n:][::-1]
        # Top negativos (indicadores de otra clase)
        top_negative = np.argsort(coefs)[:top_n]

        positive_features = [(feature_names[i], coefs[i])
                            for i in top_positive]
        negative_features = [(feature_names[i], coefs[i])
                            for i in top_negative]

        return positive_features, negative_features


# Ejemplo con dataset sintético
texts = [
    # Positivos
    "This is amazing! I love it!",
    "Great product, highly recommend",
    "Excellent quality and fast shipping",
    "Best purchase I've ever made",
    "Fantastic service, very happy",
    "Perfect! Exactly what I needed",
    "Outstanding performance, impressed",
    "Wonderful experience, will buy again",
    # Negativos
    "Terrible product, waste of money",
    "Horrible service, very disappointed",
    "Worst experience ever, avoid",
    "Broken on arrival, terrible quality",
    "Awful, do not recommend",
    "Complete disaster, want refund",
    "Garbage, doesn't work at all",
    "Frustrating experience, useless product",
]

labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# Entrenar
classifier = MLSentimentClassifier()
classifier.fit(texts, labels)

# Predecir
test_texts = [
    "This is really good!",
    "Terrible, don't buy this",
    "It's okay, nothing special",
]

predictions = classifier.predict(test_texts)
probabilities = classifier.predict_proba(test_texts)

print("=== PREDICCIONES ===\n")
for text, pred, probs in zip(test_texts, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    conf = probs[pred]
    print(f"Text: {text}")
    print(f"  Prediction: {sentiment} (confidence: {conf:.2f})\n")

# Features importantes
print("=== FEATURES IMPORTANTES ===\n")
pos_features, neg_features = classifier.get_important_features()
print("Indicadores de POSITIVO:")
for word, score in pos_features[:10]:
    print(f"  {word}: {score:.3f}")
print("\nIndicadores de NEGATIVO:")
for word, score in neg_features[:10]:
    print(f"  {word}: {score:.3f}")
```

### 3.2 Clasificador con BERT

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from typing import List, Dict

class BERTSentimentClassifier:
    """
    Clasificador de sentimiento con BERT pre-entrenado.
    """

    def __init__(self,
                 model_name: str = 'nlptown/bert-base-multilingual-uncased-sentiment'):
        """
        Args:
            model_name: Modelo pre-entrenado para sentiment
                - 'nlptown/bert-base-multilingual-uncased-sentiment': 1-5 stars
                - 'cardiffnlp/twitter-roberta-base-sentiment': pos/neg/neu
                - 'distilbert-base-uncased-finetuned-sst-2-english': pos/neg
        """
        self.model_name = model_name
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Predice sentimiento para lista de textos.
        """
        return self.classifier(texts)

    def predict_single(self, text: str) -> Dict:
        """Predice sentimiento para un texto."""
        return self.classifier(text)[0]


# Ejemplo con modelo de 5 estrellas
classifier = BERTSentimentClassifier()

texts = [
    "This is the best product I have ever bought!",
    "Absolutely terrible, waste of money.",
    "It's okay, does the job.",
    "Pretty good, minor issues but overall satisfied.",
]

print("=== SENTIMENT ANALYSIS (BERT) ===\n")
for text in texts:
    result = classifier.predict_single(text)
    print(f"Text: {text}")
    print(f"  Label: {result['label']}, Score: {result['score']:.4f}\n")
```

### 3.3 Fine-tuning BERT para Sentiment

```python
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def fine_tune_sentiment_bert(
    train_texts: list,
    train_labels: list,
    val_texts: list,
    val_labels: list,
    model_name: str = 'bert-base-uncased',
    num_labels: int = 3,  # neg, neu, pos
    output_dir: str = './sentiment_model',
    epochs: int = 3
):
    """
    Fine-tune BERT para clasificación de sentimiento.
    """
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenizar
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )

    # Crear datasets
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    }).map(tokenize, batched=True)

    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    }).map(tokenize, batched=True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Modelo
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Métricas
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer, tokenizer


# Uso (comentado para no ejecutar)
# trainer, tokenizer = fine_tune_sentiment_bert(
#     train_texts, train_labels,
#     val_texts, val_labels,
#     num_labels=3  # neg=0, neu=1, pos=2
# )
```

---

## 4. Análisis de Emociones

### 4.1 Detección de Emociones Múltiples

```python
from transformers import pipeline
from typing import List, Dict

class EmotionDetector:
    """
    Detecta emociones en texto usando modelos pre-entrenados.

    Emociones típicas:
    - Joy, Sadness, Anger, Fear, Surprise, Disgust
    - O variantes: Love, Optimism, Pessimism, etc.
    """

    def __init__(self,
                 model_name: str = 'bhadresh-savani/distilbert-base-uncased-emotion'):
        """
        Args:
            model_name: Modelo para detección de emociones
                - 'bhadresh-savani/distilbert-base-uncased-emotion'
                - 'j-hartmann/emotion-english-distilroberta-base'
        """
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None  # Retorna todas las emociones
        )

    def detect(self, text: str) -> List[Dict]:
        """Detecta emociones en texto."""
        return self.classifier(text)[0]

    def detect_top_emotion(self, text: str) -> Dict:
        """Retorna emoción dominante."""
        emotions = self.detect(text)
        return max(emotions, key=lambda x: x['score'])

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analiza múltiples textos."""
        results = []
        for text in texts:
            top = self.detect_top_emotion(text)
            results.append({
                'text': text,
                'emotion': top['label'],
                'score': top['score']
            })
        return results


# Ejemplo
detector = EmotionDetector()

texts = [
    "I'm so happy about this wonderful news!",
    "This makes me really angry and frustrated.",
    "I'm terrified of what might happen next.",
    "That's so sad, I can't stop crying.",
    "Wow, I didn't expect that at all!",
    "This is disgusting, I hate it.",
]

print("=== DETECCIÓN DE EMOCIONES ===\n")
for text in texts:
    emotions = detector.detect(text)
    top = detector.detect_top_emotion(text)

    print(f"Text: {text}")
    print(f"  Top emotion: {top['label']} ({top['score']:.2f})")
    print(f"  All emotions: ", end="")
    for e in sorted(emotions, key=lambda x: -x['score'])[:3]:
        print(f"{e['label']}:{e['score']:.2f}", end=" ")
    print("\n")
```

---

## 5. Aspect-Based Sentiment Analysis

```python
from transformers import pipeline
from typing import List, Dict, Tuple
import spacy

class AspectSentimentAnalyzer:
    """
    Análisis de sentimiento por aspecto/característica.

    Ejemplo:
    "The food was great but the service was terrible"
    → food: positive, service: negative
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def extract_aspects(self, text: str) -> List[str]:
        """
        Extrae aspectos (sustantivos) del texto.
        """
        doc = self.nlp(text)
        aspects = []

        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                aspects.append(token.text)

        return list(set(aspects))

    def analyze(self, text: str) -> List[Dict]:
        """
        Analiza sentimiento por aspecto.
        """
        doc = self.nlp(text)
        aspects = self.extract_aspects(text)
        results = []

        for sent in doc.sents:
            sent_text = sent.text

            # Encontrar aspectos en esta oración
            for aspect in aspects:
                if aspect.lower() in sent_text.lower():
                    # Analizar sentimiento de la oración
                    sentiment = self.sentiment(sent_text)[0]
                    results.append({
                        'aspect': aspect,
                        'sentence': sent_text,
                        'sentiment': sentiment['label'],
                        'score': sentiment['score']
                    })

        return results


# Ejemplo
analyzer = AspectSentimentAnalyzer()

reviews = [
    "The camera quality is amazing but the battery life is terrible.",
    "Great price for this laptop. The screen is beautiful. However, the keyboard feels cheap.",
    "The hotel location was perfect. The room was clean but small. Staff was friendly and helpful.",
]

print("=== ASPECT-BASED SENTIMENT ANALYSIS ===\n")
for review in reviews:
    print(f"Review: {review}\n")
    aspects = analyzer.analyze(review)
    for a in aspects:
        print(f"  Aspect: {a['aspect']}")
        print(f"  Sentiment: {a['sentiment']} ({a['score']:.2f})")
        print()
    print("-" * 60 + "\n")
```

---

## 6. Sentiment Analysis en Seguridad

### 6.1 Detección de Contenido Tóxico

```python
from transformers import pipeline
from typing import Dict, List

class ToxicContentDetector:
    """
    Detecta contenido tóxico, ofensivo o hate speech.
    """

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            top_k=None
        )

    def analyze(self, text: str) -> Dict:
        """
        Analiza toxicidad del texto.

        Categorías típicas:
        - toxic
        - severe_toxic
        - obscene
        - threat
        - insult
        - identity_hate
        """
        results = self.classifier(text)[0]
        return {r['label']: r['score'] for r in results}

    def is_toxic(self, text: str, threshold: float = 0.5) -> bool:
        """Determina si el texto es tóxico."""
        scores = self.analyze(text)
        return any(score > threshold for score in scores.values())

    def get_toxic_type(self, text: str,
                       threshold: float = 0.5) -> List[str]:
        """Retorna tipos de toxicidad detectados."""
        scores = self.analyze(text)
        return [label for label, score in scores.items()
                if score > threshold]


# Ejemplo
detector = ToxicContentDetector()

texts = [
    "Have a wonderful day!",
    "You're an idiot and I hate you",
    "This is a normal technical discussion",
    # Más ejemplos según uso
]

print("=== DETECCIÓN DE CONTENIDO TÓXICO ===\n")
for text in texts:
    is_toxic = detector.is_toxic(text)
    toxic_types = detector.get_toxic_type(text)

    print(f"Text: {text}")
    print(f"  Toxic: {is_toxic}")
    if toxic_types:
        print(f"  Types: {toxic_types}")
    print()
```

### 6.2 Análisis de Urgencia en Phishing

```python
from transformers import pipeline
from typing import Dict, List
import re

class PhishingUrgencyDetector:
    """
    Detecta indicadores de urgencia/presión típicos de phishing.

    Phishing usa tácticas de:
    - Urgencia: "Act now!", "Limited time"
    - Miedo: "Your account will be suspended"
    - Autoridad: "From your bank", "Official notice"
    """

    def __init__(self):
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        # Patrones de urgencia
        self.urgency_patterns = [
            r'\b(urgent|immediately|now|asap|hurry)\b',
            r'\b(limited time|act fast|don\'t delay)\b',
            r'\b(within \d+ hours?|expire|deadline)\b',
            r'\b(last chance|final notice|final warning)\b',
        ]

        # Patrones de miedo
        self.fear_patterns = [
            r'\b(suspend|terminate|close|delete|deactivate)\b',
            r'\b(unauthorized|suspicious|unusual) (activity|access)\b',
            r'\b(security (alert|breach|issue)|compromised)\b',
            r'\b(verify|confirm|update).{0,20}(account|information)\b',
        ]

        # Patrones de autoridad
        self.authority_patterns = [
            r'\b(official|security team|support team)\b',
            r'\b(your bank|paypal|amazon|microsoft)\b',
            r'\b(important notice|security update)\b',
        ]

    def analyze(self, text: str) -> Dict:
        """
        Analiza email/texto por indicadores de phishing.
        """
        text_lower = text.lower()

        # Contar patrones
        urgency_score = sum(
            len(re.findall(p, text_lower, re.IGNORECASE))
            for p in self.urgency_patterns
        )

        fear_score = sum(
            len(re.findall(p, text_lower, re.IGNORECASE))
            for p in self.fear_patterns
        )

        authority_score = sum(
            len(re.findall(p, text_lower, re.IGNORECASE))
            for p in self.authority_patterns
        )

        # Sentiment (phishing tiende a ser neutro-negativo)
        sentiment = self.sentiment(text[:512])[0]

        # Score total
        total_score = (
            urgency_score * 2 +  # Urgencia es muy indicativa
            fear_score * 1.5 +
            authority_score
        )

        return {
            'urgency_score': urgency_score,
            'fear_score': fear_score,
            'authority_score': authority_score,
            'sentiment': sentiment,
            'phishing_score': min(total_score / 10, 1.0),
            'is_suspicious': total_score >= 3
        }

    def explain(self, text: str) -> List[str]:
        """Explica por qué el texto es sospechoso."""
        text_lower = text.lower()
        reasons = []

        for pattern in self.urgency_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                reasons.append(f"Urgency indicator: '{matches[0]}'")

        for pattern in self.fear_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                reasons.append(f"Fear tactic: '{matches[0]}'")

        for pattern in self.authority_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                reasons.append(f"Authority claim: '{matches[0]}'")

        return reasons


# Ejemplo
detector = PhishingUrgencyDetector()

emails = [
    """
    Subject: URGENT: Your account will be suspended!

    Dear Customer,

    We detected unauthorized access to your account. Your account will be
    suspended within 24 hours unless you verify your information immediately.

    Click here to verify: http://fake-bank.com/verify

    This is your FINAL NOTICE. Act now!

    Security Team
    """,

    """
    Subject: Meeting tomorrow

    Hi,

    Just a reminder about our meeting tomorrow at 2pm.
    Please bring the quarterly report.

    Thanks,
    John
    """,
]

print("=== ANÁLISIS DE PHISHING ===\n")
for email in emails:
    print(f"Email preview: {email[:100].strip()}...")
    print()

    analysis = detector.analyze(email)
    print(f"  Phishing Score: {analysis['phishing_score']:.2f}")
    print(f"  Suspicious: {analysis['is_suspicious']}")
    print(f"  Urgency: {analysis['urgency_score']}, "
          f"Fear: {analysis['fear_score']}, "
          f"Authority: {analysis['authority_score']}")

    reasons = detector.explain(email)
    if reasons:
        print("  Red flags:")
        for reason in reasons[:5]:
            print(f"    - {reason}")

    print("-" * 60 + "\n")
```

### 6.3 Análisis de Comunicaciones Sospechosas

```python
from typing import Dict, List
from datetime import datetime

class InsiderThreatAnalyzer:
    """
    Analiza comunicaciones para detectar posibles insider threats.

    Indicadores:
    - Sentimiento negativo persistente
    - Menciones de frustración, resentimiento
    - Discusiones sobre acceso, datos, seguridad
    - Cambios en patrones de comunicación
    """

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.sentiment = SentimentIntensityAnalyzer()

        # Palabras clave de riesgo
        self.risk_keywords = {
            'data_access': ['password', 'credentials', 'access', 'database',
                           'download', 'export', 'copy', 'backup'],
            'frustration': ['unfair', 'hate', 'quit', 'resign', 'fired',
                           'underpaid', 'unappreciated', 'deserve'],
            'money': ['money', 'payment', 'sell', 'buyer', 'offer',
                     'bitcoin', 'crypto', 'cash'],
            'external': ['competitor', 'interview', 'recruiter', 'linkedin',
                        'new job', 'opportunity'],
        }

    def analyze_message(self, message: str, sender: str,
                       timestamp: datetime) -> Dict:
        """Analiza un mensaje individual."""
        # Sentiment
        sentiment = self.sentiment.polarity_scores(message)

        # Keyword matching
        message_lower = message.lower()
        keyword_matches = {}

        for category, keywords in self.risk_keywords.items():
            matches = [kw for kw in keywords if kw in message_lower]
            if matches:
                keyword_matches[category] = matches

        # Risk score
        risk_score = 0
        if sentiment['compound'] < -0.5:
            risk_score += 2  # Muy negativo
        elif sentiment['compound'] < -0.2:
            risk_score += 1  # Negativo

        risk_score += len(keyword_matches) * 1.5

        return {
            'sender': sender,
            'timestamp': timestamp,
            'sentiment': sentiment['compound'],
            'keyword_categories': list(keyword_matches.keys()),
            'matched_keywords': keyword_matches,
            'risk_score': min(risk_score, 10),
            'risk_level': self._get_risk_level(risk_score)
        }

    def _get_risk_level(self, score: float) -> str:
        if score >= 5:
            return 'HIGH'
        elif score >= 3:
            return 'MEDIUM'
        elif score >= 1:
            return 'LOW'
        return 'NONE'

    def analyze_conversation_history(self,
                                    messages: List[Dict]) -> Dict:
        """
        Analiza historial de conversaciones de un empleado.
        Busca patrones y tendencias.
        """
        if not messages:
            return {'error': 'No messages'}

        analyses = [
            self.analyze_message(
                m['text'],
                m['sender'],
                m.get('timestamp', datetime.now())
            )
            for m in messages
        ]

        # Estadísticas
        sentiments = [a['sentiment'] for a in analyses]
        risk_scores = [a['risk_score'] for a in analyses]

        all_categories = []
        for a in analyses:
            all_categories.extend(a['keyword_categories'])

        return {
            'total_messages': len(messages),
            'avg_sentiment': sum(sentiments) / len(sentiments),
            'sentiment_trend': 'declining' if len(sentiments) > 1 and
                              sentiments[-1] < sentiments[0] else 'stable',
            'avg_risk_score': sum(risk_scores) / len(risk_scores),
            'high_risk_messages': sum(1 for a in analyses
                                     if a['risk_level'] == 'HIGH'),
            'keyword_categories_seen': list(set(all_categories)),
            'overall_risk': self._get_risk_level(sum(risk_scores) / len(risk_scores)),
            'details': analyses
        }


# Ejemplo
analyzer = InsiderThreatAnalyzer()

# Historial de mensajes de un empleado
messages = [
    {
        'text': "I've been working so hard but no one appreciates my work.",
        'sender': 'employee1',
        'timestamp': datetime(2024, 1, 1)
    },
    {
        'text': "The project is going well, we should finish on time.",
        'sender': 'employee1',
        'timestamp': datetime(2024, 1, 15)
    },
    {
        'text': "This is so unfair. I deserve a promotion, not John.",
        'sender': 'employee1',
        'timestamp': datetime(2024, 2, 1)
    },
    {
        'text': "I hate this company. Maybe I should look for other opportunities.",
        'sender': 'employee1',
        'timestamp': datetime(2024, 2, 15)
    },
    {
        'text': "Does anyone know how to export the customer database?",
        'sender': 'employee1',
        'timestamp': datetime(2024, 3, 1)
    },
]

print("=== ANÁLISIS DE INSIDER THREAT ===\n")
report = analyzer.analyze_conversation_history(messages)

print(f"Total mensajes analizados: {report['total_messages']}")
print(f"Sentimiento promedio: {report['avg_sentiment']:.2f}")
print(f"Tendencia de sentimiento: {report['sentiment_trend']}")
print(f"Mensajes de alto riesgo: {report['high_risk_messages']}")
print(f"Categorías de keywords vistas: {report['keyword_categories_seen']}")
print(f"Nivel de riesgo general: {report['overall_risk']}")

print("\n--- Detalle por mensaje ---")
for detail in report['details']:
    if detail['risk_level'] != 'NONE':
        print(f"\n[{detail['risk_level']}] Sentiment: {detail['sentiment']:.2f}")
        print(f"  Keywords: {detail['matched_keywords']}")
```

---

## 7. Evaluación y Métricas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MÉTRICAS PARA SENTIMENT ANALYSIS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CLASIFICACIÓN BINARIA (pos/neg):                                           │
│    • Accuracy: % de predicciones correctas                                  │
│    • Precision: TP / (TP + FP)                                              │
│    • Recall: TP / (TP + FN)                                                 │
│    • F1-Score: 2 * (P * R) / (P + R)                                        │
│                                                                              │
│  CLASIFICACIÓN MULTI-CLASE (pos/neu/neg o 1-5 stars):                       │
│    • Macro F1: Promedio de F1 por clase                                     │
│    • Weighted F1: F1 ponderado por tamaño de clase                          │
│    • Confusion Matrix: Ver errores entre clases                             │
│                                                                              │
│  REGRESIÓN (score continuo):                                                 │
│    • MSE/RMSE: Error cuadrático                                             │
│    • MAE: Error absoluto                                                    │
│    • Correlation: Pearson/Spearman                                          │
│                                                                              │
│  IMPORTANTE:                                                                 │
│    • Datasets de sentiment suelen estar desbalanceados                      │
│    • Usar class_weight='balanced' o técnicas de resampling                  │
│    • Macro F1 es mejor que accuracy para clases desbalanceadas              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SENTIMENT ANALYSIS - RESUMEN                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ENFOQUES:                                                                   │
│    • Léxico (VADER, TextBlob): Rápido, sin entrenamiento                   │
│    • ML tradicional (TF-IDF + LR/SVM): Balance velocidad/precisión         │
│    • Deep Learning (BERT): Mejor precisión, más costoso                    │
│                                                                              │
│  TIPOS DE ANÁLISIS:                                                          │
│    • Polaridad: Positivo/Negativo/Neutro                                    │
│    • Emociones: Joy, Anger, Fear, Sadness, etc.                            │
│    • Aspectos: Sentimiento por característica                               │
│                                                                              │
│  EN SEGURIDAD:                                                               │
│    • Detección de contenido tóxico                                          │
│    • Análisis de phishing (urgencia, miedo)                                 │
│    • Insider threat detection                                                │
│    • Monitoreo de dark web                                                   │
│                                                                              │
│  HERRAMIENTAS:                                                               │
│    • VADER: Social media, emojis                                            │
│    • TextBlob: Simple, general                                              │
│    • Transformers/Hugging Face: Estado del arte                             │
│                                                                              │
│  MEJORES PRÁCTICAS:                                                          │
│    ✓ Usar modelo apropiado al dominio                                       │
│    ✓ Manejar clases desbalanceadas                                          │
│    ✓ Considerar contexto y sarcasmo                                         │
│    ✓ Validar con ejemplos del dominio específico                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
