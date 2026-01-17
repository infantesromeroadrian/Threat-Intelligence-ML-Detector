# Transformers y BERT

## 1. La Revolución Transformer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVOLUCIÓN DE NLP                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  2013: Word2Vec        → Embeddings estáticos                               │
│  2014: GloVe           → Embeddings globales                                │
│  2015: LSTM/Attention  → Secuencias con atención                            │
│  2017: TRANSFORMER     → "Attention Is All You Need" ⭐                      │
│  2018: BERT            → Bidireccional, pre-training                        │
│  2019: GPT-2           → Generación de texto                                │
│  2020: GPT-3           → Few-shot learning                                  │
│  2022: ChatGPT         → Instruction following                              │
│  2023+: GPT-4, Claude  → Multimodal, reasoning                              │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ¿POR QUÉ TRANSFORMER?                                                      │
│                                                                              │
│  LSTM:     Sequential     O(n) pasos, difícil paralelizar                   │
│            [w₁]→[w₂]→[w₃]→[w₄]                                              │
│                                                                              │
│  TRANSFORMER: Paralelo   Todos los tokens a la vez                          │
│            [w₁] [w₂] [w₃] [w₄]                                              │
│              ↕   ↕   ↕   ↕     ← Self-attention entre todos                │
│                                                                              │
│  Resultado: Entrena 10-100x más rápido en GPU                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Self-Attention: El Corazón del Transformer

### 2.1 Intuición

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SELF-ATTENTION                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Pregunta: ¿Cómo sabe el modelo que "it" se refiere a "animal"?             │
│                                                                              │
│  "The animal didn't cross the street because it was too tired"              │
│                                                                              │
│  Self-attention permite que cada palabra "mire" a todas las demás           │
│  y decida cuáles son relevantes para entenderla.                            │
│                                                                              │
│                    The animal didn't cross the street because it was tired  │
│                     │    │      │     │     │    │      │    │  │    │     │
│                     └────┴──────┴─────┴─────┴────┴──────┴────┼──┴────┘     │
│                                                               │              │
│                                 "it" atiende fuertemente a "animal"          │
│                                                                              │
│  PROCESO:                                                                    │
│    1. Para cada palabra, crear 3 vectores: Query (Q), Key (K), Value (V)    │
│    2. Calcular atención: ¿Cuánto debe "it" atender a cada palabra?          │
│    3. Score = Q_it · K_animal (producto punto)                              │
│    4. Normalizar con softmax                                                │
│    5. Output = suma ponderada de Values                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Fórmula Matemática

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCALED DOT-PRODUCT ATTENTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                            Q · Kᵀ                                           │
│  Attention(Q, K, V) = softmax(─────) · V                                    │
│                             √dₖ                                             │
│                                                                              │
│  Donde:                                                                      │
│    Q = Queries (lo que busco)         Shape: (seq_len, d_k)                 │
│    K = Keys (lo que ofrezco)          Shape: (seq_len, d_k)                 │
│    V = Values (la información)        Shape: (seq_len, d_v)                 │
│    dₖ = dimensión de las keys                                               │
│                                                                              │
│  ¿Por qué dividir por √dₖ?                                                  │
│    Sin escalar, los productos punto pueden ser muy grandes,                 │
│    haciendo que softmax sature y gradientes sean muy pequeños.              │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EJEMPLO NUMÉRICO:                                                           │
│                                                                              │
│  Secuencia: ["the", "cat", "sat"]                                           │
│                                                                              │
│  Q = [q_the, q_cat, q_sat]    ← Cada palabra tiene un query                │
│  K = [k_the, k_cat, k_sat]    ← Cada palabra tiene una key                 │
│  V = [v_the, v_cat, v_sat]    ← Cada palabra tiene un value                │
│                                                                              │
│  Scores para "sat":                                                          │
│    score(sat, the) = q_sat · k_the = 0.2                                    │
│    score(sat, cat) = q_sat · k_cat = 0.8  ← "sat" atiende más a "cat"      │
│    score(sat, sat) = q_sat · k_sat = 0.5                                    │
│                                                                              │
│  Después de softmax: [0.15, 0.55, 0.30]                                     │
│                                                                              │
│  Output para "sat" = 0.15·v_the + 0.55·v_cat + 0.30·v_sat                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Multi-Head Attention

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MULTI-HEAD ATTENTION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  En lugar de una sola atención, usar múltiples "cabezas" en paralelo.       │
│  Cada cabeza puede aprender diferentes tipos de relaciones.                 │
│                                                                              │
│                           Input                                              │
│                             │                                                │
│              ┌──────────────┼──────────────┐                                │
│              │              │              │                                │
│              ▼              ▼              ▼                                │
│         ┌────────┐    ┌────────┐    ┌────────┐                             │
│         │ Head 1 │    │ Head 2 │    │ Head 3 │   ...                       │
│         │(sintax)│    │(semant)│    │(coref) │                             │
│         └────────┘    └────────┘    └────────┘                             │
│              │              │              │                                │
│              └──────────────┼──────────────┘                                │
│                             │                                                │
│                             ▼                                                │
│                        Concatenar                                            │
│                             │                                                │
│                             ▼                                                │
│                      Linear (Wₒ)                                            │
│                             │                                                │
│                             ▼                                                │
│                          Output                                              │
│                                                                              │
│  MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) · Wₒ                         │
│  donde headᵢ = Attention(Q·Wᵢᵠ, K·Wᵢᵏ, V·Wᵢᵛ)                              │
│                                                                              │
│  BERT-base: 12 heads, d_model=768, d_k=d_v=64                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Arquitectura Transformer

### 3.1 Encoder

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRANSFORMER ENCODER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          Input Embeddings                                    │
│                                +                                             │
│                     Positional Encoding                                      │
│                                │                                             │
│                   ┌────────────┴────────────┐                               │
│           ×N     │                          │                               │
│                   │    ┌────────────────┐   │                               │
│                   │    │ Multi-Head     │   │                               │
│                   ├───▶│ Self-Attention │───┤  (Add & Norm)                 │
│                   │    └────────────────┘   │                               │
│                   │            │            │                               │
│                   │            ▼            │                               │
│                   │    ┌────────────────┐   │                               │
│                   │    │  Feed Forward  │   │                               │
│                   └───▶│    Network     │───┘  (Add & Norm)                 │
│                        └────────────────┘                                   │
│                                │                                             │
│                                ▼                                             │
│                      Encoder Output                                          │
│                                                                              │
│  COMPONENTES:                                                                │
│    • Positional Encoding: Añade información de posición                     │
│    • Multi-Head Attention: Cada token atiende a todos los demás            │
│    • Feed Forward: MLP aplicado a cada posición                             │
│    • Add & Norm: Residual connection + LayerNorm                            │
│                                                                              │
│  BERT usa SOLO el encoder (N=12 para base, N=24 para large)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Positional Encoding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      POSITIONAL ENCODING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PROBLEMA: Attention es permutation-invariant                               │
│            ["cat", "sat", "mat"] produce misma atención que                 │
│            ["mat", "cat", "sat"] sin posición!                              │
│                                                                              │
│  SOLUCIÓN: Añadir vector de posición a cada embedding                       │
│                                                                              │
│  FÓRMULAS ORIGINALES (sinusoidal):                                          │
│                                                                              │
│  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))                              │
│  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))                              │
│                                                                              │
│  Ejemplo para d_model=4:                                                     │
│                                                                              │
│  Posición 0: [sin(0),   cos(0),   sin(0),   cos(0)]   = [0, 1, 0, 1]       │
│  Posición 1: [sin(1),   cos(1),   sin(0.01),cos(0.01)]= [0.84,0.54,0.01,1] │
│  Posición 2: [sin(2),   cos(2),   sin(0.02),cos(0.02)]= [0.91,-0.42,...]   │
│                                                                              │
│  PROPIEDADES:                                                                │
│    • Determinístico (no requiere entrenamiento)                             │
│    • Puede generalizar a secuencias más largas que las vistas               │
│    • PE(pos+k) puede expresarse como función lineal de PE(pos)              │
│                                                                              │
│  BERT usa LEARNED positional embeddings en lugar de sinusoidal              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. BERT (Bidirectional Encoder Representations from Transformers)

### 4.1 Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BERT                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input:  [CLS] The cat sat on the mat [SEP]                                 │
│            │    │   │   │   │   │   │   │                                   │
│            ▼    ▼   ▼   ▼   ▼   ▼   ▼   ▼                                   │
│         ┌────────────────────────────────────┐                              │
│         │      Token Embeddings (30522)      │                              │
│         │           +                        │                              │
│         │      Segment Embeddings (2)        │  ← Oración A o B             │
│         │           +                        │                              │
│         │      Position Embeddings (512)     │  ← Máx 512 tokens            │
│         └────────────────────────────────────┘                              │
│                          │                                                   │
│                          ▼                                                   │
│         ┌────────────────────────────────────┐                              │
│         │                                    │                              │
│         │      Transformer Encoder ×12       │  ← BERT-base                 │
│         │      (768 hidden, 12 heads)        │                              │
│         │                                    │                              │
│         └────────────────────────────────────┘                              │
│                          │                                                   │
│            ┌─────────────┴─────────────┐                                    │
│            ▼                           ▼                                    │
│         [CLS]                    Token outputs                              │
│     (classification)           (token-level tasks)                          │
│                                                                              │
│  TAMAÑOS:                                                                    │
│    BERT-base:  12 layers, 768 hidden, 12 heads, 110M params                │
│    BERT-large: 24 layers, 1024 hidden, 16 heads, 340M params               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Pre-training Tasks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BERT PRE-TRAINING                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BERT se pre-entrena con dos tareas NO supervisadas:                        │
│                                                                              │
│  1. MASKED LANGUAGE MODEL (MLM)                                             │
│  ─────────────────────────────                                              │
│  Input:  "The [MASK] sat on the [MASK]"                                     │
│  Target: "The  cat  sat on the  mat"                                        │
│                                                                              │
│  • 15% de tokens se enmascaran aleatoriamente                               │
│  • De esos: 80% → [MASK], 10% → random, 10% → sin cambio                   │
│  • Modelo debe predecir tokens originales                                   │
│  • BIDIRECCIONAL: usa contexto de ambos lados                               │
│                                                                              │
│                                                                              │
│  2. NEXT SENTENCE PREDICTION (NSP)                                          │
│  ───────────────────────────────────                                        │
│  Input:  [CLS] Sentence A [SEP] Sentence B [SEP]                            │
│  Target: ¿B sigue a A? (IsNext / NotNext)                                   │
│                                                                              │
│  Ejemplo positivo:                                                           │
│    A: "The cat is sleeping"                                                 │
│    B: "It looks very peaceful"  ← IsNext                                    │
│                                                                              │
│  Ejemplo negativo:                                                           │
│    A: "The cat is sleeping"                                                 │
│    B: "Tokyo is the capital of Japan"  ← NotNext (random)                  │
│                                                                              │
│  • 50% pares reales, 50% pares aleatorios                                   │
│  • Ayuda a entender relaciones entre oraciones                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Fine-tuning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BERT FINE-TUNING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CLASIFICACIÓN DE TEXTO (Sentiment, Spam, etc.)                             │
│  ──────────────────────────────────────────────                             │
│                                                                              │
│  [CLS] This movie is great [SEP]                                            │
│    │                                                                         │
│    ▼                                                                         │
│  BERT Encoder                                                                │
│    │                                                                         │
│    ▼                                                                         │
│  [CLS] → Dense → Softmax → [Positive, Negative]                             │
│                                                                              │
│                                                                              │
│  NER / TOKEN CLASSIFICATION                                                  │
│  ──────────────────────────                                                 │
│                                                                              │
│  [CLS] John works at Google [SEP]                                           │
│          │     │     │   │                                                  │
│          ▼     ▼     ▼   ▼                                                  │
│        B-PER   O     O  B-ORG                                               │
│                                                                              │
│                                                                              │
│  QUESTION ANSWERING                                                          │
│  ──────────────────                                                         │
│                                                                              │
│  [CLS] Question [SEP] Context with answer [SEP]                             │
│                            ↑ start    ↑ end                                 │
│                                                                              │
│  Predecir posición de inicio y fin de la respuesta                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementación con Hugging Face

### 5.1 Clasificación de Texto

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from typing import List, Dict

class BERTClassifier:
    """
    Clasificador de texto usando BERT pre-entrenado.
    """

    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 num_labels: int = 2,
                 max_length: int = 128):
        """
        Args:
            model_name: Nombre del modelo pre-entrenado
            num_labels: Número de clases
            max_length: Longitud máxima de secuencia
        """
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cargar tokenizer y modelo
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)

    def tokenize(self, texts: List[str]) -> Dict:
        """Tokeniza textos para BERT."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predice clases para textos.
        """
        self.model.eval()

        # Tokenizar
        inputs = self.tokenize(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inferencia
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        return predictions.cpu().numpy()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predice probabilidades.
        """
        self.model.eval()

        inputs = self.tokenize(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()


# Ejemplo de uso
classifier = BERTClassifier(num_labels=2)

# Textos de ejemplo
texts = [
    "This product is amazing! Best purchase ever.",
    "Terrible quality. Complete waste of money.",
    "It's okay, nothing special.",
]

# Predecir
predictions = classifier.predict(texts)
probabilities = classifier.predict_proba(texts)

print("Predicciones:")
for text, pred, probs in zip(texts, predictions, probabilities):
    label = "Positive" if pred == 1 else "Negative"
    print(f"  '{text[:40]}...' → {label} (conf: {probs[pred]:.2f})")
```

### 5.2 Fine-tuning Completo

```python
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def fine_tune_bert_classifier(
    train_texts: list,
    train_labels: list,
    val_texts: list,
    val_labels: list,
    model_name: str = 'bert-base-uncased',
    num_labels: int = 2,
    output_dir: str = './bert_classifier',
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """
    Fine-tune BERT para clasificación.
    """
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenizar datasets
    def tokenize_function(examples):
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
    })
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Formato para PyTorch
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        learning_rate=learning_rate,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Entrenar
    trainer.train()

    # Guardar
    trainer.save_model(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')

    return trainer


# Ejemplo de uso
train_texts = [
    "This is great!", "Terrible product", "Love it!", "Waste of money",
    "Excellent quality", "Very disappointed", "Highly recommend", "Awful"
]
train_labels = [1, 0, 1, 0, 1, 0, 1, 0]

val_texts = ["Amazing!", "Bad experience"]
val_labels = [1, 0]

# Fine-tune (comentado para no ejecutar)
# trainer = fine_tune_bert_classifier(
#     train_texts, train_labels,
#     val_texts, val_labels,
#     epochs=3
# )
```

### 5.3 Feature Extraction con BERT

```python
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import List

class BERTFeatureExtractor:
    """
    Extrae embeddings de BERT para usar en otros modelos.
    """

    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 pooling: str = 'cls'):
        """
        Args:
            pooling: 'cls' (token [CLS]), 'mean' (promedio), 'max' (máximo)
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.pooling = pooling

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def extract(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extrae embeddings para una lista de textos.

        Returns:
            embeddings: Array de shape (n_texts, 768)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenizar
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extraer
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
                attention_mask = inputs['attention_mask']

            # Pooling
            if self.pooling == 'cls':
                embeddings = hidden_states[:, 0, :]  # Token [CLS]

            elif self.pooling == 'mean':
                # Mean pooling con máscara de atención
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                masked_hidden = hidden_states * mask
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                embeddings = sum_hidden / sum_mask

            elif self.pooling == 'max':
                # Max pooling
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states[mask == 0] = -1e9
                embeddings = hidden_states.max(dim=1)[0]

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


# Ejemplo: Usar BERT embeddings con classifier tradicional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Extraer features
extractor = BERTFeatureExtractor(pooling='mean')

texts = [
    "Malware detected on system",
    "Normal user login",
    "Suspicious network activity",
    "Regular file access",
    "Potential data exfiltration",
    "Standard system update",
]
labels = [1, 0, 1, 0, 1, 0]  # 1=malicious, 0=benign

# Obtener embeddings
X = extractor.extract(texts)
print(f"Embeddings shape: {X.shape}")  # (6, 768)

# Entrenar classifier
clf = LogisticRegression()
clf.fit(X, labels)

# Predecir nuevo texto
new_texts = ["Ransomware encrypting files", "User logged out"]
new_embeddings = extractor.extract(new_texts)
predictions = clf.predict(new_embeddings)
print(f"Predicciones: {predictions}")
```

---

## 6. Variantes de BERT

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FAMILIA BERT                                          │
├───────────────────┬─────────────────────────────────────────────────────────┤
│      Modelo       │                   Características                        │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ BERT-base         │ 12 layers, 110M params, baseline                        │
│ BERT-large        │ 24 layers, 340M params, mejor rendimiento               │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ RoBERTa           │ Mismo que BERT pero mejor pre-training:                 │
│                   │ - Sin NSP, más datos, más tiempo                        │
│                   │ - Dynamic masking                                        │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ DistilBERT        │ 40% más pequeño, 60% más rápido                         │
│                   │ Destilación de conocimiento                             │
│                   │ Ideal para producción                                    │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ ALBERT            │ Factorización de embeddings                             │
│                   │ Sharing de parámetros entre layers                      │
│                   │ 18x menos parámetros                                    │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ ELECTRA           │ Replaced Token Detection (más eficiente)                │
│                   │ Mejor rendimiento con menos compute                     │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ DeBERTa           │ Disentangled attention                                  │
│                   │ Enhanced mask decoder                                   │
│                   │ SOTA en muchos benchmarks                               │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ SecBERT           │ Pre-entrenado en textos de ciberseguridad              │
│                   │ CVEs, reportes de amenazas, etc.                        │
└───────────────────┴─────────────────────────────────────────────────────────┘
```

### Ejemplo con DistilBERT (para producción)

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

class FastBERTClassifier:
    """
    Clasificador rápido usando DistilBERT.
    40% más pequeño, 60% más rápido que BERT.
    """

    def __init__(self, num_labels: int = 2):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        )
        self.model.eval()

    def predict(self, texts: list) -> list:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        return predictions.tolist()


# Comparar tamaños
# BERT-base:    110M params, ~440MB
# DistilBERT:   66M params,  ~260MB
# Velocidad:    ~1.6x más rápido en CPU
```

---

## 7. GPT y Modelos Generativos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BERT vs GPT                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BERT (Encoder-only):                                                        │
│    • Bidireccional (ve toda la secuencia)                                   │
│    • Ideal para: clasificación, NER, QA                                     │
│    • Pre-training: MLM + NSP                                                │
│                                                                              │
│        [w₁] ↔ [w₂] ↔ [w₃] ↔ [w₄]                                           │
│               ↕ atención bidireccional                                      │
│                                                                              │
│                                                                              │
│  GPT (Decoder-only):                                                         │
│    • Unidireccional (solo ve tokens anteriores)                             │
│    • Ideal para: generación de texto                                        │
│    • Pre-training: Next token prediction                                    │
│                                                                              │
│        [w₁] → [w₂] → [w₃] → [w₄]                                           │
│          ↓     ↓      ↓      ↓                                              │
│        [w₂]  [w₃]   [w₄]   [w₅]    ← predice siguiente                     │
│                                                                              │
│                                                                              │
│  T5, BART (Encoder-Decoder):                                                 │
│    • Encoder bidireccional + Decoder autoregresivo                          │
│    • Ideal para: traducción, summarization                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Generación con GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class TextGenerator:
    """
    Generador de texto con GPT-2.
    """

    def __init__(self, model_name: str = 'gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

        # GPT-2 no tiene pad token por defecto
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self,
                 prompt: str,
                 max_length: int = 100,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 num_return_sequences: int = 1) -> list:
        """
        Genera texto dado un prompt.

        Args:
            temperature: Mayor = más creativo, menor = más determinístico
            top_p: Nucleus sampling (probabilidad acumulada)
            top_k: Solo considerar los k tokens más probables
        """
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated.append(text)

        return generated


# Ejemplo
generator = TextGenerator('gpt2')

prompts = [
    "The security analyst discovered that",
    "To protect against ransomware, you should",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    generated = generator.generate(prompt, max_length=50)
    print(f"Generated: {generated[0]}")
```

---

## 8. BERT para Seguridad

### 8.1 Clasificación de CVEs

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CVEClassifier:
    """
    Clasifica descripciones de CVE por tipo de vulnerabilidad.

    Clases ejemplo:
    - Buffer Overflow
    - SQL Injection
    - XSS
    - Remote Code Execution
    - Denial of Service
    """

    def __init__(self, num_classes: int = 5):
        # Usar SecBERT si está disponible, sino BERT normal
        model_name = 'jackaduma/SecBERT'  # o 'bert-base-uncased'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )

        self.class_names = [
            'Buffer Overflow',
            'SQL Injection',
            'XSS',
            'Remote Code Execution',
            'Denial of Service'
        ]

    def classify(self, cve_descriptions: list) -> list:
        """Clasifica descripciones de CVE."""
        inputs = self.tokenizer(
            cve_descriptions,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        return [self.class_names[p] for p in predictions.tolist()]


# Ejemplo
cve_descriptions = [
    "A stack-based buffer overflow in the handling of HTTP headers allows remote attackers to execute arbitrary code",
    "The login page does not properly sanitize user input, allowing SQL injection attacks",
    "User-supplied input is reflected in the page without proper encoding, leading to XSS",
]

# classifier = CVEClassifier()
# results = classifier.classify(cve_descriptions)
```

### 8.2 Extracción de IOCs con BERT NER

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch

class IOCExtractor:
    """
    Extrae Indicators of Compromise (IOCs) de texto usando NER.

    IOCs: IPs, dominios, hashes, URLs, emails maliciosos, etc.
    """

    def __init__(self):
        # Modelo NER pre-entrenado (o fine-tuned para IOCs)
        self.ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

    def extract(self, text: str) -> dict:
        """
        Extrae entidades del texto.
        """
        entities = self.ner_pipeline(text)

        # Agrupar por tipo
        iocs = {
            'organizations': [],
            'locations': [],
            'persons': [],
            'misc': []
        }

        for entity in entities:
            entity_type = entity['entity_group']
            value = entity['word']

            if entity_type == 'ORG':
                iocs['organizations'].append(value)
            elif entity_type == 'LOC':
                iocs['locations'].append(value)
            elif entity_type == 'PER':
                iocs['persons'].append(value)
            else:
                iocs['misc'].append(value)

        return iocs


# Ejemplo
extractor = IOCExtractor()

threat_report = """
The APT29 group, also known as Cozy Bear, has been targeting government
organizations in Washington DC. The malware communicates with servers
in Moscow and Beijing.
"""

iocs = extractor.extract(threat_report)
print("Entidades extraídas:")
for category, values in iocs.items():
    if values:
        print(f"  {category}: {values}")
```

---

## 9. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMERS Y BERT - RESUMEN                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TRANSFORMER:                                                                │
│    • Self-Attention: Cada token atiende a todos los demás                   │
│    • Query-Key-Value: Mecanismo de búsqueda de relevancia                   │
│    • Multi-Head: Múltiples tipos de atención en paralelo                    │
│    • Paralelizable: Mucho más rápido que LSTM en GPU                        │
│                                                                              │
│  BERT:                                                                       │
│    • Encoder-only Transformer                                               │
│    • Bidireccional (ve contexto completo)                                   │
│    • Pre-training: MLM + NSP en texto masivo                                │
│    • Fine-tuning: Adaptar a tarea específica con pocos datos                │
│                                                                              │
│  FLUJO TÍPICO:                                                               │
│    1. Cargar modelo pre-entrenado (bert-base-uncased, etc.)                 │
│    2. Añadir capa de clasificación para tu tarea                            │
│    3. Fine-tune con tus datos (2-4 epochs)                                  │
│    4. Evaluar y desplegar                                                   │
│                                                                              │
│  VARIANTES:                                                                  │
│    • DistilBERT: Más rápido, para producción                                │
│    • RoBERTa: Mejor pre-training                                            │
│    • SecBERT: Especializado en ciberseguridad                               │
│    • GPT: Generación de texto                                               │
│                                                                              │
│  EN SEGURIDAD:                                                               │
│    • Clasificación de CVEs/vulnerabilidades                                 │
│    • Extracción de IOCs                                                     │
│    • Análisis de reportes de amenazas                                       │
│    • Detección de phishing/spam                                             │
│    • NER para threat intelligence                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
