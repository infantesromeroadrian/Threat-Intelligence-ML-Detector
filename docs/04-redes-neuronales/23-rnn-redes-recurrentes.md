# Redes Neuronales Recurrentes (RNN)

## Introduccion

Las **Redes Neuronales Recurrentes (RNN)** son arquitecturas disenadas para procesar **secuencias de datos** donde el orden importa. A diferencia de las redes feedforward, las RNN mantienen un **estado oculto** que captura informacion de pasos anteriores.

```
FEEDFORWARD vs RECURRENTE
=========================

Feedforward (MLP/CNN):              Recurrente (RNN):

Input → Hidden → Output             Input_t → Hidden_t → Output_t
                                         ↑         |
                                         |    Estado    |
                                         └─────────────┘
                                         (memoria temporal)

- Cada entrada independiente          - Entradas secuenciales
- Sin memoria                         - Memoria de contexto
- Datos tabulares/imagenes            - Series temporales/texto
```

## Arquitectura RNN Basica

### Estructura Fundamental

```
DESENROLLADO DE UNA RNN (Unfolding)
===================================

Forma Compacta:                 Forma Desenrollada:

    ┌─────┐                     t=0      t=1      t=2      t=3
    │  h  │←──┐                  ↓        ↓        ↓        ↓
    └──┬──┘   │                ┌───┐    ┌───┐    ┌───┐    ┌───┐
       │      │                │ x₀│    │ x₁│    │ x₂│    │ x₃│
    ┌──▼──┐   │                └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘
    │ RNN │───┘                  │        │        │        │
    └──┬──┘                    ┌─▼─┐    ┌─▼─┐    ┌─▼─┐    ┌─▼─┐
       │                       │h₀ │───→│h₁ │───→│h₂ │───→│h₃ │
    ┌──▼──┐                    └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘
    │  y  │                      │        │        │        │
    └─────┘                    ┌─▼─┐    ┌─▼─┐    ┌─▼─┐    ┌─▼─┐
                               │ y₀│    │ y₁│    │ y₂│    │ y₃│
                               └───┘    └───┘    └───┘    └───┘

Los pesos (W_xh, W_hh, W_hy) se COMPARTEN en todos los pasos temporales
```

### Ecuaciones de la RNN Vanilla

```
PASO FORWARD EN RNN
===================

Estado oculto:
    h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)

Salida:
    y_t = W_hy · h_t + b_y

Donde:
- x_t: entrada en tiempo t
- h_t: estado oculto en tiempo t
- y_t: salida en tiempo t
- W_xh: pesos input → hidden
- W_hh: pesos hidden → hidden (recurrencia)
- W_hy: pesos hidden → output
```

### Tipos de Arquitecturas RNN

```
ARQUITECTURAS SEGUN ENTRADA/SALIDA
==================================

1. One-to-One (Feedforward clasico):
   x → [RNN] → y
   Ejemplo: Clasificacion imagen

2. One-to-Many:
   x → [RNN] → y₁ → y₂ → y₃ → ...
   Ejemplo: Generacion de descripcion de imagen

3. Many-to-One:
   x₁ → x₂ → x₃ → [RNN] → y
   Ejemplo: Clasificacion de sentimiento

4. Many-to-Many (sync):
   x₁ → x₂ → x₃ → x₄
    ↓    ↓    ↓    ↓
   y₁   y₂   y₃   y₄
   Ejemplo: Etiquetado POS

5. Many-to-Many (async):
   x₁ → x₂ → x₃ → [RNN] → y₁ → y₂
   Ejemplo: Traduccion maquina (encoder-decoder)
```

## Problema del Desvanecimiento del Gradiente

### El Problema

```
DESVANECIMIENTO DEL GRADIENTE
=============================

Durante backpropagation through time (BPTT):

∂L/∂W = Σ (∂L/∂h_T) · (∂h_T/∂h_t) · (∂h_t/∂W)
                        ↑
                  Producto de muchos terminos

∂h_T/∂h_t = Π_{i=t}^{T-1} ∂h_{i+1}/∂h_i
          = Π_{i=t}^{T-1} W_hh · diag(tanh'(h_i))

Si los valores propios de W_hh:
- < 1: gradientes → 0 (desvanecimiento)
- > 1: gradientes → ∞ (explosion)

Secuencia larga (T=100):
┌────────────────────────────────────────────────────┐
│ x₁ ──→ ... ──→ x₅₀ ──→ ... ──→ x₁₀₀              │
│ ↓              ↓                ↓                  │
│ h₁ ──→ ... ──→ h₅₀ ──→ ... ──→ h₁₀₀ ──→ Loss     │
│                                                    │
│ Gradiente en h₁ ≈ 0  (informacion perdida!)       │
└────────────────────────────────────────────────────┘
```

### Soluciones

| Problema | Solucion | Mecanismo |
|----------|----------|-----------|
| Desvanecimiento | LSTM/GRU | Gates que controlan flujo |
| Explosion | Gradient clipping | Limitar norma del gradiente |
| Dependencias largas | Attention | Conexiones directas |
| Paralelizacion | Transformers | Sin recurrencia |

---

## LSTM (Long Short-Term Memory)

### Arquitectura LSTM

```
CELULA LSTM
===========

                    ┌───────────────────────────────────────┐
                    │                                       │
    c_{t-1} ─────→ [×] ─────────→ [+] ─────────────→ c_t ──┼──→
                    ↑              ↑                        │
                    │              │                        │
                 ┌──┴──┐      ┌────┴────┐                   │
                 │  f  │      │  i × c̃  │                   │
                 │gate │      │         │                   │
                 └──┬──┘      └────┬────┘                   │
                    │              │                        │
          ┌─────────┼──────────────┼────────────┐          │
          │         │              │            │          │
          │    ┌────┴────┐   ┌─────┴─────┐      │          │
          │    │ σ(forget)│   │σ(i)·tanh(c̃)│    │          │
          │    └────┬────┘   └─────┬─────┘      │          │
          │         │              │            │          │
          │    ┌────┴────────────┴─────────┐   │          │
          │    │      [W · concat(h,x) + b] │   │          │
          │    └────────────┬───────────────┘   │          │
          │                 │                   │          │
          │    ┌────────────┴───────────────┐   │          │
          │    │        concatenate          │   │          │
          │    └────────┬───────┬───────────┘   │          │
          │             │       │               │          │
    h_{t-1}─────────────┘       │               │          │
                                │               │          │
    x_t ────────────────────────┘               │          │
                                                │          │
                                           ┌────┴────┐     │
    h_t ←──────────────────────────────────│ o × tanh│←────┘
                                           │  (c_t)  │
                                           └─────────┘

Gates:
- f (forget): que olvidar del estado anterior
- i (input): que nueva informacion agregar
- o (output): que parte del estado exponer
```

### Ecuaciones LSTM

```
ECUACIONES LSTM
===============

1. Forget Gate (que olvidar):
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

2. Input Gate (que recordar):
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

3. Cell State Update:
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

4. Output Gate (que exponer):
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t ⊙ tanh(c_t)

Donde:
- σ: funcion sigmoide (0-1)
- ⊙: producto elemento a elemento
- [·,·]: concatenacion
```

### Flujo de Gradientes en LSTM

```
POR QUE LSTM RESUELVE EL DESVANECIMIENTO
========================================

En RNN vanilla:
    ∂c_t/∂c_{t-1} = W_hh · diag(tanh')  ← valores < 1, se multiplican

En LSTM:
    ∂c_t/∂c_{t-1} = f_t  ← puede ser cercano a 1!

El "highway" del cell state:
    c_0 → c_1 → c_2 → ... → c_T
           ×f    ×f         ×f

Si f ≈ 1: gradiente fluye sin atenuacion
Si f ≈ 0: olvida informacion irrelevante

┌─────────────────────────────────────────────┐
│ "Autopista de gradientes" del cell state    │
│                                             │
│ c_0 ═══════════════════════════════════ c_T │
│     Sin multiplicaciones de matrices W      │
│     Solo operaciones elemento a elemento    │
└─────────────────────────────────────────────┘
```

---

## GRU (Gated Recurrent Unit)

### Arquitectura GRU

```
CELULA GRU (Simplificacion de LSTM)
===================================

                ┌─────────────────────────────────┐
                │                                 │
                │     ┌───────────────────┐       │
                │     │                   │       │
    h_{t-1} ────┼───→[×]────→[+]────────→ h_t ───┼──→
                │     ↑       ↑                   │
                │  ┌──┴──┐ ┌──┴──┐                │
                │  │1-z_t│ │z_t×h̃│                │
                │  └──┬──┘ └──┬──┘                │
                │     │       │                   │
                │  ┌──┴───────┴──┐                │
                │  │   h_{t-1}   │   ┌─────────┐  │
                │  │      ↓      │   │         │  │
                │  │   ┌─────┐   │   │  h̃_t    │  │
                │  │   │ z_t │   │   │=tanh(Wh·│  │
                │  │   │=σ() │   │   │[r⊙h,x]) │  │
                │  │   └─────┘   │   │         │  │
                │  └─────────────┘   └────┬────┘  │
                │                         │       │
                │         ┌───────────────┘       │
                │         │                       │
                │      ┌──┴──┐                    │
                │      │ r_t │ (reset gate)       │
                │      │=σ() │                    │
                │      └─────┘                    │
                │                                 │
    x_t ────────┴─────────────────────────────────┘

Solo 2 gates vs 3 en LSTM:
- z (update): combina forget + input de LSTM
- r (reset): controla cuanto del pasado considerar
```

### Ecuaciones GRU

```
ECUACIONES GRU
==============

1. Update Gate (mezcla viejo/nuevo):
   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

2. Reset Gate (cuanto del pasado usar):
   r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

3. Candidate Hidden State:
   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

4. Final Hidden State:
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

Interpretacion:
- Si z_t ≈ 0: h_t ≈ h_{t-1} (mantener estado)
- Si z_t ≈ 1: h_t ≈ h̃_t (actualizar estado)
```

### LSTM vs GRU

| Aspecto | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Parametros | Mas (4 matrices W) | Menos (3 matrices W) |
| Cell state | Separado (c_t) | Integrado en h_t |
| Rendimiento | Mejor en secuencias muy largas | Similar en secuencias medias |
| Velocidad | Mas lento | Mas rapido |
| Memoria | Mayor consumo | Menor consumo |

```
CUANDO USAR CADA UNO
====================

LSTM:
├── Secuencias muy largas (>500 tokens)
├── Multiples dependencias temporales
├── Tareas de lenguaje complejas
└── Cuando la memoria no es limitante

GRU:
├── Secuencias medias (<500 tokens)
├── Recursos limitados
├── Prototipado rapido
└── Datasets pequenos (menos overfitting)
```

---

## RNN Bidireccional

### Arquitectura

```
RNN BIDIRECCIONAL
=================

      x_1         x_2         x_3         x_4
       │           │           │           │
       ▼           ▼           ▼           ▼
    ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
───→│ h→_1 │──→│ h→_2 │──→│ h→_3 │──→│ h→_4 │──→  Forward
    └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
       │          │          │          │
       │ concat   │ concat   │ concat   │ concat
       │          │          │          │
    ┌──┴───┐   ┌──┴───┐   ┌──┴───┐   ┌──┴───┐
←───│ h←_1 │←──│ h←_2 │←──│ h←_3 │←──│ h←_4 │←──  Backward
    └──┬───┘   └──┴───┘   └──┴───┘   └──┬───┘
       │          │          │          │
       ▼          ▼          ▼          ▼
     [h→;h←]   [h→;h←]    [h→;h←]    [h→;h←]
       │          │          │          │
       ▼          ▼          ▼          ▼
      y_1        y_2        y_3        y_4

Cada salida tiene contexto de:
- Pasado (forward: x_1, x_2, ..., x_t)
- Futuro (backward: x_T, x_{T-1}, ..., x_t)
```

### Casos de Uso

| Tipo | Bidireccional | Razon |
|------|---------------|-------|
| Clasificacion texto | Si | Contexto completo disponible |
| NER/POS tagging | Si | Necesita contexto ambos lados |
| Traduccion | Solo encoder | Decoder genera secuencialmente |
| Generacion texto | No | Futuro desconocido |
| Analisis logs (offline) | Si | Log completo disponible |
| Deteccion tiempo real | No | Solo pasado disponible |

---

## Implementacion Practica

### RNN/LSTM en PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSequenceClassifier(nn.Module):
    """
    Clasificador de secuencias de logs para deteccion de anomalias.
    Usa LSTM bidireccional para capturar patrones en ambas direcciones.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embedding para tokens de log
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention para ponderar pasos temporales
        self.attention = nn.Linear(
            hidden_dim * self.num_directions, 1
        )

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def attention_pooling(
        self,
        lstm_output: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Aplica attention sobre la secuencia.

        Args:
            lstm_output: [batch, seq_len, hidden*directions]
            mask: [batch, seq_len] - True para posiciones validas

        Returns:
            context: [batch, hidden*directions]
        """
        # Calcular scores de attention
        attn_scores = self.attention(lstm_output).squeeze(-1)  # [batch, seq_len]

        # Aplicar mascara si existe
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Softmax para obtener pesos
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len]

        # Weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # [batch, 1, seq_len]
            lstm_output                  # [batch, seq_len, hidden*dir]
        ).squeeze(1)                     # [batch, hidden*dir]

        return context

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, seq_len] - indices de tokens
            lengths: [batch] - longitudes reales (sin padding)

        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len = x.shape

        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # Pack si tenemos longitudes variables
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # Unpack si fue packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
            # Crear mascara
            mask = torch.arange(seq_len, device=x.device).expand(
                batch_size, seq_len
            ) < lengths.unsqueeze(1)
        else:
            mask = None

        # Attention pooling
        context = self.attention_pooling(lstm_out, mask)

        # Clasificacion
        logits = self.classifier(context)

        return logits


class GRUCommandDetector(nn.Module):
    """
    Detector de comandos maliciosos usando GRU.
    Procesa secuencias de comandos shell para identificar patrones sospechosos.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Usar ultimo estado de ambas direcciones
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed]

        # GRU output
        _, h_n = self.gru(embedded)  # h_n: [2*layers, batch, hidden]

        # Concatenar forward y backward del ultimo layer
        h_forward = h_n[-2, :, :]   # [batch, hidden]
        h_backward = h_n[-1, :, :]  # [batch, hidden]
        h_combined = torch.cat([h_forward, h_backward], dim=1)

        return self.fc(h_combined)


# Ejemplo de uso
def train_log_classifier():
    """Entrena el clasificador de logs."""

    # Configuracion
    vocab_size = 10000
    max_seq_len = 200
    batch_size = 32
    num_epochs = 10

    # Modelo
    model = LogSequenceClassifier(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    )

    # Optimizador y loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop simplificado
    model.train()
    for epoch in range(num_epochs):
        # Datos de ejemplo (en produccion: DataLoader real)
        x = torch.randint(0, vocab_size, (batch_size, max_seq_len))
        lengths = torch.randint(50, max_seq_len, (batch_size,))
        y = torch.randint(0, 2, (batch_size,))

        # Forward
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)

        # Backward
        loss.backward()

        # Gradient clipping (importante para RNN!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train_log_classifier()
```

### RNN/LSTM en Keras/TensorFlow

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model


def create_lstm_classifier(
    vocab_size: int = 10000,
    max_length: int = 200,
    embedding_dim: int = 128,
    lstm_units: int = 256,
    num_classes: int = 2
) -> Model:
    """
    Crea clasificador LSTM bidireccional en Keras.
    """

    # Input
    inputs = layers.Input(shape=(max_length,), dtype='int32')

    # Embedding con mascara para padding
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    )(inputs)

    # LSTM bidireccional apilado
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.2)
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=False, dropout=0.2)
    )(x)

    # Clasificador
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_gru_autoencoder(
    vocab_size: int = 5000,
    max_length: int = 100,
    embedding_dim: int = 64,
    latent_dim: int = 128
) -> tuple[Model, Model, Model]:
    """
    Autoencoder secuencial con GRU para deteccion de anomalias.
    Comandos normales se reconstruyen bien, anomalos no.
    """

    # Encoder
    encoder_inputs = layers.Input(shape=(max_length,))
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    x = layers.GRU(latent_dim * 2, return_sequences=True)(x)
    encoder_outputs = layers.GRU(latent_dim)(x)

    encoder = Model(encoder_inputs, encoder_outputs, name='encoder')

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.RepeatVector(max_length)(decoder_inputs)
    x = layers.GRU(latent_dim, return_sequences=True)(x)
    x = layers.GRU(latent_dim * 2, return_sequences=True)(x)
    decoder_outputs = layers.TimeDistributed(
        layers.Dense(vocab_size, activation='softmax')
    )(x)

    decoder = Model(decoder_inputs, decoder_outputs, name='decoder')

    # Autoencoder completo
    autoencoder_input = layers.Input(shape=(max_length,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)

    autoencoder = Model(autoencoder_input, decoded, name='seq_autoencoder')

    autoencoder.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )

    return autoencoder, encoder, decoder


class AttentionLSTM(layers.Layer):
    """
    LSTM con mecanismo de atencion personalizado.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.lstm = layers.LSTM(units, return_sequences=True)
        self.attention_dense = layers.Dense(1)

    def call(self, inputs, mask=None):
        # LSTM sobre secuencia
        lstm_out = self.lstm(inputs)  # [batch, seq, units]

        # Calcular pesos de atencion
        attention_scores = self.attention_dense(lstm_out)  # [batch, seq, 1]
        attention_scores = tf.squeeze(attention_scores, axis=-1)  # [batch, seq]

        if mask is not None:
            attention_scores = tf.where(mask, attention_scores, -1e9)

        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # [batch, seq]

        # Weighted sum
        context = tf.reduce_sum(
            lstm_out * tf.expand_dims(attention_weights, -1),
            axis=1
        )  # [batch, units]

        return context, attention_weights


# Modelo con attention
def create_attention_model(vocab_size: int, max_length: int) -> Model:
    """Modelo con capa de attention custom."""

    inputs = layers.Input(shape=(max_length,))
    x = layers.Embedding(vocab_size, 128, mask_zero=True)(inputs)

    context, attention = AttentionLSTM(256)(x)

    x = layers.Dense(64, activation='relu')(context)
    outputs = layers.Dense(2, activation='softmax')(x)

    return Model(inputs, outputs)
```

---

## Aplicaciones en Ciberseguridad

### 1. Analisis de Secuencias de Logs

```python
import torch
import torch.nn as nn
from collections import Counter
from typing import Iterator


class LogTokenizer:
    """Tokenizador para lineas de log."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.token_to_idx: dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_token: dict[int, str] = {0: '<PAD>', 1: '<UNK>'}

    def fit(self, logs: list[str]) -> None:
        """Construye vocabulario desde logs."""
        counter: Counter[str] = Counter()

        for log in logs:
            tokens = self._tokenize(log)
            counter.update(tokens)

        # Top tokens
        for idx, (token, _) in enumerate(
            counter.most_common(self.vocab_size - 2), start=2
        ):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def _tokenize(self, log: str) -> list[str]:
        """Tokeniza una linea de log."""
        # Normalizar IPs, timestamps, etc.
        import re
        log = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', log)
        log = re.sub(r'\d{4}-\d{2}-\d{2}', '<DATE>', log)
        log = re.sub(r'\d{2}:\d{2}:\d{2}', '<TIME>', log)
        log = re.sub(r'[0-9a-f]{32,}', '<HASH>', log)

        return log.lower().split()

    def encode(
        self,
        log: str,
        max_length: int = 100
    ) -> list[int]:
        """Convierte log a indices."""
        tokens = self._tokenize(log)[:max_length]
        indices = [
            self.token_to_idx.get(t, self.token_to_idx['<UNK>'])
            for t in tokens
        ]
        # Padding
        indices += [0] * (max_length - len(indices))
        return indices


class LogAnomalyDetector:
    """
    Detector de anomalias en logs usando LSTM.

    Pipeline:
    1. Tokenizar logs
    2. Entrenar LSTM para predecir siguiente token
    3. Alta perplejidad = anomalia
    """

    def __init__(self, vocab_size: int = 10000, hidden_dim: int = 256):
        self.tokenizer = LogTokenizer(vocab_size)
        self.model = self._build_model(vocab_size, hidden_dim)
        self.threshold = 0.0

    def _build_model(
        self,
        vocab_size: int,
        hidden_dim: int
    ) -> nn.Module:
        """Construye modelo de lenguaje LSTM."""

        class LogLM(nn.Module):
            def __init__(self, vocab_size: int, hidden_dim: int):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, vocab_size)

            def forward(
                self,
                x: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                logits = self.fc(lstm_out)
                return logits

        return LogLM(vocab_size, hidden_dim)

    def fit(self, normal_logs: list[str], epochs: int = 10) -> None:
        """Entrena con logs normales."""
        self.tokenizer.fit(normal_logs)

        # Preparar datos
        sequences = [self.tokenizer.encode(log) for log in normal_logs]
        X = torch.tensor(sequences)

        # Input: secuencia[:-1], Target: secuencia[1:]
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            logits = self.model(X[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                X[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # Calcular threshold con datos normales
        self.model.eval()
        with torch.no_grad():
            perplexities = self._compute_perplexity(X)
            self.threshold = perplexities.mean() + 2 * perplexities.std()

    def _compute_perplexity(self, X: torch.Tensor) -> torch.Tensor:
        """Calcula perplejidad por secuencia."""
        with torch.no_grad():
            logits = self.model(X[:, :-1])
            log_probs = torch.log_softmax(logits, dim=-1)

            # Gather probabilidades de tokens correctos
            targets = X[:, 1:].unsqueeze(-1)
            token_log_probs = log_probs.gather(-1, targets).squeeze(-1)

            # Mascara para ignorar padding
            mask = (X[:, 1:] != 0).float()

            # Perplejidad por secuencia
            seq_log_prob = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1)
            perplexity = torch.exp(-seq_log_prob)

            return perplexity

    def predict(self, logs: list[str]) -> list[tuple[str, bool, float]]:
        """
        Predice si los logs son anomalos.

        Returns:
            Lista de (log, es_anomalo, score)
        """
        sequences = [self.tokenizer.encode(log) for log in logs]
        X = torch.tensor(sequences)

        perplexities = self._compute_perplexity(X)

        results = []
        for log, perp in zip(logs, perplexities.tolist()):
            is_anomaly = perp > self.threshold
            results.append((log, is_anomaly, perp))

        return results


# Uso
if __name__ == "__main__":
    # Logs normales de entrenamiento
    normal_logs = [
        "2024-01-15 10:23:45 INFO User login successful from 192.168.1.100",
        "2024-01-15 10:24:01 INFO GET /api/users returned 200",
        "2024-01-15 10:24:15 INFO POST /api/data returned 201",
        # ... mas logs normales
    ]

    # Detector
    detector = LogAnomalyDetector()
    detector.fit(normal_logs)

    # Logs a analizar
    test_logs = [
        "2024-01-15 11:00:00 INFO User login successful from 192.168.1.50",
        "2024-01-15 11:00:05 CRITICAL Multiple failed auth from 10.0.0.1",
        "2024-01-15 11:00:10 INFO rm -rf executed by unknown user",
    ]

    for log, is_anomaly, score in detector.predict(test_logs):
        status = "ANOMALO" if is_anomaly else "Normal"
        print(f"[{status}] Score: {score:.2f} - {log[:50]}...")
```

### 2. Deteccion de Secuencias de Comandos Maliciosos

```
PIPELINE DE DETECCION
=====================

Input: Secuencia de comandos de usuario

    ["cd /tmp", "wget http://x.x", "chmod +x a.sh", "./a.sh"]
                            │
                            ▼
    ┌─────────────────────────────────────────────────┐
    │              Tokenizacion                        │
    │  Normalizar URLs, IPs, paths                    │
    └─────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────┐
    │           LSTM Bidireccional                    │
    │  Captura patrones en ambas direcciones          │
    └─────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────┐
    │              Attention                          │
    │  Identifica comandos clave                      │
    └─────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────┐
    │           Clasificador                          │
    │  Benigno / Sospechoso / Malicioso              │
    └─────────────────────────────────────────────────┘
                            │
                            ▼
    Output: {clase: "malicious", confidence: 0.94,
             attention: ["wget", "chmod +x", "./a.sh"]}
```

### 3. Network Traffic Analysis

```python
class PacketSequenceClassifier(nn.Module):
    """
    Clasifica secuencias de paquetes de red.
    Input: Features extraidas de N paquetes consecutivos.
    Output: Tipo de trafico (normal, scan, ddos, exfiltration, etc.)
    """

    def __init__(
        self,
        input_features: int = 20,  # Features por paquete
        hidden_dim: int = 128,
        num_classes: int = 5
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # GRU para secuencia de paquetes
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features] - secuencia de paquetes
        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len, _ = x.shape

        # Feature extraction por paquete
        x = x.view(-1, x.size(-1))  # [batch*seq, features]
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)  # [batch, seq, 64]

        # GRU
        _, h_n = self.gru(x)  # h_n: [4, batch, hidden]

        # Concat ultimos estados forward/backward
        h_combined = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return self.classifier(h_combined)


# Features por paquete:
# - Tamano payload
# - Protocolo (one-hot)
# - Puerto destino (binned)
# - Flags TCP (bit vector)
# - Inter-arrival time
# - Direccion (in/out)
# - TTL normalizado
# - etc.
```

---

## Hiperparametros y Best Practices

### Tabla de Hiperparametros

| Hiperparametro | Rango Tipico | Consideraciones |
|----------------|--------------|-----------------|
| hidden_dim | 64-512 | Mayor = mas capacidad, mas lento |
| num_layers | 1-3 | >3 raramente ayuda, mas dificil entrenar |
| dropout | 0.2-0.5 | Entre capas, no en ultima |
| learning_rate | 1e-4 - 1e-2 | Empezar bajo para RNN |
| gradient_clip | 1.0-5.0 | Previene explosion |
| batch_size | 32-128 | Secuencias largas = batch pequeno |
| sequence_length | Depende | Truncar si es necesario |

### Checklist de Entrenamiento

```
BEST PRACTICES RNN
==================

[ ] Gradient clipping (norma 1.0-5.0)
[ ] Learning rate bajo (1e-3 o menos)
[ ] Inicializacion ortogonal para W_hh
[ ] Dropout ENTRE capas, no dentro
[ ] Pack/pad sequences correctamente
[ ] Teacher forcing ratio decreciente
[ ] Bidireccional cuando sea posible
[ ] Attention para secuencias largas

DEBUGGING:
[ ] Monitorear norma de gradientes
[ ] Verificar que loss decrece
[ ] Checkear hidden states (no NaN/Inf)
[ ] Validar longitudes de secuencia
```

### Gradient Clipping

```python
# PyTorch - despues de loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Keras/TensorFlow - en el optimizador
optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
```

---

## Limitaciones y Cuando NO Usar RNN

```
LIMITACIONES
============

1. Procesamiento Secuencial (no paralelizable):
   x_1 → x_2 → x_3 → ... → x_T
   Cada paso depende del anterior, GPU subutilizada

2. Memoria limitada a pesar de LSTM/GRU:
   En la practica, >500 tokens = problemas

3. Dificil capturar dependencias muy largas:
   Incluso LSTM tiene limite practico

4. Alternativas mas eficientes:
   - Transformers (GPT, BERT) para NLP
   - TCN (Temporal Conv) para series temporales
   - State Space Models (Mamba) para secuencias largas


CUANDO USAR RNN:
├── Recursos computacionales limitados
├── Secuencias relativamente cortas (<500)
├── Necesidad de procesamiento online/streaming
├── Datos escasos (Transformers necesitan mas datos)
└── Embeddings preentrenados no disponibles

CUANDO NO USAR RNN:
├── Secuencias muy largas (usar Transformer)
├── NLP moderno (usar BERT, GPT)
├── Datos abundantes (Transformer escala mejor)
├── Necesidad de alta paralelizacion
└── Tareas donde el orden no importa
```

---

## Resumen

| Arquitectura | Caracteristica Principal | Caso de Uso |
|--------------|-------------------------|-------------|
| RNN Vanilla | Simple, rapida | Secuencias cortas, baseline |
| LSTM | Memoria a largo plazo | Texto, logs, secuencias complejas |
| GRU | LSTM simplificada | Recursos limitados, prototipado |
| Bidireccional | Contexto completo | Clasificacion, NER, analisis offline |

### Puntos Clave

1. **RNN vanilla** sufre desvanecimiento de gradiente
2. **LSTM** resuelve esto con cell state y gates
3. **GRU** es alternativa mas simple y eficiente
4. **Bidireccional** cuando tienes secuencia completa
5. **Gradient clipping** es obligatorio
6. Para secuencias muy largas, considerar **Transformers**
