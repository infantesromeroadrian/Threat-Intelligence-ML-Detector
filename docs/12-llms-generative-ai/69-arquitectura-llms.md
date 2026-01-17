# 69. Arquitectura de Large Language Models (LLMs)

## Tabla de Contenidos

1. [Introduccion a los LLMs](#introduccion)
2. [Arquitectura Transformer](#arquitectura-transformer)
3. [Mecanismos de Atencion](#mecanismos-de-atencion)
4. [Tokenizacion](#tokenizacion)
5. [Positional Encoding](#positional-encoding)
6. [Comparativa: GPT vs BERT vs LLaMA](#comparativa-arquitecturas)
7. [Scaling Laws](#scaling-laws)
8. [Implementacion Practica](#implementacion-practica)
9. [Aplicaciones en Ciberseguridad](#aplicaciones-ciberseguridad)

---

## 1. Introduccion a los LLMs {#introduccion}

Los Large Language Models (LLMs) son modelos de deep learning entrenados en cantidades masivas de texto que pueden generar, entender y manipular lenguaje natural. La arquitectura dominante es el **Transformer**, introducido en "Attention Is All You Need" (Vaswani et al., 2017).

### Evolucion Historica

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVOLUCION DE MODELOS DE LENGUAJE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  2017 ──► 2018 ──► 2019 ──► 2020 ──► 2021 ──► 2022 ──► 2023 ──► 2024      │
│    │       │        │        │        │        │        │        │          │
│    ▼       ▼        ▼        ▼        ▼        ▼        ▼        ▼          │
│ Trans-   BERT    GPT-2    GPT-3   Codex    ChatGPT  GPT-4   LLaMA-3        │
│ former   (340M)  (1.5B)   (175B)  (12B)    (175B+)  (1.7T?) (70B+)         │
│ (65M)     ELMo            T5              Claude    LLaMA   Mixtral        │
│                   XLNet   (11B)           PaLM      2       Gemini         │
│                                           (540B)            Claude 3       │
│                                                                             │
│  Parametros:  Millones ──────────────────────────► Trillones               │
│  Capacidad:   Tareas simples ────────────────────► Razonamiento complejo   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Caracteristicas Clave de los LLMs Modernos

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class LLMCharacteristics:
    """Caracteristicas principales de un LLM moderno."""

    name: str
    parameters: int  # En billones (1B = 1,000,000,000)
    context_length: int  # Tokens maximos
    architecture: Literal["decoder-only", "encoder-only", "encoder-decoder"]
    training_data: str  # Descripcion del dataset
    capabilities: list[str]

    def compute_memory_requirements(self, precision: int = 16) -> float:
        """
        Calcula memoria aproximada en GB para inference.

        Args:
            precision: Bits por parametro (16 para fp16, 32 para fp32)

        Returns:
            Memoria aproximada en GB
        """
        bytes_per_param = precision / 8
        total_bytes = self.parameters * 1e9 * bytes_per_param
        return total_bytes / (1024 ** 3)

    def estimate_training_flops(self) -> float:
        """
        Estima FLOPs de entrenamiento usando la aproximacion 6*N*D.

        Chinchilla scaling: FLOPs ≈ 6 * N * D
        donde N = parametros, D = tokens de entrenamiento

        Returns:
            FLOPs estimados
        """
        # Asumiendo entrenamiento optimo: D ≈ 20 * N
        optimal_tokens = 20 * self.parameters * 1e9
        return 6 * self.parameters * 1e9 * optimal_tokens


# Ejemplos de LLMs modernos
GPT4 = LLMCharacteristics(
    name="GPT-4",
    parameters=1700,  # Estimado, no confirmado
    context_length=128000,
    architecture="decoder-only",
    training_data="Internet + libros + codigo + RLHF",
    capabilities=[
        "razonamiento_complejo",
        "generacion_codigo",
        "analisis_multimodal",
        "seguimiento_instrucciones"
    ]
)

LLAMA3_70B = LLMCharacteristics(
    name="LLaMA-3-70B",
    parameters=70,
    context_length=8192,
    architecture="decoder-only",
    training_data="15T tokens de texto publico",
    capabilities=[
        "generacion_texto",
        "razonamiento",
        "codigo",
        "multilingue"
    ]
)

print(f"GPT-4 memoria estimada (fp16): {GPT4.compute_memory_requirements():.1f} GB")
print(f"LLaMA-3-70B memoria estimada (fp16): {LLAMA3_70B.compute_memory_requirements():.1f} GB")
```

---

## 2. Arquitectura Transformer {#arquitectura-transformer}

El Transformer es la arquitectura base de todos los LLMs modernos. Elimina las recurrencias de RNN/LSTM usando **self-attention** para procesar secuencias en paralelo.

### Diagrama de Arquitectura Completa

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARQUITECTURA TRANSFORMER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐      ┌─────────────────────────┐              │
│  │       ENCODER           │      │        DECODER          │              │
│  │  (BERT, RoBERTa)        │      │  (GPT, LLaMA, Claude)   │              │
│  │                         │      │                         │              │
│  │  ┌───────────────────┐  │      │  ┌───────────────────┐  │              │
│  │  │  Output Encoder   │  │      │  │    Output Probs   │  │              │
│  │  └─────────┬─────────┘  │      │  └─────────┬─────────┘  │              │
│  │            │            │      │            │            │              │
│  │  ┌─────────▼─────────┐  │      │  ┌─────────▼─────────┐  │              │
│  │  │   Feed Forward    │  │      │  │   Feed Forward    │  │              │
│  │  │   + LayerNorm     │  │      │  │   + LayerNorm     │  │              │
│  │  └─────────┬─────────┘  │      │  └─────────┬─────────┘  │              │
│  │            │            │      │            │            │              │
│  │  ┌─────────▼─────────┐  │      │  ┌─────────▼─────────┐  │              │
│  │  │  Multi-Head       │  │      │  │  Cross-Attention  │◄─┼──┐           │
│  │  │  Self-Attention   │  │      │  │  (encoder-decoder)│  │  │           │
│  │  └─────────┬─────────┘  │      │  └─────────┬─────────┘  │  │           │
│  │            │            │      │            │            │  │           │
│  │  ┌─────────▼─────────┐  │      │  ┌─────────▼─────────┐  │  │           │
│  │  │  Add & Norm       │  │      │  │  Masked Self-Att  │  │  │           │
│  │  └─────────┬─────────┘  │      │  │  (causal mask)    │  │  │           │
│  │            │            │      │  └─────────┬─────────┘  │  │           │
│  │       ×N layers         │      │       ×N layers         │  │           │
│  │            │            │      │            │            │  │           │
│  │  ┌─────────▼─────────┐  │      │  ┌─────────▼─────────┐  │  │           │
│  │  │  Positional       │  │      │  │  Positional       │  │  │           │
│  │  │  Encoding         │  │      │  │  Encoding         │  │  │           │
│  │  └─────────┬─────────┘  │      │  └─────────┬─────────┘  │  │           │
│  │            │            │      │            │            │  │           │
│  │  ┌─────────▼─────────┐  │      │  ┌─────────▼─────────┐  │  │           │
│  │  │  Input Embedding  │  │      │  │  Output Embedding │  │  │           │
│  │  └─────────┬─────────┘  │──────┼─►└─────────┬─────────┘  │  │           │
│  │            │            │      │            │            │  │           │
│  │            ▲            │      │            ▲            │  │           │
│  │     [Input Tokens]      │      │     [Output Tokens]     │  │           │
│  │                         │      │     (shifted right)     │  │           │
│  └─────────────────────────┘      └─────────────────────────┘  │           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion del Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention como en 'Attention Is All You Need'.

    Permite al modelo atender a diferentes posiciones y aprender
    diferentes tipos de relaciones en paralelo.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ) -> None:
        """
        Args:
            d_model: Dimension del modelo
            n_heads: Numero de cabezas de atencion
            dropout: Tasa de dropout
            bias: Si incluir bias en proyecciones
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension por cabeza

        # Proyecciones lineales para Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)

        # Proyeccion de salida
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass de Multi-Head Attention.

        Args:
            query: Tensor de queries [batch, seq_len, d_model]
            key: Tensor de keys [batch, seq_len, d_model]
            value: Tensor de values [batch, seq_len, d_model]
            mask: Mascara de atencion opcional
            return_attention: Si retornar pesos de atencion

        Returns:
            Tuple de (output, attention_weights o None)
        """
        batch_size = query.size(0)

        # Proyecciones lineales: [batch, seq, d_model] -> [batch, seq, d_model]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Reshape para multi-head: [batch, seq, d_model] -> [batch, n_heads, seq, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        # scores: [batch, n_heads, seq_q, seq_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Aplicar mascara si existe (para causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax y dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Aplicar atencion a valores
        # context: [batch, n_heads, seq, d_k]
        context = torch.matmul(attention_weights, V)

        # Concatenar cabezas: [batch, seq, d_model]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Proyeccion final
        output = self.w_o(context)

        if return_attention:
            return output, attention_weights
        return output, None


class FeedForward(nn.Module):
    """
    Feed-Forward Network del Transformer.

    Dos capas lineales con activacion no lineal en medio.
    FFN(x) = max(0, xW1 + b1)W2 + b2

    En LLMs modernos se usa SwiGLU o GeGLU en lugar de ReLU.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ) -> None:
        """
        Args:
            d_model: Dimension del modelo
            d_ff: Dimension de la capa intermedia (tipicamente 4*d_model)
            dropout: Tasa de dropout
            activation: Tipo de activacion ('relu', 'gelu', 'swiglu')
        """
        super().__init__()

        self.activation = activation

        if activation == "swiglu":
            # SwiGLU: (Swish(xW) * xV)W2
            # Usado en LLaMA, PaLM
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        else:
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del FFN."""
        if self.activation == "swiglu":
            # SwiGLU activation
            swish = F.silu(self.w1(x))  # Swish = x * sigmoid(x)
            gate = self.w3(x)
            x = swish * gate
            x = self.w2(x)
        elif self.activation == "gelu":
            x = self.w1(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.w2(x)
        else:  # relu
            x = self.w1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.w2(x)

        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Un bloque Transformer completo (Pre-LN variant).

    Pre-LN (usado en GPT-2+):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Post-LN (original):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True
    ) -> None:
        super().__init__()

        self.pre_norm = pre_norm

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass del bloque Transformer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Mascara de atencion opcional

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        if self.pre_norm:
            # Pre-LayerNorm (GPT-2, LLaMA style)
            attn_out, _ = self.attention(
                self.norm1(x), self.norm1(x), self.norm1(x), mask
            )
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            # Post-LayerNorm (original Transformer)
            attn_out, _ = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.dropout(self.ffn(x)))

        return x
```

---

## 3. Mecanismos de Atencion {#mecanismos-de-atencion}

### Tipos de Atencion en LLMs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TIPOS DE ATENCION EN LLMs                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. FULL SELF-ATTENTION (Bidireccional)                                    │
│     ┌─────────────────────────────────┐                                     │
│     │ Token puede atender a TODOS     │  Usado en: BERT, RoBERTa           │
│     │                                 │                                     │
│     │   The  cat  sat  on   the  mat  │                                     │
│     │    ↕    ↕    ↕   ↕    ↕    ↕    │  Cada token ve todos los demas     │
│     │    ●────●────●───●────●────●    │                                     │
│     └─────────────────────────────────┘                                     │
│                                                                             │
│  2. CAUSAL/MASKED ATTENTION (Unidireccional)                               │
│     ┌─────────────────────────────────┐                                     │
│     │ Token solo atiende a anteriores │  Usado en: GPT, LLaMA, Claude      │
│     │                                 │                                     │
│     │   The  cat  sat  on   the  mat  │                                     │
│     │    │    │    │   │    │    │    │                                     │
│     │    ●    ●←───●   │    │    │    │  Mascara triangular inferior       │
│     │    │    │    │   ●←───●←───●    │                                     │
│     └─────────────────────────────────┘                                     │
│                                                                             │
│  3. CROSS-ATTENTION (Encoder-Decoder)                                      │
│     ┌─────────────────────────────────┐                                     │
│     │ Decoder atiende al Encoder      │  Usado en: T5, BART, traduccion   │
│     │                                 │                                     │
│     │   Encoder: [E1] [E2] [E3]       │                                     │
│     │              ↓    ↓    ↓        │                                     │
│     │   Decoder: [D1]──────────►[D2]  │                                     │
│     └─────────────────────────────────┘                                     │
│                                                                             │
│  4. SPARSE ATTENTION (Eficiente)                                           │
│     ┌─────────────────────────────────┐                                     │
│     │ Atencion local + global         │  Usado en: Longformer, BigBird     │
│     │                                 │                                     │
│     │   ●─●─●   ●       ●   ●─●─●    │  Complejidad: O(n) vs O(n²)        │
│     │     │     │       │     │       │                                     │
│     │     └─────●───────●─────┘       │  Tokens globales + ventana local   │
│     └─────────────────────────────────┘                                     │
│                                                                             │
│  5. GROUPED-QUERY ATTENTION (GQA)                                          │
│     ┌─────────────────────────────────┐                                     │
│     │ Queries agrupadas comparten K,V │  Usado en: LLaMA-2, Mistral        │
│     │                                 │                                     │
│     │   Q1 Q2 Q3 Q4   Q5 Q6 Q7 Q8    │                                     │
│     │     \  |  /       \  |  /       │  Reduce memoria de KV-cache        │
│     │       K1,V1        K2,V2        │                                     │
│     └─────────────────────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion de Atencion Eficiente

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) usado en LLaMA-2 y Mistral.

    En lugar de tener n_heads K y V por separado, agrupa queries
    para compartir K,V. Reduce memoria de KV-cache significativamente.

    - MHA: n_kv_heads = n_heads (tradicional)
    - MQA: n_kv_heads = 1 (Multi-Query Attention)
    - GQA: 1 < n_kv_heads < n_heads (compromiso)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,  # Numero de grupos de K,V
        dropout: float = 0.0,
        max_seq_len: int = 8192
    ) -> None:
        super().__init__()

        assert n_heads % n_kv_heads == 0, "n_heads debe ser divisible por n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # Repeticiones por grupo
        self.head_dim = d_model // n_heads

        # Proyecciones
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Pre-computar mascara causal
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repite K,V para cada grupo de queries.

        [batch, n_kv_heads, seq, head_dim] -> [batch, n_heads, seq, head_dim]
        """
        if self.n_rep == 1:
            return x

        batch, n_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, self.n_rep, seq_len, head_dim)
        return x.reshape(batch, self.n_heads, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward con soporte para KV-cache (inference incremental).

        Args:
            x: Input [batch, seq_len, d_model]
            start_pos: Posicion inicial (para KV-cache)
            kv_cache: Tuple de (cached_k, cached_v) o None

        Returns:
            Tuple de (output, (new_k_cache, new_v_cache))
        """
        batch, seq_len, _ = x.shape

        # Proyecciones
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Aplicar KV-cache si existe
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        # Repetir K,V para GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Mascara causal
        if seq_len > 1:  # Solo durante entrenamiento o primer token
            mask = self.causal_mask[start_pos:start_pos + seq_len, :start_pos + seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.wo(out), new_kv_cache


class FlashAttention(nn.Module):
    """
    Wrapper para Flash Attention (requiere flash-attn package).

    Flash Attention es un algoritmo IO-aware que:
    1. Reduce accesos a memoria HBM
    2. Computa atencion en bloques (tiling)
    3. No materializa la matriz de atencion completa
    4. Complejidad: O(N²) en FLOPS pero O(N) en memoria
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        causal: bool = True
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward usando Flash Attention si esta disponible.
        """
        batch, seq_len, d_model = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch, seq_len, 3, self.n_heads, self.head_dim)

        try:
            from flash_attn import flash_attn_qkvpacked_func

            # Flash Attention espera [batch, seq, 3, heads, head_dim]
            out = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal
            )
        except ImportError:
            # Fallback a atencion standard
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Usar F.scaled_dot_product_attention (PyTorch 2.0+)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal
            )
            out = out.transpose(1, 2)

        out = out.reshape(batch, seq_len, d_model)
        return self.out_proj(out)
```

---

## 4. Tokenizacion {#tokenizacion}

La tokenizacion convierte texto en secuencias de IDs numericos que el modelo puede procesar.

### Algoritmos de Tokenizacion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALGORITMOS DE TOKENIZACION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. BYTE-PAIR ENCODING (BPE)                                               │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │ Proceso iterativo que fusiona pares de bytes mas frecuentes │        │
│     │                                                              │        │
│     │ Texto:    "low lower lowest"                                 │        │
│     │           ↓                                                  │        │
│     │ Chars:    ['l','o','w',' ','l','o','w','e','r',...]         │        │
│     │           ↓ (fusionar pares frecuentes)                      │        │
│     │ Iter 1:   ['lo','w',' ','lo','w','e','r',...]  (l+o→lo)     │        │
│     │ Iter 2:   ['low',' ','low','e','r',...]        (lo+w→low)   │        │
│     │ Iter 3:   ['low',' ','lower','st']             (low+er→lower)│        │
│     │                                                              │        │
│     │ Usado en: GPT-2, GPT-3, RoBERTa                             │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
│  2. WORDPIECE                                                              │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │ Similar a BPE pero maximiza likelihood del corpus            │        │
│     │                                                              │        │
│     │ "playing" → ["play", "##ing"]                                │        │
│     │ "unhappy" → ["un", "##happy"]                                │        │
│     │                                                              │        │
│     │ ## indica continuacion de palabra                            │        │
│     │ Usado en: BERT, DistilBERT                                  │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
│  3. SENTENCEPIECE (Unigram LM)                                             │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │ Modelo probabilistico que opera directamente en raw text     │        │
│     │                                                              │        │
│     │ - No requiere pre-tokenizacion                              │        │
│     │ - Funciona con cualquier idioma                             │        │
│     │ - Incluye caracter especial ▁ para espacios                 │        │
│     │                                                              │        │
│     │ "Hello World" → ["▁Hello", "▁World"]                        │        │
│     │                                                              │        │
│     │ Usado en: T5, LLaMA, Mistral, XLNet                         │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
│  4. TIKTOKEN (byte-level BPE)                                              │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │ BPE optimizado de OpenAI, muy rapido                         │        │
│     │                                                              │        │
│     │ - Opera a nivel de bytes (maneja cualquier input)            │        │
│     │ - Vocabulary ~100k tokens para GPT-4                         │        │
│     │ - Encodings especificos: cl100k_base, p50k_base              │        │
│     │                                                              │        │
│     │ Usado en: GPT-3.5, GPT-4, Claude                            │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion de Tokenizadores

```python
from collections import defaultdict
from typing import Iterator
import regex as re


class SimpleBPETokenizer:
    """
    Implementacion simplificada de BPE para entender el algoritmo.

    En produccion usa: tiktoken, sentencepiece, o transformers tokenizers.
    """

    def __init__(self, vocab_size: int = 1000) -> None:
        """
        Args:
            vocab_size: Tamano maximo del vocabulario
        """
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.merges: list[tuple[str, str]] = []

        # Regex para pre-tokenizacion (similar a GPT-2)
        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def _get_stats(self, splits: list[list[str]]) -> dict[tuple[str, str], int]:
        """Cuenta frecuencia de pares adyacentes."""
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for split in splits:
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                counts[pair] += 1
        return counts

    def _merge(
        self,
        splits: list[list[str]],
        pair: tuple[str, str]
    ) -> list[list[str]]:
        """Fusiona todas las ocurrencias de un par."""
        new_splits = []
        for split in splits:
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    new_split.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits.append(new_split)
        return new_splits

    def train(self, text: str) -> None:
        """
        Entrena el tokenizador BPE en un corpus de texto.

        Args:
            text: Texto de entrenamiento
        """
        # Pre-tokenizacion
        words = self.pattern.findall(text)

        # Inicializar con caracteres
        splits = [list(word) for word in words]

        # Construir vocabulario base (caracteres unicos)
        all_chars = set()
        for split in splits:
            all_chars.update(split)

        self.vocab = {char: idx for idx, char in enumerate(sorted(all_chars))}

        # BPE merges
        while len(self.vocab) < self.vocab_size:
            stats = self._get_stats(splits)
            if not stats:
                break

            # Encontrar par mas frecuente
            best_pair = max(stats, key=stats.get)

            # Merge
            splits = self._merge(splits, best_pair)

            # Actualizar vocabulario
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)

            print(f"Merge {len(self.merges)}: {best_pair} -> '{new_token}' (freq: {stats[best_pair]})")

    def encode(self, text: str) -> list[int]:
        """Tokeniza texto a IDs."""
        words = self.pattern.findall(text)
        tokens = []

        for word in words:
            split = list(word)

            # Aplicar merges aprendidos
            for merge_pair in self.merges:
                i = 0
                while i < len(split) - 1:
                    if split[i] == merge_pair[0] and split[i + 1] == merge_pair[1]:
                        split = split[:i] + [merge_pair[0] + merge_pair[1]] + split[i + 2:]
                    else:
                        i += 1

            # Convertir a IDs
            for token in split:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # Token desconocido - usar bytes
                    for byte in token.encode('utf-8'):
                        tokens.append(self.vocab.get(chr(byte), 0))

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decodifica IDs a texto."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        return ''.join(id_to_token.get(id, '') for id in ids)


# Uso de tokenizadores profesionales
def demo_tokenizers() -> None:
    """Demuestra uso de tokenizadores profesionales."""

    text = "Hello, how are you doing today? Let's test tokenization!"

    # 1. Tiktoken (OpenAI)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)

        print(f"Tiktoken (cl100k_base):")
        print(f"  Tokens: {tokens}")
        print(f"  Num tokens: {len(tokens)}")
        print(f"  Decoded: {decoded}")
    except ImportError:
        print("tiktoken no instalado")

    # 2. Hugging Face Tokenizers
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        print(f"\nLLaMA-2 Tokenizer:")
        print(f"  Tokens: {tokens}")
        print(f"  Num tokens: {len(tokens)}")
        print(f"  Token strings: {tokenizer.convert_ids_to_tokens(tokens)}")
    except Exception as e:
        print(f"Error cargando tokenizer: {e}")

    # 3. SentencePiece directo
    try:
        import sentencepiece as spm

        # Entrenar un modelo simple (normalmente se carga pre-entrenado)
        # spm.SentencePieceTrainer.train(...)
        print("\nSentencePiece requiere modelo pre-entrenado")
    except ImportError:
        print("sentencepiece no instalado")


if __name__ == "__main__":
    # Demo de BPE simple
    bpe = SimpleBPETokenizer(vocab_size=50)
    bpe.train("the cat sat on the mat. the cat sat. the mat sat.")

    encoded = bpe.encode("the cat")
    print(f"\nEncoded 'the cat': {encoded}")
    print(f"Decoded: {bpe.decode(encoded)}")
```

---

## 5. Positional Encoding {#positional-encoding}

Los Transformers no tienen nocion inherente de posicion, necesitan positional encodings.

### Tipos de Positional Encoding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIPOS DE POSITIONAL ENCODING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SINUSOIDAL (Original Transformer)                                      │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │ PE(pos, 2i) = sin(pos / 10000^(2i/d_model))                 │        │
│     │ PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))               │        │
│     │                                                              │        │
│     │ Ventajas: Extrapolable a secuencias mas largas              │        │
│     │ Desventajas: No aprendido, puede ser suboptimo              │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
│  2. LEARNED POSITIONAL EMBEDDINGS (GPT, BERT)                              │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │ PE = nn.Embedding(max_seq_len, d_model)                     │        │
│     │                                                              │        │
│     │ Ventajas: Optimizado para la tarea                          │        │
│     │ Desventajas: Limitado a max_seq_len de entrenamiento        │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
│  3. ROTARY POSITIONAL EMBEDDING (RoPE) - LLaMA, Mistral                   │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │ Rota vectores Q,K en pares de dimensiones                   │        │
│     │                                                              │        │
│     │ q_rot = [q0*cos(mθ0) - q1*sin(mθ0),                        │        │
│     │          q0*sin(mθ0) + q1*cos(mθ0), ...]                    │        │
│     │                                                              │        │
│     │ Ventajas: Posiciones relativas, buen extrapolacion          │        │
│     │ Usado en: LLaMA, LLaMA-2, Mistral, Qwen                     │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
│  4. ALiBi (Attention with Linear Biases) - MPT, BLOOM                      │
│     ┌─────────────────────────────────────────────────────────────┐        │
│     │ Aniade bias lineal a scores de atencion                     │        │
│     │                                                              │        │
│     │ softmax(Q @ K^T - m * |i - j|)                              │        │
│     │                                                              │        │
│     │ donde m es un slope especifico por cabeza                   │        │
│     │                                                              │        │
│     │ Ventajas: Excelente extrapolacion sin entrenamiento         │        │
│     │ Desventajas: Puede perder precision en contextos largos     │        │
│     └─────────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion de RoPE

```python
import torch
import torch.nn as nn
from typing import Optional


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) de LLaMA.

    RoPE codifica posiciones rotando pares de dimensiones de Q y K,
    permitiendo que el producto punto Q*K dependa de posiciones relativas.

    Matematicamente:
    - Agrupa dimensiones en pares: (d0, d1), (d2, d3), ...
    - Rota cada par por un angulo proporcional a la posicion
    - El angulo theta_i = base^(-2i/d) donde base=10000
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0
    ) -> None:
        """
        Args:
            dim: Dimension del embedding (debe ser par)
            max_seq_len: Longitud maxima de secuencia
            base: Base para frecuencias (10000 es standard)
        """
        super().__init__()

        assert dim % 2 == 0, "Dimension debe ser par para RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-computar frecuencias: theta_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-computar cos/sin para todas las posiciones
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-computa matrices de rotacion."""
        # Posiciones: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=self.inv_freq.device)

        # Frecuencias para cada posicion: [seq_len, dim/2]
        freqs = torch.outer(positions, self.inv_freq)

        # Concatenar para obtener dim completo: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        # Guardar cos y sin
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reorganiza tensor para rotacion.

        [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        """
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica RoPE a queries y keys.

        Args:
            q: Query tensor [batch, n_heads, seq_len, head_dim]
            k: Key tensor [batch, n_heads, seq_len, head_dim]
            start_pos: Posicion inicial (para KV-cache)

        Returns:
            Tuple de (q_rotated, k_rotated)
        """
        seq_len = q.size(2)

        # Extender cache si es necesario
        if start_pos + seq_len > self.cos_cached.size(2):
            self._build_cache(start_pos + seq_len)

        # Obtener cos/sin para las posiciones relevantes
        cos = self.cos_cached[:, :, start_pos:start_pos + seq_len, :]
        sin = self.sin_cached[:, :, start_pos:start_pos + seq_len, :]

        # Aplicar rotacion: x * cos + rotate_half(x) * sin
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    En lugar de anadir posiciones al embedding, aniade un bias
    a los scores de atencion proporcional a la distancia.

    bias[i,j] = -m * |i - j|

    donde m es un slope diferente para cada cabeza de atencion.
    """

    def __init__(self, n_heads: int, max_seq_len: int = 8192) -> None:
        super().__init__()

        self.n_heads = n_heads

        # Calcular slopes (geometricamente espaciados)
        # slopes = 2^(-8/n * [1, 2, ..., n])
        ratio = 2 ** (-8 / n_heads)
        slopes = torch.tensor([ratio ** i for i in range(1, n_heads + 1)])
        self.register_buffer("slopes", slopes)

        # Pre-computar matriz de distancias
        positions = torch.arange(max_seq_len)
        distance_matrix = torch.abs(positions[:, None] - positions[None, :])
        self.register_buffer("distance_matrix", distance_matrix)

    def forward(
        self,
        attention_scores: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Aniade bias ALiBi a scores de atencion.

        Args:
            attention_scores: [batch, n_heads, seq_len, seq_len]
            seq_len: Longitud de secuencia actual

        Returns:
            Attention scores con bias ALiBi
        """
        # Obtener distancias para esta secuencia
        distances = self.distance_matrix[:seq_len, :seq_len]

        # Calcular bias: -slope * distance
        # [n_heads, 1, 1] * [seq_len, seq_len] -> [n_heads, seq_len, seq_len]
        bias = -self.slopes[:, None, None] * distances[None, :, :]

        # Anadir a scores: [batch, n_heads, seq_len, seq_len]
        return attention_scores + bias


# Demo de positional encodings
def demo_positional_encodings() -> None:
    """Demuestra diferentes esquemas de positional encoding."""

    batch_size = 2
    seq_len = 128
    n_heads = 8
    head_dim = 64

    # Input aleatorio
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    # RoPE
    rope = RotaryPositionalEmbedding(dim=head_dim)
    q_rope, k_rope = rope(q, k)

    print(f"RoPE output shape: {q_rope.shape}")
    print(f"Posiciones codificadas en la rotacion")

    # ALiBi
    alibi = ALiBi(n_heads=n_heads)
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores_with_alibi = alibi(scores, seq_len)

    print(f"\nALiBi bias shape: {alibi.distance_matrix[:seq_len, :seq_len].shape}")
    print(f"Slopes por cabeza: {alibi.slopes}")


if __name__ == "__main__":
    demo_positional_encodings()
```

---

## 6. Comparativa: GPT vs BERT vs LLaMA {#comparativa-arquitecturas}

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPARATIVA DE ARQUITECTURAS LLM                        │
├───────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   Caracteristica  │      BERT       │      GPT        │      LLaMA          │
├───────────────────┼─────────────────┼─────────────────┼─────────────────────┤
│ Arquitectura      │ Encoder-only    │ Decoder-only    │ Decoder-only        │
│                   │                 │                 │                     │
│ Pre-training      │ MLM + NSP       │ Causal LM       │ Causal LM           │
│ objective         │ (bidireccional) │ (autoregresivo) │ (autoregresivo)     │
│                   │                 │                 │                     │
│ Atencion          │ Full (bidir)    │ Causal (mask)   │ Causal + GQA        │
│                   │                 │                 │                     │
│ Tokenizador       │ WordPiece       │ BPE             │ SentencePiece       │
│                   │ (30k vocab)     │ (50k vocab)     │ (32k vocab)         │
│                   │                 │                 │                     │
│ Positional Enc    │ Learned         │ Learned         │ RoPE                │
│                   │                 │                 │                     │
│ Normalization     │ Post-LN         │ Pre-LN          │ RMSNorm (Pre)       │
│                   │                 │                 │                     │
│ Activation        │ GELU            │ GELU            │ SwiGLU              │
│                   │                 │                 │                     │
│ Context Length    │ 512             │ 2048-8192       │ 4096-128K           │
│                   │                 │                 │                     │
│ Mejor para        │ NLU, clasif,    │ Generacion,     │ Balance generacion  │
│                   │ embeddings      │ chat, codigo    │ open-source         │
│                   │                 │                 │                     │
│ Open Source       │ Si              │ No (GPT-4)      │ Si                  │
├───────────────────┴─────────────────┴─────────────────┴─────────────────────┤
│                                                                             │
│  EVOLUCION DE LLAMA:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LLaMA-1 (2023)                                                       │   │
│  │  └─► 7B, 13B, 33B, 65B params                                       │   │
│  │  └─► Pre-LN, RoPE, SwiGLU                                           │   │
│  │                                                                      │   │
│  │ LLaMA-2 (2023)                                                       │   │
│  │  └─► 7B, 13B, 70B params                                            │   │
│  │  └─► + Grouped Query Attention (GQA)                                │   │
│  │  └─► + 4K context (40% mas datos)                                   │   │
│  │  └─► + Chat fine-tuned versions                                     │   │
│  │                                                                      │   │
│  │ LLaMA-3 (2024)                                                       │   │
│  │  └─► 8B, 70B, 405B params                                           │   │
│  │  └─► + 128K context                                                 │   │
│  │  └─► + Mejor tokenizador (128K vocab)                               │   │
│  │  └─► + 15T tokens de entrenamiento                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion de Variantes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuracion para diferentes arquitecturas de LLM."""

    vocab_size: int = 32000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None  # Para GQA (LLaMA-2+)
    d_ff: int = 11008
    max_seq_len: int = 4096
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_base: float = 10000.0

    # Variantes
    use_gqa: bool = False
    use_swiglu: bool = True
    use_rmsnorm: bool = True


# Configuraciones pre-definidas
LLAMA_7B = ModelConfig(
    vocab_size=32000,
    d_model=4096,
    n_layers=32,
    n_heads=32,
    d_ff=11008,
    max_seq_len=4096,
    use_gqa=False,
    use_swiglu=True,
    use_rmsnorm=True
)

LLAMA2_7B = ModelConfig(
    vocab_size=32000,
    d_model=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,  # Full MHA en 7B
    d_ff=11008,
    max_seq_len=4096,
    use_gqa=True,
    use_swiglu=True,
    use_rmsnorm=True
)

LLAMA2_70B = ModelConfig(
    vocab_size=32000,
    d_model=8192,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,  # GQA: 8 grupos
    d_ff=28672,
    max_seq_len=4096,
    use_gqa=True,
    use_swiglu=True,
    use_rmsnorm=True
)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Usado en LLaMA en lugar de LayerNorm.
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

    Mas eficiente que LayerNorm (no calcula media).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation usado en LLaMA.

    SwiGLU(x) = Swish(xW1) * xW3
    donde Swish(x) = x * sigmoid(x)

    Mejor rendimiento que GELU/ReLU en LLMs.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = F.silu(self.w1(x))  # Swish = SiLU
        gate = self.w3(x)
        return self.w2(swish * gate)


class LLaMABlock(nn.Module):
    """
    Bloque Transformer estilo LLaMA.

    Diferencias con Transformer original:
    - RMSNorm en lugar de LayerNorm
    - SwiGLU en lugar de GELU FFN
    - RoPE en lugar de learned positions
    - Pre-normalization
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # Attention
        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads or config.n_heads,
            max_seq_len=config.max_seq_len
        )

        # Feed-forward
        if config.use_swiglu:
            self.ffn = SwiGLU(config.d_model, config.d_ff)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Linear(config.d_ff, config.d_model)
            )

        # Normalization
        if config.use_rmsnorm:
            self.norm1 = RMSNorm(config.d_model, config.norm_eps)
            self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        else:
            self.norm1 = nn.LayerNorm(config.d_model, config.norm_eps)
            self.norm2 = nn.LayerNorm(config.d_model, config.norm_eps)

        # RoPE
        self.rope = RotaryPositionalEmbedding(
            dim=config.d_model // config.n_heads,
            max_seq_len=config.max_seq_len,
            base=config.rope_base
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass con soporte para KV-cache."""

        # Attention con pre-norm
        residual = x
        x = self.norm1(x)
        x, new_cache = self.attention(x, start_pos, kv_cache)
        x = residual + x

        # FFN con pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, new_cache


class LLaMAModel(nn.Module):
    """
    Modelo LLaMA completo.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            LLaMABlock(config, i) for i in range(config.n_layers)
        ])

        # Output normalization
        if config.use_rmsnorm:
            self.norm = RMSNorm(config.d_model, config.norm_eps)
        else:
            self.norm = nn.LayerNorm(config.d_model, config.norm_eps)

        # LM head (tied con embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Inicializacion
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializacion de pesos al estilo LLaMA."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass del modelo.

        Args:
            input_ids: Token IDs [batch, seq_len]
            start_pos: Posicion inicial para KV-cache
            kv_caches: Lista de caches por capa

        Returns:
            Tuple de (logits, new_kv_caches)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.embed_tokens(input_ids)

        # Preparar caches
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        new_caches = []

        # Transformer layers
        for i, layer in enumerate(self.layers):
            x, cache = layer(x, start_pos, kv_caches[i])
            new_caches.append(cache)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_caches

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Genera texto autoregressivamente.

        Args:
            prompt_tokens: Tokens del prompt [1, prompt_len]
            max_new_tokens: Maximo de tokens a generar
            temperature: Temperatura para sampling
            top_p: Nucleus sampling threshold

        Returns:
            Tensor con todos los tokens generados
        """
        batch_size = prompt_tokens.size(0)
        device = prompt_tokens.device

        # Procesar prompt completo
        logits, kv_caches = self(prompt_tokens)

        generated = prompt_tokens.clone()

        for i in range(max_new_tokens):
            # Obtener logits del ultimo token
            next_token_logits = logits[:, -1, :] / temperature

            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remover tokens fuera del nucleus
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Forward con KV-cache (solo nuevo token)
            logits, kv_caches = self(
                next_token,
                start_pos=generated.size(1) - 1,
                kv_caches=kv_caches
            )

        return generated
```

---

## 7. Scaling Laws {#scaling-laws}

Las Scaling Laws describen como el rendimiento de los LLMs escala con compute, datos y parametros.

### Leyes de Escalado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCALING LAWS PARA LLMs                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KAPLAN ET AL. (2020) - OpenAI Scaling Laws:                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Loss ∝ N^(-0.076)  (parametros)                                    │   │
│  │  Loss ∝ D^(-0.095)  (datos/tokens)                                  │   │
│  │  Loss ∝ C^(-0.050)  (compute)                                       │   │
│  │                                                                      │   │
│  │  El loss disminuye como power law con cada factor                   │   │
│  │  10x mas parametros → ~1.8x mejor loss                              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  HOFFMANN ET AL. (2022) - Chinchilla Scaling:                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Para compute optimo: D ≈ 20 * N                                    │   │
│  │                                                                      │   │
│  │  Donde:                                                              │   │
│  │    D = numero de tokens de entrenamiento                            │   │
│  │    N = numero de parametros                                         │   │
│  │                                                                      │   │
│  │  Implicacion: La mayoria de modelos estaban under-trained           │   │
│  │                                                                      │   │
│  │  Chinchilla (70B) > Gopher (280B) con mismo compute                 │   │
│  │  porque Chinchilla uso mas datos (1.4T tokens vs 300B)              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  COMPUTE OPTIMAL FRONTIER:                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Loss │                                                              │   │
│  │       │╲                                                             │   │
│  │       │ ╲                                                            │   │
│  │       │  ╲   Under-parameterized                                    │   │
│  │       │   ╲  (mas datos que params)                                 │   │
│  │       │    ╲                                                         │   │
│  │       │     ●────────● Optimal                                      │   │
│  │       │              ╲                                               │   │
│  │       │               ╲ Over-parameterized                          │   │
│  │       │                ╲(mas params que datos)                      │   │
│  │       └──────────────────────────────── Compute                     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  FLOPS ESTIMATION:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  Forward pass:  ~2 * N FLOPs per token                              │   │
│  │  Backward pass: ~4 * N FLOPs per token                              │   │
│  │  Total training: ~6 * N * D FLOPs                                   │   │
│  │                                                                      │   │
│  │  Ejemplo: LLaMA-70B con 1.4T tokens                                 │   │
│  │    = 6 * 70e9 * 1.4e12                                              │   │
│  │    = 5.88e23 FLOPs                                                  │   │
│  │    ≈ 588 ZFLOPs                                                     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Calculadora de Scaling

```python
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScalingEstimates:
    """Estimaciones basadas en scaling laws."""

    parameters: int
    tokens: int
    flops: float
    loss_estimate: float
    gpu_hours: float
    cost_estimate: float


def compute_optimal_training(
    compute_budget_flops: float,
    a: float = 0.34,  # Chinchilla constant for N
    b: float = 0.28   # Chinchilla constant for D
) -> tuple[int, int]:
    """
    Calcula parametros y tokens optimos dado un budget de compute.

    Chinchilla optimal:
    N_opt ∝ C^a
    D_opt ∝ C^b

    Args:
        compute_budget_flops: FLOPs disponibles
        a: Exponente para parametros (0.34 para Chinchilla)
        b: Exponente para datos (0.28 para Chinchilla)

    Returns:
        Tuple de (parametros_optimos, tokens_optimos)
    """
    # Normalizacion basada en Chinchilla
    # N = k1 * C^0.5, D = k2 * C^0.5 con D/N ≈ 20

    # Aproximacion practica: N = sqrt(C/6/20), D = 20*N
    n_opt = int(math.sqrt(compute_budget_flops / 6 / 20))
    d_opt = 20 * n_opt

    return n_opt, d_opt


def estimate_training_requirements(
    parameters: int,
    tokens: int,
    gpu_tflops: float = 312.0,  # A100 FP16 peak
    efficiency: float = 0.4,    # Typical training efficiency
    gpu_hour_cost: float = 2.0  # USD per GPU-hour
) -> ScalingEstimates:
    """
    Estima requisitos de entrenamiento para un LLM.

    Args:
        parameters: Numero de parametros
        tokens: Tokens de entrenamiento
        gpu_tflops: TFLOPS de la GPU
        efficiency: Eficiencia de utilizacion (0-1)
        gpu_hour_cost: Costo por GPU-hora en USD

    Returns:
        ScalingEstimates con todas las metricas
    """
    # FLOPs totales: 6 * N * D
    total_flops = 6 * parameters * tokens

    # Estimate loss using Chinchilla formula
    # L(N,D) = E + A/N^α + B/D^β
    # Simplificado: L ≈ 1.69 + 406.4/N^0.34 + 410.7/D^0.28
    E = 1.69
    A, alpha = 406.4, 0.34
    B, beta = 410.7, 0.28
    loss_estimate = E + A / (parameters ** alpha) + B / (tokens ** beta)

    # GPU hours
    effective_tflops = gpu_tflops * efficiency * 1e12  # FLOPS
    seconds = total_flops / effective_tflops
    gpu_hours = seconds / 3600

    # Cost
    cost = gpu_hours * gpu_hour_cost

    return ScalingEstimates(
        parameters=parameters,
        tokens=tokens,
        flops=total_flops,
        loss_estimate=loss_estimate,
        gpu_hours=gpu_hours,
        cost_estimate=cost
    )


def compare_scaling_strategies() -> None:
    """Compara diferentes estrategias de scaling."""

    print("=" * 70)
    print("COMPARATIVA DE ESTRATEGIAS DE SCALING")
    print("=" * 70)

    # Compute budget: ~10^24 FLOPs (aproximadamente GPT-3 training)
    compute_budget = 3.14e23

    strategies = [
        ("Over-parameterized (GPT-3 style)", 175e9, 300e9),
        ("Chinchilla optimal", *compute_optimal_training(compute_budget)),
        ("Under-parameterized", 10e9, 2e12),
    ]

    print(f"\nCompute budget: {compute_budget:.2e} FLOPs\n")

    for name, params, tokens in strategies:
        estimates = estimate_training_requirements(int(params), int(tokens))

        print(f"{name}:")
        print(f"  Parameters: {params/1e9:.1f}B")
        print(f"  Tokens: {tokens/1e12:.2f}T")
        print(f"  FLOPs: {estimates.flops:.2e}")
        print(f"  Estimated Loss: {estimates.loss_estimate:.3f}")
        print(f"  GPU Hours (A100): {estimates.gpu_hours:,.0f}")
        print(f"  Estimated Cost: ${estimates.cost_estimate:,.0f}")
        print(f"  Tokens/Param ratio: {tokens/params:.1f}")
        print()


def emergent_abilities() -> None:
    """
    Documenta habilidades emergentes en LLMs.

    Habilidades emergentes: capacidades que aparecen solo
    cuando el modelo supera cierto umbral de escala.
    """

    abilities = {
        "Chain-of-thought": {
            "threshold": "~100B params",
            "description": "Razonamiento paso a paso mejora significativamente",
            "example": "Resolver problemas matematicos complejos"
        },
        "In-context learning": {
            "threshold": "~1B params",
            "description": "Aprender de ejemplos en el prompt sin fine-tuning",
            "example": "Few-shot classification"
        },
        "Code generation": {
            "threshold": "~10B params",
            "description": "Generar codigo funcional",
            "example": "Resolver problemas de programacion"
        },
        "Multi-step reasoning": {
            "threshold": "~50B params",
            "description": "Resolver tareas que requieren multiples pasos logicos",
            "example": "Word problems, planning"
        },
        "Theory of Mind": {
            "threshold": "~100B params",
            "description": "Entender estados mentales de otros",
            "example": "False-belief tasks"
        }
    }

    print("=" * 70)
    print("HABILIDADES EMERGENTES EN LLMs")
    print("=" * 70)

    for ability, info in abilities.items():
        print(f"\n{ability}")
        print(f"  Threshold: {info['threshold']}")
        print(f"  {info['description']}")
        print(f"  Ejemplo: {info['example']}")


if __name__ == "__main__":
    compare_scaling_strategies()
    print("\n")
    emergent_abilities()
```

---

## 8. Implementacion Practica {#implementacion-practica}

### Cargando y Usando LLMs con Hugging Face

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)
from typing import Optional, Iterator
import time


class LLMInference:
    """
    Wrapper para inference con LLMs de Hugging Face.

    Soporta:
    - Carga con quantizacion (4-bit, 8-bit)
    - Generacion con streaming
    - Batch inference
    """

    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,  # "4bit", "8bit", None
        device: str = "auto",
        max_memory: Optional[dict] = None
    ) -> None:
        """
        Args:
            model_name: Nombre del modelo en HuggingFace
            quantization: Tipo de quantizacion
            device: Dispositivo ("auto", "cuda", "cpu")
            max_memory: Dict de memoria maxima por dispositivo
        """
        self.model_name = model_name

        # Configurar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configurar quantizacion
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Cargar modelo
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            max_memory=max_memory,
            torch_dtype=torch.bfloat16 if quantization is None else None,
            attn_implementation="flash_attention_2"  # Si disponible
        )

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """
        Genera texto dado un prompt.

        Args:
            prompt: Texto de entrada
            max_new_tokens: Maximo de tokens a generar
            temperature: Temperatura de sampling
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: Penalizacion por repeticion
            do_sample: Si usar sampling (False = greedy)

        Returns:
            Texto generado (sin el prompt)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config
        )

        # Decodificar solo los tokens nuevos
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Iterator[str]:
        """
        Genera texto con streaming (token por token).

        Yields:
            Tokens generados uno por uno
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            streamer=streamer
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token

    def batch_generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> list[str]:
        """
        Genera para multiples prompts en batch.

        Args:
            prompts: Lista de prompts
            max_new_tokens: Tokens maximos por generacion
            temperature: Temperatura de sampling

        Returns:
            Lista de textos generados
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Decodificar cada secuencia
        generated = []
        for i, output in enumerate(outputs):
            prompt_len = inputs['input_ids'][i].shape[0]
            generated_tokens = output[prompt_len:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated.append(text)

        return generated


def benchmark_inference(
    model: LLMInference,
    prompt: str,
    num_tokens: int = 100,
    num_runs: int = 5
) -> dict:
    """
    Benchmark de velocidad de inference.

    Returns:
        Dict con metricas de performance
    """
    times = []
    tokens_generated = []

    for _ in range(num_runs):
        start = time.perf_counter()
        output = model.generate(prompt, max_new_tokens=num_tokens)
        elapsed = time.perf_counter() - start

        num_output_tokens = len(model.tokenizer.encode(output))
        times.append(elapsed)
        tokens_generated.append(num_output_tokens)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)

    return {
        "avg_time_seconds": avg_time,
        "avg_tokens_generated": avg_tokens,
        "tokens_per_second": avg_tokens / avg_time,
        "time_to_first_token": times[0] / tokens_generated[0] if tokens_generated[0] > 0 else 0
    }


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar modelo con quantizacion 4-bit
    llm = LLMInference(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        quantization="4bit"
    )

    prompt = """<s>[INST] Explain how transformers work in simple terms. [/INST]"""

    # Generacion normal
    print("Generando respuesta...")
    response = llm.generate(prompt, max_new_tokens=200)
    print(response)

    # Generacion con streaming
    print("\nStreaming:")
    for token in llm.generate_stream(prompt, max_new_tokens=50):
        print(token, end="", flush=True)
    print()

    # Benchmark
    print("\nBenchmark:")
    metrics = benchmark_inference(llm, prompt, num_tokens=50, num_runs=3)
    print(f"Tokens/segundo: {metrics['tokens_per_second']:.1f}")
```

---

## 9. Aplicaciones en Ciberseguridad {#aplicaciones-ciberseguridad}

### LLMs para Analisis de Seguridad

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re


class ThreatCategory(Enum):
    """Categorias de amenazas detectables por LLM."""
    PHISHING = "phishing"
    MALWARE = "malware"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    SENSITIVE_DATA = "sensitive_data"
    SOCIAL_ENGINEERING = "social_engineering"


@dataclass
class SecurityAnalysis:
    """Resultado de analisis de seguridad con LLM."""

    threat_detected: bool
    threat_category: Optional[ThreatCategory]
    confidence: float
    explanation: str
    indicators: list[str]
    recommendations: list[str]


class SecurityAnalyzerLLM:
    """
    Analizador de seguridad basado en LLM.

    Uso en ciberseguridad:
    1. Analisis de emails/phishing
    2. Deteccion de codigo malicioso
    3. Analisis de logs de seguridad
    4. Generacion de reglas de deteccion
    """

    def __init__(self, llm: "LLMInference") -> None:
        self.llm = llm

        self.system_prompt = """You are a cybersecurity expert AI assistant.
Analyze the provided content for security threats.
Be thorough but avoid false positives.
Always explain your reasoning."""

    def analyze_email(self, email_content: str) -> SecurityAnalysis:
        """
        Analiza un email en busca de indicadores de phishing.

        Args:
            email_content: Contenido completo del email

        Returns:
            SecurityAnalysis con el resultado
        """
        prompt = f"""{self.system_prompt}

Analyze this email for phishing indicators:

---
{email_content}
---

Provide your analysis in this exact format:
THREAT_DETECTED: [YES/NO]
CATEGORY: [phishing/social_engineering/none]
CONFIDENCE: [0.0-1.0]
INDICATORS:
- [indicator 1]
- [indicator 2]
EXPLANATION: [brief explanation]
RECOMMENDATIONS:
- [recommendation 1]
- [recommendation 2]"""

        response = self.llm.generate(prompt, temperature=0.1)
        return self._parse_analysis(response)

    def analyze_code_snippet(
        self,
        code: str,
        language: str = "python"
    ) -> SecurityAnalysis:
        """
        Analiza codigo en busca de vulnerabilidades.

        Args:
            code: Codigo fuente a analizar
            language: Lenguaje de programacion

        Returns:
            SecurityAnalysis con vulnerabilidades encontradas
        """
        prompt = f"""{self.system_prompt}

Analyze this {language} code for security vulnerabilities:

```{language}
{code}
```

Look for:
- SQL injection
- Command injection
- XSS
- Path traversal
- Hardcoded secrets
- Insecure deserialization
- Buffer overflows

Provide your analysis in this exact format:
THREAT_DETECTED: [YES/NO]
CATEGORY: [vulnerability type or none]
CONFIDENCE: [0.0-1.0]
INDICATORS:
- [line X: issue description]
EXPLANATION: [brief explanation]
RECOMMENDATIONS:
- [fix recommendation]"""

        response = self.llm.generate(prompt, temperature=0.1)
        return self._parse_analysis(response)

    def analyze_log_entry(self, log_entry: str) -> SecurityAnalysis:
        """
        Analiza entrada de log en busca de actividad maliciosa.

        Args:
            log_entry: Linea o bloque de log

        Returns:
            SecurityAnalysis con el resultado
        """
        prompt = f"""{self.system_prompt}

Analyze this log entry for suspicious activity:

{log_entry}

Look for:
- Brute force attempts
- Privilege escalation
- Data exfiltration
- Lateral movement
- Command and control communication
- Anomalous access patterns

Provide your analysis in this exact format:
THREAT_DETECTED: [YES/NO]
CATEGORY: [attack type or none]
CONFIDENCE: [0.0-1.0]
INDICATORS:
- [indicator 1]
EXPLANATION: [brief explanation]
RECOMMENDATIONS:
- [recommendation]"""

        response = self.llm.generate(prompt, temperature=0.1)
        return self._parse_analysis(response)

    def generate_detection_rule(
        self,
        threat_description: str,
        rule_format: str = "sigma"
    ) -> str:
        """
        Genera regla de deteccion basada en descripcion de amenaza.

        Args:
            threat_description: Descripcion de la amenaza
            rule_format: Formato de regla (sigma, yara, snort)

        Returns:
            Regla de deteccion en el formato especificado
        """
        format_examples = {
            "sigma": """title: Example Rule
status: experimental
logsource:
    category: process_creation
    product: windows
detection:
    selection:
        CommandLine|contains: 'suspicious'
    condition: selection""",

            "yara": """rule example_rule {
    meta:
        description = "Example"
    strings:
        $s1 = "suspicious" ascii
    condition:
        $s1
}""",

            "snort": """alert tcp any any -> any any (msg:"Example"; content:"suspicious"; sid:1000001;)"""
        }

        prompt = f"""{self.system_prompt}

Generate a {rule_format.upper()} detection rule for the following threat:

{threat_description}

Example {rule_format} format:
```
{format_examples.get(rule_format, format_examples['sigma'])}
```

Generate a complete, functional rule:"""

        return self.llm.generate(prompt, temperature=0.3)

    def _parse_analysis(self, response: str) -> SecurityAnalysis:
        """Parsea la respuesta del LLM a SecurityAnalysis."""

        # Extraer campos con regex
        threat_match = re.search(r"THREAT_DETECTED:\s*(YES|NO)", response, re.I)
        category_match = re.search(r"CATEGORY:\s*(\w+)", response, re.I)
        confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response)

        # Extraer listas
        indicators = re.findall(r"^-\s*(.+)$", response, re.MULTILINE)

        # Extraer explicacion
        explanation_match = re.search(
            r"EXPLANATION:\s*(.+?)(?=RECOMMENDATIONS:|$)",
            response,
            re.DOTALL
        )

        threat_detected = threat_match and threat_match.group(1).upper() == "YES"

        category = None
        if category_match:
            cat_str = category_match.group(1).lower()
            for cat in ThreatCategory:
                if cat.value in cat_str:
                    category = cat
                    break

        confidence = 0.0
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except ValueError:
                pass

        explanation = ""
        if explanation_match:
            explanation = explanation_match.group(1).strip()

        return SecurityAnalysis(
            threat_detected=threat_detected,
            threat_category=category,
            confidence=confidence,
            explanation=explanation,
            indicators=indicators[:5],  # Top 5
            recommendations=indicators[5:10] if len(indicators) > 5 else []
        )


# Ejemplo de uso para ciberseguridad
def demo_security_analysis() -> None:
    """Demuestra analisis de seguridad con LLM."""

    # Ejemplo de email sospechoso
    suspicious_email = """
    From: security@bankk-of-america.com
    Subject: URGENT: Your account has been compromised!

    Dear Valued Customer,

    We have detected unusual activity in your account.
    Click here immediately to verify your identity:
    http://secure-boa-verify.xyz/login

    If you don't act within 24 hours, your account will be suspended.

    Best regards,
    Bank Security Team
    """

    # Ejemplo de codigo vulnerable
    vulnerable_code = """
    def get_user(user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        cursor.execute(query)
        return cursor.fetchone()

    def run_command(cmd):
        import os
        os.system(cmd)  # User input passed directly
    """

    # Ejemplo de log sospechoso
    suspicious_log = """
    2024-01-15 03:42:18 Failed login attempt for user 'admin' from 192.168.1.100
    2024-01-15 03:42:19 Failed login attempt for user 'admin' from 192.168.1.100
    2024-01-15 03:42:20 Failed login attempt for user 'admin' from 192.168.1.100
    2024-01-15 03:42:21 Successful login for user 'admin' from 192.168.1.100
    2024-01-15 03:42:25 User 'admin' executed: whoami
    2024-01-15 03:42:28 User 'admin' executed: cat /etc/shadow
    """

    print("=" * 70)
    print("DEMO: Analisis de Seguridad con LLM")
    print("=" * 70)

    print("\n1. Email sospechoso:")
    print(f"   Indicadores tipicos de phishing:")
    print(f"   - Dominio similar pero no identico (bankk-of-america)")
    print(f"   - Urgencia artificial")
    print(f"   - URL sospechosa (.xyz)")
    print(f"   - Amenaza de suspension")

    print("\n2. Codigo vulnerable:")
    print(f"   - SQL Injection en linea 2")
    print(f"   - Command Injection en linea 6")

    print("\n3. Log sospechoso:")
    print(f"   - Brute force (multiples intentos fallidos)")
    print(f"   - Comportamiento post-compromise (whoami, /etc/shadow)")


if __name__ == "__main__":
    demo_security_analysis()
```

---

## Resumen

Esta seccion ha cubierto la arquitectura fundamental de los LLMs:

1. **Arquitectura Transformer**: Self-attention, feed-forward networks, normalization
2. **Mecanismos de Atencion**: Full, causal, cross-attention, GQA, Flash Attention
3. **Tokenizacion**: BPE, WordPiece, SentencePiece, tiktoken
4. **Positional Encoding**: Sinusoidal, learned, RoPE, ALiBi
5. **Comparativa de Arquitecturas**: GPT vs BERT vs LLaMA
6. **Scaling Laws**: Chinchilla, compute-optimal training
7. **Aplicaciones en Ciberseguridad**: Analisis de amenazas, deteccion

### Recursos Adicionales

- Paper: "Attention Is All You Need" (Vaswani et al., 2017)
- Paper: "Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020)
- Paper: "Training Compute-Optimal LLMs" (Chinchilla, Hoffmann et al., 2022)
- Paper: "LLaMA: Open and Efficient LLMs" (Touvron et al., 2023)
- Codigo: https://github.com/huggingface/transformers
- Codigo: https://github.com/facebookresearch/llama

---

## Ejercicios Propuestos

1. **Implementar BPE desde cero**: Entrenar un tokenizador BPE en un corpus pequeno
2. **Visualizar atencion**: Implementar visualizacion de attention weights
3. **Comparar RoPE vs ALiBi**: Medir extrapolacion a secuencias largas
4. **Scaling experiment**: Entrenar modelos de diferentes tamanos y verificar scaling laws
5. **Security application**: Implementar detector de phishing usando LLM

---

*Siguiente: [70. Fine-tuning de LLMs](./70-fine-tuning-llms.md)*
