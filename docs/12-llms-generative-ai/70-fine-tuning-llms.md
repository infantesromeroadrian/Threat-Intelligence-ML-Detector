# 70. Fine-tuning de Large Language Models

## Tabla de Contenidos

1. [Introduccion al Fine-tuning](#introduccion)
2. [Full Fine-tuning](#full-fine-tuning)
3. [Parameter-Efficient Fine-Tuning (PEFT)](#peft)
4. [LoRA: Low-Rank Adaptation](#lora)
5. [QLoRA: Quantized LoRA](#qlora)
6. [Otras Tecnicas PEFT](#otras-peft)
7. [Cuando Usar Cada Tecnica](#cuando-usar)
8. [Implementacion Practica con Hugging Face](#implementacion)
9. [Datasets para Fine-tuning](#datasets)
10. [Evaluacion y Mejores Practicas](#evaluacion)
11. [Aplicaciones en Ciberseguridad](#ciberseguridad)

---

## 1. Introduccion al Fine-tuning {#introduccion}

El fine-tuning adapta un modelo pre-entrenado a tareas especificas o dominios particulares. Es el proceso de continuar el entrenamiento de un LLM en datos especializados.

### Por Que Fine-tuning?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    JERARQUIA DE ADAPTACION DE LLMs                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Menos adaptacion ◄─────────────────────────────────► Mas adaptacion        │
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ Zero-    │   │ Few-     │   │   RAG    │   │  PEFT/   │   │   Full   │  │
│  │ Shot     │   │ Shot     │   │          │   │  LoRA    │   │Fine-tune │  │
│  │ Prompting│   │ Prompting│   │          │   │          │   │          │  │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘  │
│       │              │              │              │              │         │
│   No training    No training   No training     Minimal        Full         │
│   Solo prompt    Ejemplos en   + Retrieval     training       training     │
│                  contexto                       (<1% params)   (100%)      │
│                                                                             │
│  Costo:        Bajo ────────────────────────────────────────► Alto         │
│  Personalizacion: Bajo ──────────────────────────────────────► Alto        │
│  Datos necesarios: 0 ─────────────────────────────────────────► Miles      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CUANDO USAR FINE-TUNING:                                                  │
│                                                                             │
│  ✓ Dominio muy especializado (medicina, legal, seguridad)                  │
│  ✓ Formato de salida muy especifico                                        │
│  ✓ Comportamiento consistente necesario                                    │
│  ✓ Reducir latencia (vs RAG con retrieval)                                 │
│  ✓ Datos propietarios que no puedes enviar a API externa                   │
│                                                                             │
│  CUANDO NO USAR FINE-TUNING:                                               │
│                                                                             │
│  ✗ Pocos ejemplos (<100)                                                   │
│  ✗ Conocimiento que cambia frecuentemente (usar RAG)                       │
│  ✗ Tarea ya bien resuelta por modelo base                                  │
│  ✗ Sin GPU/recursos de entrenamiento                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tipos de Fine-tuning

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FineTuningType(Enum):
    """Tipos de fine-tuning disponibles."""

    FULL = "full"              # Todos los parametros
    LORA = "lora"              # Low-Rank Adaptation
    QLORA = "qlora"            # Quantized LoRA
    PREFIX_TUNING = "prefix"   # Solo prefijos
    PROMPT_TUNING = "prompt"   # Solo embeddings de prompt
    ADAPTER = "adapter"        # Modulos adapter
    IA3 = "ia3"                # Infused Adapter by Inhibiting


@dataclass
class FineTuningConfig:
    """Configuracion para fine-tuning."""

    # Tipo de fine-tuning
    method: FineTuningType

    # Parametros de entrenamiento
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # LoRA especifico
    lora_r: int = 16              # Rank
    lora_alpha: int = 32          # Scaling factor
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    # Quantizacion (QLoRA)
    use_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Optimizaciones
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    bf16: bool = True

    def estimated_vram_gb(self, model_params_b: float) -> dict[str, float]:
        """
        Estima VRAM necesaria segun metodo.

        Args:
            model_params_b: Parametros del modelo en billones

        Returns:
            Dict con estimaciones de VRAM
        """
        # Base: 2 bytes por param en bf16
        base_vram = model_params_b * 2

        estimates = {
            "full_fp16": base_vram * 4,      # model + gradients + optimizer
            "full_bf16": base_vram * 4,
            "lora_bf16": base_vram + 0.5,    # model frozen + adapters
            "qlora_4bit": base_vram * 0.5 + 0.5,  # 4-bit model + adapters
        }

        if self.method == FineTuningType.FULL:
            return {"estimated_vram_gb": estimates["full_bf16"]}
        elif self.method == FineTuningType.LORA:
            return {"estimated_vram_gb": estimates["lora_bf16"]}
        elif self.method == FineTuningType.QLORA:
            return {"estimated_vram_gb": estimates["qlora_4bit"]}

        return estimates


# Configuraciones recomendadas
CONFIGS = {
    "7b_full": FineTuningConfig(
        method=FineTuningType.FULL,
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True
    ),
    "7b_lora": FineTuningConfig(
        method=FineTuningType.LORA,
        learning_rate=2e-4,
        batch_size=8,
        lora_r=16,
        lora_alpha=32
    ),
    "7b_qlora": FineTuningConfig(
        method=FineTuningType.QLORA,
        learning_rate=2e-4,
        batch_size=4,
        use_4bit=True,
        lora_r=64,
        lora_alpha=16
    ),
    "70b_qlora": FineTuningConfig(
        method=FineTuningType.QLORA,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation_steps=16,
        use_4bit=True,
        lora_r=64,
        lora_alpha=16,
        gradient_checkpointing=True
    )
}
```

---

## 2. Full Fine-tuning {#full-fine-tuning}

Full fine-tuning actualiza **todos** los parametros del modelo. Es el mas expresivo pero requiere mas recursos.

### Diagrama de Full Fine-tuning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FULL FINE-TUNING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    MODELO PRE-ENTRENADO                         │      │
│   │                                                                 │      │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐     ┌──────────┐      │      │
│   │   │Embedding │ │ Layer 1  │ │ Layer 2  │ ... │ Layer N  │      │      │
│   │   │  Matrix  │ │          │ │          │     │          │      │      │
│   │   │  (ALL)   │ │  (ALL)   │ │  (ALL)   │     │  (ALL)   │      │      │
│   │   └────┬─────┘ └────┬─────┘ └────┬─────┘     └────┬─────┘      │      │
│   │        │            │            │                │             │      │
│   │   ┌────▼────────────▼────────────▼────────────────▼─────┐      │      │
│   │   │           TODOS LOS PARAMETROS SE ACTUALIZAN        │      │      │
│   │   │              (backprop a traves de todo)            │      │      │
│   │   └─────────────────────────────────────────────────────┘      │      │
│   │                                                                 │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   VENTAJAS:                           DESVENTAJAS:                         │
│   ✓ Maxima expresividad              ✗ Requiere mucha VRAM                 │
│   ✓ Mejor rendimiento potencial      ✗ Riesgo de catastrophic forgetting   │
│   ✓ Sin limitaciones de adaptacion   ✗ Necesita mas datos                  │
│                                       ✗ Un modelo por tarea                 │
│                                                                             │
│   VRAM REQUERIDA (aproximada):                                             │
│   ┌─────────────────────────────────────────────────────────┐              │
│   │  Modelo   │ Params │ FP32    │ FP16/BF16 │ + Optimizer  │              │
│   │───────────┼────────┼─────────┼───────────┼──────────────│              │
│   │  7B       │ 7B     │ 28 GB   │ 14 GB     │ ~56 GB       │              │
│   │  13B      │ 13B    │ 52 GB   │ 26 GB     │ ~104 GB      │              │
│   │  70B      │ 70B    │ 280 GB  │ 140 GB    │ ~560 GB      │              │
│   └─────────────────────────────────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion de Full Fine-tuning

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import Optional
import json


class InstructionDataset(Dataset):
    """Dataset para fine-tuning de instrucciones."""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048
    ) -> None:
        """
        Args:
            data_path: Path a JSON con formato instruction/input/output
            tokenizer: Tokenizador del modelo
            max_length: Longitud maxima de secuencia
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        # Formato de prompt (Alpaca style)
        if item.get("input"):
            prompt = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}"""
        else:
            prompt = f"""### Instruction:
{item['instruction']}

### Response:
{item['output']}"""

        # Tokenizar
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }


def full_finetune(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 8,
    use_gradient_checkpointing: bool = True
) -> None:
    """
    Ejecuta full fine-tuning de un modelo.

    Args:
        model_name: Nombre del modelo en HuggingFace
        train_data_path: Path a datos de entrenamiento
        output_dir: Directorio de salida
        num_epochs: Numero de epochs
        batch_size: Tamano de batch por GPU
        learning_rate: Tasa de aprendizaje
        gradient_accumulation_steps: Pasos de acumulacion
        use_gradient_checkpointing: Activar gradient checkpointing
    """
    # Cargar tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Preparar dataset
    train_dataset = InstructionDataset(
        train_data_path,
        tokenizer,
        max_length=2048
    )

    # Configurar entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=use_gradient_checkpointing,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        report_to="tensorboard"
    )

    # Data collator para language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, no masked
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # Entrenar
    trainer.train()

    # Guardar modelo final
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Modelo guardado en {output_dir}")
```

---

## 3. Parameter-Efficient Fine-Tuning (PEFT) {#peft}

PEFT engloba tecnicas que actualizan solo una pequena fraccion de parametros.

### Comparativa de Metodos PEFT

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPARATIVA DE METODOS PEFT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Metodo          │ Params      │ VRAM      │ Rendimiento │ Complejidad     │
│  ────────────────┼─────────────┼───────────┼─────────────┼─────────────────│
│  Full Fine-tune  │ 100%        │ Muy alta  │ ★★★★★       │ Baja            │
│  LoRA            │ 0.1-1%      │ Media     │ ★★★★☆       │ Baja            │
│  QLoRA           │ 0.1-1%      │ Baja      │ ★★★★☆       │ Media           │
│  Prefix Tuning   │ 0.01-0.1%   │ Baja      │ ★★★☆☆       │ Baja            │
│  Prompt Tuning   │ <0.01%      │ Muy baja  │ ★★☆☆☆       │ Muy baja        │
│  Adapters        │ 1-5%        │ Media     │ ★★★★☆       │ Media           │
│  IA3             │ 0.01%       │ Muy baja  │ ★★★☆☆       │ Baja            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VISUALIZACION DE CADA METODO:                                             │
│                                                                             │
│  1. LoRA: Descompone actualizaciones en matrices de bajo rango             │
│     ┌─────────────────────────────────────────┐                            │
│     │  W_original (frozen)                    │                            │
│     │       ↓                                 │                            │
│     │  h = W_original(x) + B @ A(x)          │  A: d×r, B: r×d            │
│     │                      ↑                  │  r << d (rank)            │
│     │              LoRA adapters              │                            │
│     └─────────────────────────────────────────┘                            │
│                                                                             │
│  2. Prefix Tuning: Prefijos virtuales aprendidos                           │
│     ┌─────────────────────────────────────────┐                            │
│     │  [PREFIX_1][PREFIX_2]...[Input tokens]  │                            │
│     │       ↑        ↑                        │                            │
│     │    Learned    Learned                   │  Prefijos en cada capa    │
│     └─────────────────────────────────────────┘                            │
│                                                                             │
│  3. Prompt Tuning: Solo embeddings de prompt                               │
│     ┌─────────────────────────────────────────┐                            │
│     │  [SOFT_1][SOFT_2]...[Input tokens]      │                            │
│     │       ↑        ↑                        │                            │
│     │    Learned embeddings (solo capa 0)     │                            │
│     └─────────────────────────────────────────┘                            │
│                                                                             │
│  4. Adapters: Modulos adicionales entre capas                              │
│     ┌─────────────────────────────────────────┐                            │
│     │  FFN → [Adapter] → LayerNorm            │                            │
│     │         ↓↑                              │                            │
│     │    Bottleneck layer                     │  Down → Act → Up          │
│     └─────────────────────────────────────────┘                            │
│                                                                             │
│  5. IA3: Rescaling de activaciones                                         │
│     ┌─────────────────────────────────────────┐                            │
│     │  K = K * l_k                            │  l_k, l_v, l_ff: learned  │
│     │  V = V * l_v                            │  vectors (muy pocos       │
│     │  FFN_out = FFN_out * l_ff               │  parametros)              │
│     └─────────────────────────────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. LoRA: Low-Rank Adaptation {#lora}

LoRA es la tecnica PEFT mas popular por su equilibrio entre eficiencia y rendimiento.

### Matematica de LoRA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LoRA: MATEMATICA                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  IDEA CENTRAL:                                                              │
│  Los cambios en los pesos durante fine-tuning tienen bajo rango intrinseco  │
│                                                                             │
│  En lugar de aprender:  ΔW ∈ R^{d×d}  (millones de parametros)             │
│  Aprendemos:           ΔW = B @ A      donde A ∈ R^{d×r}, B ∈ R^{r×d}      │
│                                         y r << d (tipicamente r=8,16,64)    │
│                                                                             │
│  FORWARD PASS:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │    h = W_0 @ x + ΔW @ x                                             │   │
│  │      = W_0 @ x + (B @ A) @ x                                        │   │
│  │      = W_0 @ x + B @ (A @ x)                                        │   │
│  │                   ↑     ↑                                           │   │
│  │              down-proj  up-proj                                     │   │
│  │                                                                      │   │
│  │    Input (x)                                                        │   │
│  │       │                                                              │   │
│  │       ├──────────────┐                                              │   │
│  │       │              │                                              │   │
│  │       ▼              ▼                                              │   │
│  │   ┌───────┐      ┌───────┐                                          │   │
│  │   │  W_0  │      │   A   │ ← d×r (inicializado Gaussian)           │   │
│  │   │(frozen)│      │       │                                          │   │
│  │   └───┬───┘      └───┬───┘                                          │   │
│  │       │              │                                              │   │
│  │       │          ┌───▼───┐                                          │   │
│  │       │          │   B   │ ← r×d (inicializado 0)                  │   │
│  │       │          └───┬───┘                                          │   │
│  │       │              │ × (α/r) scaling                              │   │
│  │       └──────┬───────┘                                              │   │
│  │              ▼                                                       │   │
│  │           Output (h)                                                │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  HIPERPARAMETROS:                                                          │
│  • r (rank): Dimension del espacio bajo rango. Mayor r = mas expresividad  │
│  • α (alpha): Factor de escalado. Tipicamente α = 2r                       │
│  • dropout: Regularizacion en matrices LoRA                                │
│  • target_modules: Que capas adaptar (q_proj, v_proj, etc.)               │
│                                                                             │
│  NUMERO DE PARAMETROS ENTRENABLES:                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Full: d × d = d²                                                   │   │
│  │  LoRA: d × r + r × d = 2dr                                         │   │
│  │                                                                      │   │
│  │  Si d=4096, r=16: Full = 16.7M, LoRA = 131K (127x menos)           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion de LoRA

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class LoRALayer(nn.Module):
    """
    Capa LoRA que se anade a una capa Linear existente.

    Implementa: h = W @ x + (B @ A) @ x * (alpha/r)
    donde W es la matriz original (frozen) y A, B son las matrices LoRA.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0
    ) -> None:
        """
        Args:
            in_features: Dimension de entrada
            out_features: Dimension de salida
            rank: Rango de la descomposicion (r)
            alpha: Factor de escalado
            dropout: Dropout para regularizacion
        """
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Matriz A: down-projection (in_features -> rank)
        self.lora_A = nn.Linear(in_features, rank, bias=False)

        # Matriz B: up-projection (rank -> out_features)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Inicializacion
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Inicializa pesos al estilo LoRA original.
        A: inicializacion Kaiming uniform
        B: inicializado a 0 (LoRA empieza con ΔW=0)
        """
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de LoRA.

        Args:
            x: Input tensor

        Returns:
            Delta de LoRA (se suma al output original)
        """
        # A @ x -> B @ (A @ x) -> scale
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))
        return lora_out * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Capa Linear con LoRA integrado.

    Permite congelar la matriz original y solo entrenar LoRA.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0
    ) -> None:
        """
        Args:
            original_layer: Capa Linear original a adaptar
            rank: Rango LoRA
            alpha: Factor de escalado
            dropout: Dropout LoRA
        """
        super().__init__()

        self.original_layer = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

        # Congelar capa original
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: original + LoRA delta."""
        return self.original_layer(x) + self.lora(x)

    def merge_weights(self) -> nn.Linear:
        """
        Fusiona pesos LoRA con originales para inference.

        Returns:
            Nueva capa Linear con pesos fusionados
        """
        # W_merged = W_0 + B @ A * scaling
        delta_w = (
            self.lora.lora_B.weight @
            self.lora.lora_A.weight *
            self.lora.scaling
        )

        merged = nn.Linear(
            self.original_layer.in_features,
            self.original_layer.out_features,
            bias=self.original_layer.bias is not None
        )

        merged.weight.data = self.original_layer.weight.data + delta_w

        if self.original_layer.bias is not None:
            merged.bias.data = self.original_layer.bias.data

        return merged


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0
) -> nn.Module:
    """
    Aplica LoRA a modulos especificos de un modelo.

    Args:
        model: Modelo PyTorch
        target_modules: Lista de nombres de modulos a adaptar
        rank: Rango LoRA
        alpha: Factor de escalado
        dropout: Dropout

    Returns:
        Modelo con LoRA aplicado
    """
    for name, module in model.named_modules():
        # Verificar si es un modulo objetivo
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Obtener modulo padre
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                # Reemplazar con LinearWithLoRA
                lora_layer = LinearWithLoRA(
                    module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                setattr(parent, child_name, lora_layer)

                print(f"LoRA aplicado a: {name}")

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Cuenta parametros entrenables vs totales.

    Returns:
        Dict con conteos de parametros
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params,
        "frozen_parameters": total_params - trainable_params
    }
```

### LoRA con Hugging Face PEFT

```python
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from typing import Optional


def setup_lora_model(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None
) -> tuple:
    """
    Configura un modelo con LoRA usando PEFT.

    Args:
        model_name: Nombre del modelo base
        lora_r: Rango de LoRA
        lora_alpha: Factor de escalado
        lora_dropout: Dropout
        target_modules: Modulos a adaptar

    Returns:
        Tuple de (modelo_peft, tokenizer)
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Modelo base
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Configuracion LoRA
    if target_modules is None:
        # Modulos tipicos para LLaMA/Mistral
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"      # MLP
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",  # No entrenar biases
        modules_to_save=None  # Modulos adicionales a entrenar completos
    )

    # Aplicar LoRA
    model = get_peft_model(model, lora_config)

    # Mostrar parametros
    model.print_trainable_parameters()

    return model, tokenizer


def lora_finetune(
    model_name: str,
    train_dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32
) -> None:
    """
    Fine-tune con LoRA usando PEFT y Trainer.

    Args:
        model_name: Nombre del modelo
        train_dataset: Dataset de entrenamiento
        output_dir: Directorio de salida
        num_epochs: Epochs
        batch_size: Batch size
        learning_rate: Learning rate
        lora_r: Rango LoRA
        lora_alpha: Alpha LoRA
    """
    # Setup modelo con LoRA
    model, tokenizer = setup_lora_model(
        model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch_fused",
        report_to="tensorboard"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Guardar adaptadores LoRA (no el modelo completo)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Adaptadores LoRA guardados en {output_dir}")


def load_and_merge_lora(
    base_model_name: str,
    lora_path: str,
    merge: bool = True
) -> AutoModelForCausalLM:
    """
    Carga un modelo con adaptadores LoRA.

    Args:
        base_model_name: Nombre del modelo base
        lora_path: Path a adaptadores LoRA
        merge: Si fusionar pesos (para inference eficiente)

    Returns:
        Modelo cargado
    """
    # Cargar modelo base
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Cargar adaptadores LoRA
    model = PeftModel.from_pretrained(model, lora_path)

    if merge:
        # Fusionar LoRA en modelo base (elimina overhead de inference)
        model = model.merge_and_unload()
        print("Adaptadores LoRA fusionados con modelo base")

    return model
```

---

## 5. QLoRA: Quantized LoRA {#qlora}

QLoRA combina quantizacion 4-bit con LoRA para entrenar modelos grandes en GPUs consumer.

### Arquitectura QLoRA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QLoRA                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INNOVACIONES CLAVE:                                                        │
│  1. 4-bit NormalFloat (NF4): Quantizacion optima para distribucion normal  │
│  2. Double Quantization: Quantiza los parametros de quantizacion           │
│  3. Paged Optimizers: Evita OOM con gradient checkpointing agresivo        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         FORWARD PASS                                 │   │
│  │                                                                      │   │
│  │   Input (x)                                                         │   │
│  │       │                                                              │   │
│  │       ├──────────────────────────┐                                  │   │
│  │       │                          │                                  │   │
│  │       ▼                          ▼                                  │   │
│  │   ┌───────────┐             ┌─────────┐                             │   │
│  │   │   W_4bit  │             │    A    │  ← BFloat16                 │   │
│  │   │(frozen,   │             │  (LoRA) │                             │   │
│  │   │ NF4)     │             └────┬────┘                             │   │
│  │   └────┬──────┘                  │                                  │   │
│  │        │                    ┌────▼────┐                             │   │
│  │   Dequantize               │    B    │  ← BFloat16                 │   │
│  │   to BF16                  │  (LoRA) │                             │   │
│  │        │                    └────┬────┘                             │   │
│  │        │                         │ × scaling                        │   │
│  │        └───────────┬─────────────┘                                  │   │
│  │                    ▼                                                 │   │
│  │               Output (h)                                            │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TIPOS DE QUANTIZACION:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  INT8 (8-bit): 256 niveles, buena precision                         │   │
│  │  │   │   │   │   │   │   │   │   │                                  │   │
│  │  ─128────────────0───────────127→                                   │   │
│  │                                                                      │   │
│  │  NF4 (4-bit NormalFloat): 16 niveles, optimo para Normal(0,1)      │   │
│  │  │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │                                   │   │
│  │  Valores elegidos para minimizar error en distribucion normal       │   │
│  │                                                                      │   │
│  │  FP4: 4-bit floating point (E2M1 format)                           │   │
│  │  Menos preciso que NF4 para pesos de NNs                           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  VRAM COMPARISON (LLaMA-7B):                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Method          │ VRAM Training │ VRAM Inference │ Quality        │   │
│  │──────────────────┼───────────────┼────────────────┼────────────────│   │
│  │  Full FP16       │ ~56 GB        │ 14 GB          │ Baseline       │   │
│  │  LoRA FP16       │ ~16 GB        │ 14 GB          │ ~98% baseline  │   │
│  │  QLoRA 4-bit     │ ~6 GB         │ 4 GB           │ ~97% baseline  │   │
│  │  QLoRA 4-bit DQ  │ ~5 GB         │ 3.5 GB         │ ~97% baseline  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion de QLoRA

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Optional
from datasets import load_dataset


def setup_qlora_model(
    model_name: str,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[list[str]] = None,
    use_double_quant: bool = True
) -> tuple:
    """
    Configura modelo con QLoRA (4-bit quantization + LoRA).

    Args:
        model_name: Nombre del modelo
        lora_r: Rango LoRA (mayor para QLoRA, tipicamente 64)
        lora_alpha: Alpha LoRA (tipicamente r/4 o igual a r)
        lora_dropout: Dropout
        target_modules: Modulos objetivo
        use_double_quant: Usar double quantization

    Returns:
        Tuple de (modelo, tokenizer)
    """
    # Configuracion de quantizacion 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=use_double_quant  # Quantiza parametros de quant
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Cargar modelo en 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Preparar modelo para k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    # Configuracion LoRA
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none"
    )

    # Aplicar LoRA
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer


def qlora_finetune(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 64,
    lora_alpha: int = 16,
    max_seq_length: int = 2048
) -> None:
    """
    Fine-tuning completo con QLoRA.

    Args:
        model_name: Modelo base
        dataset_name: Dataset de HuggingFace
        output_dir: Directorio de salida
        num_epochs: Epochs
        batch_size: Batch size
        gradient_accumulation_steps: Acumulacion de gradientes
        learning_rate: Learning rate
        lora_r: Rango LoRA
        lora_alpha: Alpha LoRA
        max_seq_length: Longitud maxima
    """
    # Setup modelo
    model, tokenizer = setup_qlora_model(
        model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )

    # Cargar dataset
    dataset = load_dataset(dataset_name, split="train")

    # Formatear dataset (ejemplo con formato de instrucciones)
    def format_instruction(sample: dict) -> dict:
        """Formatea muestra al formato de instrucciones."""
        # Ajustar segun formato del dataset
        if "instruction" in sample:
            text = f"""### Instruction:
{sample['instruction']}

### Response:
{sample.get('output', sample.get('response', ''))}"""
        else:
            text = sample.get("text", str(sample))

        return {"text": text}

    # Tokenizar
    def tokenize(sample: dict) -> dict:
        result = tokenizer(
            sample["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length"
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(format_instruction)
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Training arguments para QLoRA
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="epoch",
        bf16=True,
        tf32=True,  # Usar TensorFloat-32 si disponible
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",  # Optimizer paginado para QLoRA
        max_grad_norm=0.3,  # Gradient clipping
        report_to="tensorboard",
        dataloader_num_workers=4
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Train
    print("Iniciando entrenamiento QLoRA...")
    trainer.train()

    # Guardar
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Modelo QLoRA guardado en {output_dir}")


# Ejemplo de uso para 70B con QLoRA
def train_70b_qlora_example() -> None:
    """
    Ejemplo de configuracion para entrenar LLaMA-2-70B con QLoRA.

    Requisitos: ~48GB VRAM (A100 o 2x3090/4090)
    """
    config = {
        "model_name": "meta-llama/Llama-2-70b-hf",
        "lora_r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-4,
        "max_seq_length": 4096,
        "num_epochs": 1
    }

    print("Configuracion para LLaMA-2-70B con QLoRA:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nVRAM estimada: ~48GB")
    print("Tiempo estimado: ~10-20 horas en A100 para 10K samples")
```

---

## 6. Otras Tecnicas PEFT {#otras-peft}

### Prefix Tuning y Prompt Tuning

```python
from peft import (
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    get_peft_model,
    TaskType
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def setup_prefix_tuning(
    model_name: str,
    num_virtual_tokens: int = 20
) -> tuple:
    """
    Configura Prefix Tuning.

    Prefix Tuning aprende "prefijos virtuales" que se prependen
    a las keys y values de cada capa de atencion.

    Args:
        model_name: Nombre del modelo
        num_virtual_tokens: Numero de tokens virtuales de prefijo

    Returns:
        Tuple de (modelo, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Configuracion Prefix Tuning
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
        prefix_projection=True,  # Usar MLP para proyeccion
        encoder_hidden_size=model.config.hidden_size
    )

    model = get_peft_model(model, prefix_config)
    model.print_trainable_parameters()

    return model, tokenizer


def setup_prompt_tuning(
    model_name: str,
    num_virtual_tokens: int = 8,
    init_text: str | None = None
) -> tuple:
    """
    Configura Prompt Tuning.

    Prompt Tuning aprende embeddings "soft" que se prependen
    a la secuencia de entrada (solo en la capa de embeddings).

    Mas eficiente que Prefix Tuning pero menos expresivo.

    Args:
        model_name: Nombre del modelo
        num_virtual_tokens: Numero de tokens soft
        init_text: Texto para inicializar prompts (opcional)

    Returns:
        Tuple de (modelo, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Configuracion Prompt Tuning
    if init_text:
        prompt_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=init_text,
            tokenizer_name_or_path=model_name
        )
    else:
        prompt_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init=PromptTuningInit.RANDOM
        )

    model = get_peft_model(model, prompt_config)
    model.print_trainable_parameters()

    return model, tokenizer


# IA3: Infused Adapter by Inhibiting and Amplifying Inner Activations
def setup_ia3(model_name: str) -> tuple:
    """
    Configura IA3.

    IA3 aprende vectores de rescaling para K, V y FFN outputs.
    Muy pocos parametros (~0.01% del modelo).

    Args:
        model_name: Nombre del modelo

    Returns:
        Tuple de (modelo, tokenizer)
    """
    from peft import IA3Config

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # IA3 aplica rescaling a K, V y FFN
    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["k_proj", "v_proj", "down_proj"],
        feedforward_modules=["down_proj"]
    )

    model = get_peft_model(model, ia3_config)
    model.print_trainable_parameters()

    return model, tokenizer
```

### Adapters

```python
from peft import AdaptionPromptConfig
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """
    Adapter bottleneck clasico.

    Arquitectura: Down-project -> Activation -> Up-project + Residual

    Anade ~1-5% de parametros pero muy efectivo.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        activation: str = "relu"
    ) -> None:
        super().__init__()

        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU()

        # Inicializacion near-identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward con conexion residual."""
        return x + self.up_proj(self.activation(self.down_proj(x)))


def add_adapters_to_model(
    model: nn.Module,
    bottleneck_dim: int = 64,
    add_after: str = "attention"  # "attention" o "ffn"
) -> nn.Module:
    """
    Anade adapters a un modelo Transformer.

    Args:
        model: Modelo a modificar
        bottleneck_dim: Dimension del bottleneck
        add_after: Donde anadir ("attention" o "ffn")

    Returns:
        Modelo con adapters
    """
    for name, module in model.named_modules():
        if "attention" in name.lower() and add_after == "attention":
            if hasattr(module, "output"):
                # Obtener dimension
                if hasattr(module.output, "dense"):
                    dim = module.output.dense.out_features
                    adapter = BottleneckAdapter(dim, bottleneck_dim)
                    # Insertar adapter
                    original_forward = module.forward

                    def new_forward(*args, adapter=adapter, orig=original_forward, **kwargs):
                        out = orig(*args, **kwargs)
                        if isinstance(out, tuple):
                            return (adapter(out[0]),) + out[1:]
                        return adapter(out)

                    module.forward = new_forward

    return model
```

---

## 7. Cuando Usar Cada Tecnica {#cuando-usar}

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GUIA DE SELECCION DE METODO                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DECISION TREE:                                                             │
│                                                                             │
│                    ┌────────────────────┐                                  │
│                    │  Tienes GPU con    │                                  │
│                    │  suficiente VRAM?  │                                  │
│                    └─────────┬──────────┘                                  │
│                              │                                              │
│               ┌──────────────┴──────────────┐                              │
│               │                             │                              │
│        ┌──────▼──────┐              ┌───────▼───────┐                      │
│        │    >80GB    │              │    <24GB      │                      │
│        │   (A100)    │              │  (Consumer)   │                      │
│        └──────┬──────┘              └───────┬───────┘                      │
│               │                             │                              │
│        ┌──────▼──────┐              ┌───────▼───────┐                      │
│        │ Full FT o   │              │    QLoRA      │                      │
│        │ LoRA        │              │  (siempre)    │                      │
│        └─────────────┘              └───────────────┘                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RECOMENDACIONES POR ESCENARIO:                                            │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ ESCENARIO                     │ METODO RECOMENDADO                 │    │
│  │───────────────────────────────┼────────────────────────────────────│    │
│  │ Modelo pequeno (<7B), GPU pro │ Full fine-tuning                   │    │
│  │ Modelo mediano (7-13B), A100  │ LoRA o Full FT                     │    │
│  │ Modelo grande (70B+), A100    │ QLoRA                              │    │
│  │ Cualquier modelo, GPU consumer│ QLoRA                              │    │
│  │ Muchas tareas, un modelo      │ LoRA (un adapter por tarea)        │    │
│  │ Pocos datos (<1000 samples)   │ Prompt/Prefix Tuning               │    │
│  │ Inference muy rapido critico  │ Full FT (merge all weights)        │    │
│  │ Clasificacion simple          │ Prompt Tuning o IA3                │    │
│  │ Dominio muy especializado     │ Full FT o LoRA con r alto          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  HIPERPARAMETROS RECOMENDADOS:                                             │
│                                                                             │
│  LoRA:                                                                     │
│  • r = 8-16 para tareas simples                                           │
│  • r = 32-64 para tareas complejas                                        │
│  • alpha = 2*r generalmente                                               │
│  • target_modules: al menos q,v (k,o opcionales)                          │
│  • dropout = 0.05-0.1                                                      │
│                                                                             │
│  QLoRA:                                                                    │
│  • r = 64 (compensar quantizacion)                                        │
│  • alpha = 16 o r                                                         │
│  • 4-bit NF4 + double quant                                               │
│  • paged_adamw_32bit optimizer                                            │
│                                                                             │
│  Learning rates:                                                           │
│  • Full FT: 1e-5 a 5e-5                                                   │
│  • LoRA: 1e-4 a 3e-4                                                      │
│  • QLoRA: 1e-4 a 2e-4                                                     │
│  • Prefix/Prompt: 3e-4 a 1e-3                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementacion Practica Completa {#implementacion}

### Pipeline Completo de Fine-tuning

```python
import os
import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
import wandb


@dataclass
class FineTuningArgs:
    """Argumentos para fine-tuning."""

    # Modelo
    model_name: str = "meta-llama/Llama-2-7b-hf"
    use_4bit: bool = True
    use_flash_attention: bool = True

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Entrenamiento
    output_dir: str = "./output"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # Datos
    dataset_name: Optional[str] = None
    train_file: Optional[str] = None
    val_file: Optional[str] = None

    # Logging
    use_wandb: bool = False
    wandb_project: str = "llm-finetuning"


class FineTuningPipeline:
    """Pipeline completo para fine-tuning de LLMs."""

    def __init__(self, args: FineTuningArgs) -> None:
        self.args = args
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

    def setup_model(self) -> None:
        """Configura modelo y tokenizer."""
        print(f"Cargando modelo: {self.args.model_name}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Quantization config
        bnb_config = None
        if self.args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

        # Modelo
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        if self.args.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            **model_kwargs
        )

        # Preparar para k-bit training
        if self.args.use_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True
            )

        # Aplicar LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=self.args.target_modules,
            bias="none"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self) -> None:
        """Prepara datasets de entrenamiento y validacion."""
        print("Preparando datasets...")

        if self.args.dataset_name:
            # Cargar de HuggingFace
            dataset = load_dataset(self.args.dataset_name)
            self.train_dataset = dataset["train"]
            if "validation" in dataset:
                self.eval_dataset = dataset["validation"]
            elif "test" in dataset:
                self.eval_dataset = dataset["test"]

        elif self.args.train_file:
            # Cargar de archivo local
            self.train_dataset = self._load_local_dataset(self.args.train_file)
            if self.args.val_file:
                self.eval_dataset = self._load_local_dataset(self.args.val_file)

        else:
            raise ValueError("Especifica dataset_name o train_file")

        # Formatear y tokenizar
        self.train_dataset = self._format_dataset(self.train_dataset)
        if self.eval_dataset:
            self.eval_dataset = self._format_dataset(self.eval_dataset)

        print(f"Train samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"Eval samples: {len(self.eval_dataset)}")

    def _load_local_dataset(self, file_path: str) -> Dataset:
        """Carga dataset de archivo JSON local."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return Dataset.from_list(data)

    def _format_dataset(self, dataset: Dataset) -> Dataset:
        """Formatea y tokeniza dataset."""

        def format_and_tokenize(sample: dict) -> dict:
            # Formato de instrucciones (ajustar segun tu dataset)
            if "instruction" in sample:
                if sample.get("input"):
                    text = f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample.get('output', sample.get('response', ''))}"""
                else:
                    text = f"""### Instruction:
{sample['instruction']}

### Response:
{sample.get('output', sample.get('response', ''))}"""

            elif "conversations" in sample:
                # Formato de conversaciones
                text = ""
                for turn in sample["conversations"]:
                    role = turn.get("role", turn.get("from", ""))
                    content = turn.get("content", turn.get("value", ""))
                    if role in ["user", "human"]:
                        text += f"### User:\n{content}\n\n"
                    else:
                        text += f"### Assistant:\n{content}\n\n"

            else:
                text = sample.get("text", str(sample))

            # Tokenizar
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.args.max_seq_length,
                padding="max_length"
            )

            encodings["labels"] = encodings["input_ids"].copy()

            return encodings

        # Aplicar formateo
        dataset = dataset.map(
            format_and_tokenize,
            remove_columns=dataset.column_names,
            num_proc=4
        )

        return dataset

    def train(self) -> None:
        """Ejecuta entrenamiento."""
        if self.model is None:
            self.setup_model()
        if self.train_dataset is None:
            self.prepare_dataset()

        # Configurar wandb si especificado
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project,
                config=vars(self.args)
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=0.001,
            warmup_ratio=self.args.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            eval_steps=500 if self.eval_dataset else None,
            load_best_model_at_end=True if self.eval_dataset else False,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            max_grad_norm=0.3,
            report_to="wandb" if self.args.use_wandb else "tensorboard",
            dataloader_num_workers=4,
            save_total_limit=3
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Callbacks
        callbacks = []
        if self.eval_dataset:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=3)
            )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )

        # Entrenar
        print("Iniciando entrenamiento...")
        trainer.train()

        # Guardar modelo final
        print(f"Guardando modelo en {self.args.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.args.output_dir)

        if self.args.use_wandb:
            wandb.finish()

    def evaluate(self, test_prompts: list[str]) -> list[str]:
        """Evalua el modelo con prompts de prueba."""
        self.model.eval()

        responses = []
        for prompt in test_prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses


def main() -> None:
    """Ejemplo de uso del pipeline."""

    # Configuracion
    args = FineTuningArgs(
        model_name="meta-llama/Llama-2-7b-hf",
        use_4bit=True,
        lora_r=64,
        lora_alpha=16,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        dataset_name="tatsu-lab/alpaca",
        output_dir="./llama2-7b-alpaca-qlora",
        use_wandb=False
    )

    # Pipeline
    pipeline = FineTuningPipeline(args)
    pipeline.train()

    # Evaluar
    test_prompts = [
        "### Instruction:\nExplain what machine learning is.\n\n### Response:\n",
        "### Instruction:\nWrite a Python function to sort a list.\n\n### Response:\n"
    ]

    responses = pipeline.evaluate(test_prompts)
    for prompt, response in zip(test_prompts, responses):
        print(f"Prompt: {prompt[:50]}...")
        print(f"Response: {response[:200]}...")
        print("-" * 50)


if __name__ == "__main__":
    main()
```

---

## 9. Datasets para Fine-tuning {#datasets}

### Datasets Populares

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATASETS PARA FINE-TUNING DE LLMs                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INSTRUCTION FOLLOWING:                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Dataset           │ Size    │ Formato        │ Uso                 │    │
│  │───────────────────┼─────────┼────────────────┼─────────────────────│    │
│  │ Alpaca            │ 52K     │ instruction    │ General instruction │    │
│  │ Dolly             │ 15K     │ instruction    │ Open source         │    │
│  │ OpenAssistant     │ 161K    │ conversations  │ Chatbot             │    │
│  │ ShareGPT          │ 90K     │ conversations  │ ChatGPT style       │    │
│  │ LIMA              │ 1K      │ instruction    │ Quality over qty    │    │
│  │ WizardLM          │ 196K    │ instruction    │ Complex reasoning   │    │
│  │ UltraChat         │ 1.5M    │ conversations  │ Large scale chat    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  CODIGO:                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Dataset           │ Size    │ Lenguajes      │ Uso                 │    │
│  │───────────────────┼─────────┼────────────────┼─────────────────────│    │
│  │ Code Alpaca       │ 20K     │ Multi          │ Code generation     │    │
│  │ CodeContests      │ 165     │ Python/C++     │ Competitive prog    │    │
│  │ HumanEval         │ 164     │ Python         │ Evaluation          │    │
│  │ MBPP              │ 974     │ Python         │ Evaluation          │    │
│  │ The Stack         │ 6TB     │ 358 langs      │ Pre-training        │    │
│  │ StarCoder Data    │ 35B tk  │ 86 langs       │ Code LLMs           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  MATEMATICAS Y RAZONAMIENTO:                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ GSM8K             │ 8.5K    │ Math word problems                   │    │
│  │ MATH              │ 12.5K   │ Competition math                     │    │
│  │ MetaMath          │ 395K    │ Math augmented                       │    │
│  │ Orca-Math         │ 200K    │ Math reasoning                       │    │
│  │ ARC               │ 7.7K    │ Science reasoning                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  SEGURIDAD/SAFETY:                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ HH-RLHF           │ 169K    │ Helpful/Harmless preference data     │    │
│  │ PKU-SafeRLHF      │ 330K    │ Safety preference                    │    │
│  │ ToxiGen           │ 274K    │ Toxicity detection                   │    │
│  │ CyberSecEval      │ -       │ Cybersecurity evaluation             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Creando Dataset Propio

```python
from datasets import Dataset
import json
from typing import Optional
from pathlib import Path


def create_instruction_dataset(
    data: list[dict],
    output_path: Optional[str] = None
) -> Dataset:
    """
    Crea dataset de instrucciones.

    Formato esperado:
    {
        "instruction": "...",
        "input": "..." (opcional),
        "output": "..."
    }

    Args:
        data: Lista de diccionarios con instrucciones
        output_path: Path opcional para guardar

    Returns:
        Dataset de HuggingFace
    """
    # Validar formato
    required_keys = ["instruction", "output"]
    for i, item in enumerate(data):
        for key in required_keys:
            if key not in item:
                raise ValueError(f"Item {i} missing key: {key}")

    dataset = Dataset.from_list(data)

    if output_path:
        dataset.save_to_disk(output_path)

    return dataset


def create_conversation_dataset(
    conversations: list[list[dict]],
    output_path: Optional[str] = None
) -> Dataset:
    """
    Crea dataset de conversaciones multi-turno.

    Formato esperado:
    [
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        ...
    ]

    Args:
        conversations: Lista de conversaciones
        output_path: Path opcional para guardar

    Returns:
        Dataset de HuggingFace
    """
    data = [{"conversations": conv} for conv in conversations]
    dataset = Dataset.from_list(data)

    if output_path:
        dataset.save_to_disk(output_path)

    return dataset


# Ejemplo de dataset de ciberseguridad
CYBERSEC_EXAMPLES = [
    {
        "instruction": "Analyze this network log for suspicious activity",
        "input": "192.168.1.100 - - [10/Jan/2024:03:45:23] 'GET /admin/../../../etc/passwd HTTP/1.1' 403 287",
        "output": """This log entry shows a path traversal attack attempt:

1. **Attack Type**: Path Traversal / Directory Traversal
2. **Target**: /etc/passwd (Linux password file)
3. **Technique**: Using ../ sequences to escape web root
4. **Status**: 403 Forbidden (attack blocked)
5. **Source IP**: 192.168.1.100 (internal network)

**Recommendations**:
- Investigate source IP for compromise
- Review web server security configurations
- Check for similar patterns in logs
- Consider blocking the source IP"""
    },
    {
        "instruction": "Explain this YARA rule",
        "input": """rule Suspicious_PowerShell {
    strings:
        $s1 = "IEX" ascii nocase
        $s2 = "Invoke-Expression" ascii nocase
        $s3 = "-enc" ascii nocase
        $s4 = "FromBase64String" ascii nocase
    condition:
        2 of them
}""",
        "output": """This YARA rule detects suspicious PowerShell execution patterns:

**Strings detected**:
1. `IEX` - Alias for Invoke-Expression
2. `Invoke-Expression` - Executes strings as commands
3. `-enc` - Flag for encoded commands
4. `FromBase64String` - Base64 decoding

**Condition**: Triggers if 2+ strings are found

**Why suspicious**: These are common in:
- Fileless malware
- Living-off-the-land attacks
- Encoded payloads
- Obfuscated scripts

**Context**: Legitimate use exists, but combination suggests malicious intent."""
    },
    {
        "instruction": "Generate a Sigma rule for detecting credential dumping",
        "output": """title: Potential Credential Dumping via LSASS Access
status: experimental
description: Detects potential credential dumping by monitoring access to LSASS process
author: Security Team
date: 2024/01/10
references:
    - https://attack.mitre.org/techniques/T1003/001/

logsource:
    category: process_access
    product: windows

detection:
    selection:
        TargetImage|endswith: '\\lsass.exe'
        GrantedAccess|contains:
            - '0x1010'  # PROCESS_QUERY_LIMITED_INFORMATION + PROCESS_VM_READ
            - '0x1410'  # Above + PROCESS_QUERY_INFORMATION
            - '0x1438'  # Common mimikatz access rights
    filter:
        SourceImage|endswith:
            - '\\wmiprvse.exe'
            - '\\taskmgr.exe'
            - '\\procexp64.exe'
    condition: selection and not filter

falsepositives:
    - Legitimate security tools
    - System administration tools

level: high
tags:
    - attack.credential_access
    - attack.t1003.001"""
    }
]


def create_cybersec_dataset() -> Dataset:
    """Crea dataset de ejemplo para ciberseguridad."""
    return create_instruction_dataset(CYBERSEC_EXAMPLES)
```

---

## 10. Evaluacion y Mejores Practicas {#evaluacion}

### Metricas de Evaluacion

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from tqdm import tqdm
import numpy as np


class LLMEvaluator:
    """Evaluador de modelos fine-tuned."""

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None
    ) -> None:
        """
        Args:
            model_path: Path al modelo fine-tuned
            base_model_path: Path al modelo base (para comparacion)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Cargar modelo fine-tuned
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        # Cargar modelo base si especificado
        self.base_model = None
        if base_model_path:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.base_model.eval()

    @torch.inference_mode()
    def compute_perplexity(
        self,
        texts: list[str],
        model: Optional[AutoModelForCausalLM] = None
    ) -> float:
        """
        Calcula perplexity promedio en textos.

        Perplexity = exp(average negative log-likelihood)
        Menor es mejor.
        """
        if model is None:
            model = self.model

        total_loss = 0.0
        total_tokens = 0

        for text in tqdm(texts, desc="Computing perplexity"):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(model.device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens
        return float(np.exp(avg_loss))

    def evaluate_instruction_following(
        self,
        test_cases: list[dict],
        max_new_tokens: int = 256
    ) -> dict:
        """
        Evalua calidad de seguimiento de instrucciones.

        Args:
            test_cases: Lista de {"instruction": ..., "expected_keywords": [...]}
            max_new_tokens: Tokens maximos a generar

        Returns:
            Dict con metricas
        """
        results = {
            "total": len(test_cases),
            "keyword_match_rate": 0.0,
            "avg_response_length": 0.0,
            "responses": []
        }

        keyword_matches = 0
        total_length = 0

        for case in tqdm(test_cases, desc="Evaluating"):
            prompt = f"### Instruction:\n{case['instruction']}\n\n### Response:\n"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Verificar keywords
            if "expected_keywords" in case:
                found = sum(
                    1 for kw in case["expected_keywords"]
                    if kw.lower() in response.lower()
                )
                keyword_matches += found / len(case["expected_keywords"])

            total_length += len(response)
            results["responses"].append(response)

        results["keyword_match_rate"] = keyword_matches / len(test_cases)
        results["avg_response_length"] = total_length / len(test_cases)

        return results

    def compare_with_base(
        self,
        test_texts: list[str]
    ) -> dict:
        """
        Compara modelo fine-tuned con modelo base.

        Returns:
            Dict con comparacion de metricas
        """
        if self.base_model is None:
            raise ValueError("Base model not loaded")

        finetuned_ppl = self.compute_perplexity(test_texts, self.model)
        base_ppl = self.compute_perplexity(test_texts, self.base_model)

        return {
            "finetuned_perplexity": finetuned_ppl,
            "base_perplexity": base_ppl,
            "improvement": (base_ppl - finetuned_ppl) / base_ppl * 100
        }


# Mejores practicas
BEST_PRACTICES = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEJORES PRACTICAS PARA FINE-TUNING                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ANTES DE ENTRENAR:                                                        │
│  ✓ Evaluar si fine-tuning es necesario (vs prompt engineering, RAG)        │
│  ✓ Preparar dataset de calidad (>1000 ejemplos idealmente)                 │
│  ✓ Limpiar datos: remover duplicados, errores, contenido toxico           │
│  ✓ Dividir en train/val/test (80/10/10 o similar)                         │
│  ✓ Definir metricas de evaluacion claras                                   │
│                                                                             │
│  DURANTE ENTRENAMIENTO:                                                    │
│  ✓ Usar learning rate bajo (1e-5 a 3e-4 dependiendo del metodo)           │
│  ✓ Monitorear loss en validation para evitar overfitting                   │
│  ✓ Usar early stopping si validation loss aumenta                          │
│  ✓ Guardar checkpoints frecuentemente                                      │
│  ✓ Usar gradient clipping (max_grad_norm=0.3-1.0)                         │
│                                                                             │
│  DESPUES DE ENTRENAR:                                                      │
│  ✓ Evaluar en test set no visto durante entrenamiento                      │
│  ✓ Comparar con modelo base para medir mejora real                         │
│  ✓ Probar casos edge y posibles failures                                   │
│  ✓ Verificar que no hay regression en otras capacidades                    │
│  ✓ Documentar hiperparametros y resultados                                 │
│                                                                             │
│  ERRORES COMUNES A EVITAR:                                                 │
│  ✗ Dataset muy pequeno (<100 ejemplos)                                     │
│  ✗ Learning rate muy alto (causa catastrophic forgetting)                  │
│  ✗ Demasiadas epochs (overfitting)                                         │
│  ✗ No evaluar en datos no vistos                                           │
│  ✗ Ignorar la calidad de datos (garbage in, garbage out)                   │
│  ✗ Fine-tuning cuando prompt engineering seria suficiente                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
```

---

## 11. Aplicaciones en Ciberseguridad {#ciberseguridad}

### Fine-tuning para Tareas de Seguridad

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SecurityTask(Enum):
    """Tareas de seguridad para fine-tuning."""
    THREAT_DETECTION = "threat_detection"
    MALWARE_ANALYSIS = "malware_analysis"
    LOG_ANALYSIS = "log_analysis"
    VULN_ASSESSMENT = "vulnerability_assessment"
    INCIDENT_RESPONSE = "incident_response"
    RULE_GENERATION = "rule_generation"


@dataclass
class SecurityFineTuningConfig:
    """Configuracion para fine-tuning de seguridad."""

    task: SecurityTask
    model_name: str = "meta-llama/Llama-2-7b-hf"
    use_qlora: bool = True
    lora_r: int = 64

    # Configuraciones especificas por tarea
    def get_system_prompt(self) -> str:
        """Retorna system prompt optimizado para la tarea."""
        prompts = {
            SecurityTask.THREAT_DETECTION: """You are a cybersecurity threat analyst.
Analyze inputs for security threats, IOCs, and attack patterns.
Be precise and cite specific evidence.""",

            SecurityTask.MALWARE_ANALYSIS: """You are a malware analyst expert.
Analyze code, behavior, and artifacts for malicious indicators.
Provide technical details and MITRE ATT&CK mappings.""",

            SecurityTask.LOG_ANALYSIS: """You are a SOC analyst specializing in log analysis.
Identify anomalies, correlate events, and detect attack patterns.
Reference specific log fields and timestamps.""",

            SecurityTask.VULN_ASSESSMENT: """You are a vulnerability researcher.
Analyze code and configurations for security weaknesses.
Provide CVE references and remediation guidance.""",

            SecurityTask.INCIDENT_RESPONSE: """You are an incident response expert.
Guide through containment, eradication, and recovery steps.
Prioritize by severity and business impact.""",

            SecurityTask.RULE_GENERATION: """You are a detection engineer.
Create precise detection rules in requested formats (Sigma, YARA, Snort).
Minimize false positives while maintaining coverage."""
        }
        return prompts.get(self.task, "You are a security expert.")


def create_security_finetuning_dataset(
    task: SecurityTask,
    raw_data: list[dict]
) -> list[dict]:
    """
    Formatea datos para fine-tuning de seguridad.

    Args:
        task: Tarea de seguridad
        raw_data: Datos crudos

    Returns:
        Dataset formateado para instrucciones
    """
    config = SecurityFineTuningConfig(task=task)
    system_prompt = config.get_system_prompt()

    formatted_data = []

    for item in raw_data:
        # Formato depende de la tarea
        if task == SecurityTask.LOG_ANALYSIS:
            formatted_data.append({
                "instruction": f"{system_prompt}\n\nAnalyze these logs:",
                "input": item.get("logs", ""),
                "output": item.get("analysis", "")
            })

        elif task == SecurityTask.RULE_GENERATION:
            formatted_data.append({
                "instruction": f"{system_prompt}\n\nGenerate a {item.get('format', 'Sigma')} rule for:",
                "input": item.get("threat_description", ""),
                "output": item.get("rule", "")
            })

        elif task == SecurityTask.MALWARE_ANALYSIS:
            formatted_data.append({
                "instruction": f"{system_prompt}\n\nAnalyze this sample:",
                "input": item.get("sample_info", ""),
                "output": item.get("analysis", "")
            })

        else:
            # Formato generico
            formatted_data.append({
                "instruction": f"{system_prompt}\n\n{item.get('question', '')}",
                "input": item.get("context", ""),
                "output": item.get("answer", "")
            })

    return formatted_data


# Ejemplo de uso para ciberseguridad
def demo_security_finetuning() -> None:
    """Demo de fine-tuning para ciberseguridad."""

    # Ejemplo de datos de entrenamiento para analisis de logs
    log_analysis_data = [
        {
            "logs": """2024-01-10 03:45:23 Failed password for admin from 10.0.0.50 port 22
2024-01-10 03:45:24 Failed password for admin from 10.0.0.50 port 22
2024-01-10 03:45:25 Failed password for admin from 10.0.0.50 port 22
2024-01-10 03:45:26 Failed password for admin from 10.0.0.50 port 22
2024-01-10 03:45:27 Accepted password for admin from 10.0.0.50 port 22""",
            "analysis": """**Finding**: Brute Force Attack - Successful
**Severity**: Critical
**Source IP**: 10.0.0.50
**Target**: SSH (port 22), user 'admin'
**Timeline**: 4 failed attempts followed by success in 4 seconds
**MITRE ATT&CK**: T1110.001 (Brute Force: Password Guessing)

**Evidence**:
- Rapid succession of failed attempts (1/second)
- Same source IP and target user
- Eventual success indicates compromised credentials

**Recommendations**:
1. Immediately disable 'admin' account
2. Investigate 10.0.0.50 for compromise
3. Review all actions from this session
4. Implement account lockout policies
5. Enable MFA for SSH access"""
        }
    ]

    # Crear dataset formateado
    formatted = create_security_finetuning_dataset(
        SecurityTask.LOG_ANALYSIS,
        log_analysis_data
    )

    print("Dataset formateado para fine-tuning de log analysis:")
    print(json.dumps(formatted[0], indent=2))


if __name__ == "__main__":
    import json
    demo_security_finetuning()
```

---

## Resumen

Este capitulo ha cubierto las tecnicas fundamentales de fine-tuning para LLMs:

1. **Full Fine-tuning**: Maxima expresividad pero alto costo computacional
2. **LoRA**: Balance excelente entre eficiencia y rendimiento
3. **QLoRA**: Permite entrenar modelos grandes en GPUs consumer
4. **Otras tecnicas PEFT**: Prefix Tuning, Prompt Tuning, IA3, Adapters
5. **Seleccion de metodo**: Guia practica segun recursos y necesidades
6. **Datasets**: Fuentes y creacion de datos de entrenamiento
7. **Evaluacion**: Metricas y mejores practicas
8. **Ciberseguridad**: Aplicaciones especificas del dominio

### Recursos Adicionales

- Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
- Paper: "QLoRA: Efficient Finetuning of Quantized LLMs"
- Codigo: https://github.com/huggingface/peft
- Codigo: https://github.com/artidoro/qlora

---

*Siguiente: [71. RLHF - Reinforcement Learning from Human Feedback](./71-rlhf.md)*
