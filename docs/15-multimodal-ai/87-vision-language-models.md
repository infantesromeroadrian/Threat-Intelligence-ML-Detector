# Vision-Language Models (VLMs)

## Introduccion

Los Vision-Language Models (VLMs) combinan capacidades de vision y lenguaje para realizar tareas complejas como: image captioning, visual question answering (VQA), visual reasoning, y conversacion multimodal. A diferencia de CLIP que alinea embeddings, los VLMs generan texto basado en imagenes.

```
Evolucion de Vision-Language:

Era 1: Modelos Especializados (2015-2019)
┌───────────────────────────────────────────────────────────────┐
│ Image Captioning:    CNN → LSTM → "A dog playing in park"   │
│ VQA:                 CNN + Question Encoder → Answer        │
│ Visual Grounding:    Modelo separado por tarea              │
│                                                              │
│ Limitacion: Un modelo por tarea, sin generalizacion          │
└───────────────────────────────────────────────────────────────┘

Era 2: Vision-Language Pre-training (2020-2022)
┌───────────────────────────────────────────────────────────────┐
│ CLIP:     Contrastive learning imagen-texto                  │
│ ALIGN:    Scale up contrastive learning                      │
│ BLIP:     Unified framework captioning + retrieval           │
│                                                              │
│ Avance: Pre-training masivo, transfer learning               │
└───────────────────────────────────────────────────────────────┘

Era 3: Large Vision-Language Models (2023+)
┌───────────────────────────────────────────────────────────────┐
│ LLaVA:      Vision encoder + LLM                             │
│ GPT-4V:     Multimodal GPT                                   │
│ Gemini:     Native multimodal                                │
│ BLIP-2:     Efficient vision-LLM bridging                    │
│                                                              │
│ Revolucion: Capacidades de razonamiento visual complejas     │
└───────────────────────────────────────────────────────────────┘
```

## Arquitecturas VLM

### Taxonomia de Arquitecturas

```
Arquitecturas VLM:

1. ENCODER-DECODER (BLIP)
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │   Image     │    │   Cross     │    │    Text     │
   │   Encoder   │───▶│  Attention  │───▶│   Decoder   │
   │   (ViT)     │    │   Fusion    │    │   (Causal)  │
   └─────────────┘    └─────────────┘    └─────────────┘
                                               │
                                               ▼
                                         "A photo of..."

2. CONNECTOR-BASED (LLaVA, BLIP-2)
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │   Vision    │    │  Connector  │    │    LLM      │
   │   Encoder   │───▶│  (Q-Former/ │───▶│  (LLaMA/    │
   │   (CLIP)    │    │   Linear)   │    │   Vicuna)   │
   └─────────────┘    └─────────────┘    └─────────────┘
         ▲                   │                  │
         │                   │                  ▼
      Frozen            Trainable         Generated Text

3. NATIVE MULTIMODAL (Flamingo, Gemini)
   ┌────────────────────────────────────────────────────┐
   │              Interleaved Transformer               │
   │  ┌─────────────────────────────────────────────┐  │
   │  │ [IMG] [IMG] The image shows [TXT] [TXT]... │  │
   │  │                                             │  │
   │  │ Cross-attention cada N capas               │  │
   │  └─────────────────────────────────────────────┘  │
   └────────────────────────────────────────────────────┘
```

## BLIP y BLIP-2

### BLIP (Bootstrapping Language-Image Pre-training)

```
BLIP Architecture:

┌──────────────────────────────────────────────────────────────────┐
│                           BLIP                                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Image Encoder (ViT)                    │   │
│  │                          │                                │   │
│  │                    Image Features                         │   │
│  │                          │                                │   │
│  └──────────────────────────┼────────────────────────────────┘   │
│                             │                                    │
│        ┌────────────────────┼────────────────────┐               │
│        │                    │                    │               │
│        ▼                    ▼                    ▼               │
│  ┌───────────┐       ┌───────────┐       ┌───────────┐          │
│  │   ITC     │       │   ITM     │       │   LM      │          │
│  │ (Image-   │       │ (Image-   │       │ (Language │          │
│  │  Text     │       │  Text     │       │  Model    │          │
│  │Contrastive│       │ Matching) │       │  Caption) │          │
│  └───────────┘       └───────────┘       └───────────┘          │
│        │                    │                    │               │
│    Retrieval            Matching             Captioning          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Tres objetivos de entrenamiento:
1. ITC: Contrastive learning (como CLIP)
2. ITM: Binary classification si imagen-texto coinciden
3. LM: Autoregressive caption generation
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval
)
from PIL import Image


class BLIPWrapper:
    """
    Wrapper para usar BLIP en diferentes tareas.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device

        # Cargar modelo y procesador
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 50,
        num_beams: int = 5,
        min_length: int = 5
    ) -> str:
        """
        Genera caption para una imagen.

        Args:
            image: Imagen PIL
            max_length: Longitud maxima del caption
            num_beams: Beam search width
            min_length: Longitud minima

        Returns:
            Caption generado
        """
        # Preprocesar
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Generar
        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            min_length=min_length
        )

        # Decodificar
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return caption

    @torch.no_grad()
    def conditional_caption(
        self,
        image: Image.Image,
        prompt: str,
        max_length: int = 50
    ) -> str:
        """
        Genera caption condicionado a un prompt.

        Args:
            image: Imagen PIL
            prompt: Texto inicial (ej: "a photography of")

        Returns:
            Caption completado
        """
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_length=max_length
        )

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return caption

    @torch.no_grad()
    def batch_caption(
        self,
        images: List[Image.Image],
        prompts: Optional[List[str]] = None,
        max_length: int = 50
    ) -> List[str]:
        """
        Genera captions para un batch de imagenes.
        """
        if prompts is None:
            prompts = [None] * len(images)

        captions = []
        for image, prompt in zip(images, prompts):
            if prompt:
                caption = self.conditional_caption(image, prompt, max_length)
            else:
                caption = self.generate_caption(image, max_length)
            captions.append(caption)

        return captions


class BLIPRetrieval:
    """
    BLIP para image-text retrieval.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-itm-large-coco",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def compute_itm_score(
        self,
        image: Image.Image,
        text: str
    ) -> float:
        """
        Computa Image-Text Matching score.

        Returns:
            Probabilidad de que imagen y texto coincidan (0-1)
        """
        inputs = self.processor(image, text, return_tensors="pt").to(self.device)

        # ITM head output
        outputs = self.model(**inputs)

        # Softmax sobre las dos clases [no_match, match]
        itm_scores = F.softmax(outputs.itm_score, dim=-1)

        # Retornar probabilidad de match
        return itm_scores[0, 1].item()

    @torch.no_grad()
    def rank_texts(
        self,
        image: Image.Image,
        candidate_texts: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Rankea textos por relevancia a una imagen.

        Returns:
            Lista de (texto, score) ordenada por score descendente
        """
        scores = []
        for text in candidate_texts:
            score = self.compute_itm_score(image, text)
            scores.append((text, score))

        # Ordenar por score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    @torch.no_grad()
    def rank_images(
        self,
        text: str,
        candidate_images: List[Image.Image]
    ) -> List[Tuple[int, float]]:
        """
        Rankea imagenes por relevancia a un texto.

        Returns:
            Lista de (indice, score) ordenada por score descendente
        """
        scores = []
        for idx, image in enumerate(candidate_images):
            score = self.compute_itm_score(image, text)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores
```

### BLIP-2: Efficient Vision-Language Pre-training

```
BLIP-2 Architecture:

┌────────────────────────────────────────────────────────────────────┐
│                            BLIP-2                                  │
│                                                                    │
│   ┌─────────────┐                                                  │
│   │   Frozen    │                                                  │
│   │   Vision    │                                                  │
│   │   Encoder   │                                                  │
│   │  (ViT-G)    │                                                  │
│   └──────┬──────┘                                                  │
│          │                                                         │
│          ▼                                                         │
│   ┌──────────────────────────────────────────┐                     │
│   │           Q-Former (Trainable)           │                     │
│   │  ┌───────────────────────────────────┐  │                     │
│   │  │      Learned Query Tokens         │  │                     │
│   │  │      (32 queries × 768 dim)       │  │                     │
│   │  └───────────────┬───────────────────┘  │                     │
│   │                  │                       │                     │
│   │    ┌─────────────┴─────────────┐        │                     │
│   │    │    Cross-Attention to     │        │                     │
│   │    │    Image Features         │        │                     │
│   │    └─────────────┬─────────────┘        │                     │
│   │                  │                       │                     │
│   │    ┌─────────────┴─────────────┐        │                     │
│   │    │    Self-Attention +       │        │                     │
│   │    │    Optional Text Input    │        │                     │
│   │    └─────────────┬─────────────┘        │                     │
│   └──────────────────┼───────────────────────┘                     │
│                      │                                             │
│          32 visual tokens (768d)                                   │
│                      │                                             │
│          ┌───────────┴───────────┐                                │
│          │   Linear Projection   │                                │
│          └───────────┬───────────┘                                │
│                      │                                             │
│          ┌───────────┴───────────┐                                │
│          │      Frozen LLM       │                                │
│          │   (OPT-2.7B/6.7B or   │                                │
│          │    FlanT5-XL/XXL)     │                                │
│          └───────────────────────┘                                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Ventajas de Q-Former:
- Solo 32 tokens visuales (vs 256+ de ViT)
- Eficiente computacionalmente
- Funciona con diferentes LLMs
- Vision encoder y LLM frozen → menos parametros entrenables
```

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2VQA:
    """
    BLIP-2 para Visual Question Answering y tareas generativas.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False
    ):
        self.device = device

        self.processor = Blip2Processor.from_pretrained(model_name)

        # Cargar modelo (opcionalmente en 8-bit para menos memoria)
        if load_in_8bit:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(device)

        self.model.eval()

    @torch.no_grad()
    def answer_question(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 50
    ) -> str:
        """
        Responde una pregunta sobre una imagen.

        Args:
            image: Imagen PIL
            question: Pregunta en lenguaje natural

        Returns:
            Respuesta generada
        """
        # Formato de prompt para VQA
        prompt = f"Question: {question} Answer:"

        inputs = self.processor(image, prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False
        )

        answer = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return answer.strip()

    @torch.no_grad()
    def describe_image(
        self,
        image: Image.Image,
        prompt: str = "This image shows",
        max_length: int = 100
    ) -> str:
        """
        Genera descripcion detallada de una imagen.
        """
        inputs = self.processor(image, prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        description = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return description

    @torch.no_grad()
    def analyze_for_security(
        self,
        image: Image.Image
    ) -> Dict[str, str]:
        """
        Analisis de seguridad de una imagen.
        Responde multiples preguntas relevantes para seguridad.
        """
        security_questions = [
            "What type of interface or application is shown?",
            "Is this a login page or authentication screen?",
            "What brand or company logos are visible?",
            "Are there any suspicious elements or warnings visible?",
            "What is the URL shown in the browser if any?",
            "Is this interface asking for personal information?"
        ]

        analysis = {}
        for question in security_questions:
            answer = self.answer_question(image, question)
            analysis[question] = answer

        return analysis
```

## LLaVA (Large Language and Vision Assistant)

```
LLaVA Architecture:

┌────────────────────────────────────────────────────────────────────┐
│                            LLaVA                                    │
│                                                                    │
│   ┌─────────────┐                                                  │
│   │   CLIP      │                                                  │
│   │   Vision    │                                                  │
│   │   Encoder   │                                                  │
│   │ (ViT-L/14)  │                                                  │
│   └──────┬──────┘                                                  │
│          │                                                         │
│    Image Features                                                  │
│    (257 × 1024)                                                    │
│          │                                                         │
│          ▼                                                         │
│   ┌──────────────┐                                                 │
│   │   Linear     │  ← Simple projection (trainable)                │
│   │  Projection  │     W: (1024, 4096)                             │
│   └──────┬───────┘                                                 │
│          │                                                         │
│    Visual Tokens                                                   │
│    (257 × 4096)                                                    │
│          │                                                         │
│          ▼                                                         │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │                                                          │    │
│   │   [Visual Tokens] [User Prompt Tokens]                   │    │
│   │        ↓                    ↓                            │    │
│   │   ┌────────────────────────────────────────────────┐    │    │
│   │   │              LLaMA / Vicuna                     │    │    │
│   │   │         (7B / 13B parameters)                   │    │    │
│   │   │                                                 │    │    │
│   │   │   Standard causal language modeling            │    │    │
│   │   └────────────────────────────────────────────────┘    │    │
│   │                          │                               │    │
│   │                          ▼                               │    │
│   │                    Generated Text                        │    │
│   │                                                          │    │
│   └──────────────────────────────────────────────────────────┘    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Entrenamiento LLaVA (2 fases):

Fase 1: Pre-training (Feature Alignment)
- Dataset: CC3M filtered (595K image-text pairs)
- Solo se entrena la proyeccion linear
- Vision encoder y LLM frozen
- Objetivo: alinear features visuales con espacio LLM

Fase 2: Fine-tuning (Visual Instruction Tuning)
- Dataset: LLaVA-Instruct-150K (GPT-4 generated)
- Se entrena proyeccion + LLM (LoRA o full fine-tune)
- Vision encoder frozen
- Objetivo: seguir instrucciones multimodales
```

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import List, Dict, Any, Optional
import torch


class LLaVAWrapper:
    """
    Wrapper para LLaVA con diferentes capacidades.
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = False
    ):
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_name)

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.model.eval()

    def format_prompt(
        self,
        user_message: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Formatea prompt siguiendo el template de LLaVA.
        """
        if system_message:
            prompt = f"<|system|>\n{system_message}\n<|user|>\n<image>\n{user_message}\n<|assistant|>\n"
        else:
            prompt = f"USER: <image>\n{user_message}\nASSISTANT:"

        return prompt

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        system_message: Optional[str] = None
    ) -> str:
        """
        Genera respuesta dada una imagen y prompt.
        """
        formatted_prompt = self.format_prompt(prompt, system_message)

        inputs = self.processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.pad_token_id
        )

        # Decodificar solo la parte generada
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    @torch.no_grad()
    def conversation(
        self,
        image: Image.Image,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Conversacion multi-turno sobre una imagen.

        Args:
            image: Imagen PIL
            messages: Lista de {"role": "user"|"assistant", "content": str}

        Returns:
            Respuesta del modelo
        """
        # Construir prompt de conversacion
        conversation_text = ""
        for msg in messages:
            if msg["role"] == "user":
                if conversation_text == "":
                    conversation_text += f"USER: <image>\n{msg['content']}\n"
                else:
                    conversation_text += f"USER: {msg['content']}\n"
            else:
                conversation_text += f"ASSISTANT: {msg['content']}\n"

        conversation_text += "ASSISTANT:"

        inputs = self.processor(
            text=conversation_text,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )

        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)

        return response.strip()


def demo_llava():
    """Demo de capacidades de LLaVA."""

    llava = LLaVAWrapper(load_in_4bit=True)

    # Ejemplo: Descripcion detallada
    # image = Image.open("example.jpg")
    # description = llava.generate(
    #     image,
    #     "Describe this image in detail. What objects, people, and actions do you see?"
    # )

    # Ejemplo: VQA
    # answer = llava.generate(image, "How many people are in this image?")

    # Ejemplo: OCR
    # text = llava.generate(image, "Read and transcribe all text visible in this image.")

    # Ejemplo: Razonamiento
    # reasoning = llava.generate(
    #     image,
    #     "Based on this image, what event or situation is taking place? Explain your reasoning."
    # )

    print("LLaVA demo initialized")
    return llava
```

## Flamingo Architecture

```
Flamingo: Interleaved Vision-Language Model

┌────────────────────────────────────────────────────────────────────┐
│                          Flamingo                                   │
│                                                                    │
│   Input: <image1>The cat<image2>sat on the<image3>mat.            │
│                                                                    │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │              Frozen Vision Encoder (NFNet)                  │  │
│   │                                                             │  │
│   │   Image1 → Perceiver Resampler → Visual Tokens (64)        │  │
│   │   Image2 → Perceiver Resampler → Visual Tokens (64)        │  │
│   │   Image3 → Perceiver Resampler → Visual Tokens (64)        │  │
│   └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │              Frozen Language Model (Chinchilla)             │  │
│   │                                                             │  │
│   │   Capa 1: Self-Attention                                    │  │
│   │   Capa 2: Self-Attention                                    │  │
│   │   Capa 3: Self-Attention                                    │  │
│   │   Capa 4: GATED XATTN-DENSE (trainable)  ← Visual tokens   │  │
│   │   Capa 5: Self-Attention                                    │  │
│   │   ...                                                       │  │
│   │   Capa N: GATED XATTN-DENSE (trainable)  ← Visual tokens   │  │
│   │   Capa N+1: Self-Attention                                  │  │
│   │   ...                                                       │  │
│   └────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   Gated Cross-Attention:                                          │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │   y = x + tanh(α) * XATTN(LN(x), visual_tokens)            │  │
│   │                                                             │  │
│   │   α se inicializa a 0 → al principio ignora vision         │  │
│   │   α se aprende → gradualmente integra informacion visual   │  │
│   └────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Ventajas:
- Soporta multiples imagenes/videos intercalados
- Few-shot learning visual (dar ejemplos en prompt)
- LLM frozen → preserva capacidades de lenguaje
- Gated mechanism → integracion suave
```

```python
class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler: reduce secuencia visual a tokens fijos.
    Usado en Flamingo para comprimir features de imagen.
    """

    def __init__(
        self,
        dim: int = 1024,
        num_queries: int = 64,
        num_layers: int = 6,
        num_heads: int = 16,
        ff_mult: int = 4
    ):
        super().__init__()

        self.num_queries = num_queries

        # Learned query tokens
        self.queries = nn.Parameter(torch.randn(num_queries, dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(dim, num_heads, batch_first=True),
                'cross_norm': nn.LayerNorm(dim),
                'self_attn': nn.MultiheadAttention(dim, num_heads, batch_first=True),
                'self_norm': nn.LayerNorm(dim),
                'ff': nn.Sequential(
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(dim * ff_mult, dim)
                ),
                'ff_norm': nn.LayerNorm(dim)
            })
            for _ in range(num_layers)
        ])

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, num_patches, dim) from vision encoder

        Returns:
            (batch, num_queries, dim) compressed visual tokens
        """
        batch_size = visual_features.shape[0]

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            # Cross attention: queries attend to visual features
            cross_out, _ = layer['cross_attn'](
                query=layer['cross_norm'](queries),
                key=visual_features,
                value=visual_features
            )
            queries = queries + cross_out

            # Self attention among queries
            self_out, _ = layer['self_attn'](
                query=layer['self_norm'](queries),
                key=layer['self_norm'](queries),
                value=layer['self_norm'](queries)
            )
            queries = queries + self_out

            # Feed-forward
            queries = queries + layer['ff'](layer['ff_norm'](queries))

        return queries


class GatedCrossAttentionDense(nn.Module):
    """
    Gated Cross-Attention layer usado en Flamingo.
    Permite integracion gradual de informacion visual.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        dim_head: int = 64
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Gating parameter (initialized to 0)
        self.gate = nn.Parameter(torch.zeros(1))

        # Dense layer after attention
        self.dense = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        visual_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim) language hidden states
            visual_tokens: (batch, num_visual, dim) from perceiver resampler

        Returns:
            (batch, seq_len, dim) updated hidden states
        """
        # Cross-attention
        normed = self.norm(x)
        attn_out, _ = self.cross_attn(
            query=normed,
            key=visual_tokens,
            value=visual_tokens
        )

        # Dense
        attn_out = self.dense(attn_out)

        # Gated residual: tanh(gate) empieza en 0, crece durante training
        x = x + torch.tanh(self.gate) * attn_out

        return x
```

## Visual Instruction Tuning

```
Visual Instruction Tuning:

Problema:
┌────────────────────────────────────────────────────────────────┐
│ Modelos pre-entrenados son buenos en tareas generales, pero   │
│ no siguen instrucciones complejas ni mantienen conversaciones │
└────────────────────────────────────────────────────────────────┘

Solucion: Fine-tune con datos de instrucciones visuales
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Generacion de Datos (LLaVA approach):                        │
│                                                                │
│  1. Tomar imagen de COCO/CC3M                                 │
│  2. Usar GPT-4 para generar:                                  │
│     - Conversaciones multi-turno sobre la imagen              │
│     - Descripciones detalladas                                │
│     - Preguntas de razonamiento complejo                      │
│                                                                │
│  Ejemplo de dato generado:                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ <image>                                                  │ │
│  │ User: What is unusual about this image?                  │ │
│  │ Assistant: The unusual aspect of this image is that      │ │
│  │ a cat appears to be using a laptop computer, which is    │ │
│  │ not typical behavior for a cat...                        │ │
│  │                                                          │ │
│  │ User: Why might someone create such an image?            │ │
│  │ Assistant: Such images are often created for humor or    │ │
│  │ to demonstrate the concept of anthropomorphism...        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Tipos de Instrucciones:

1. Conversation: Multi-turno sobre imagen
2. Detailed Description: Descripcion exhaustiva
3. Complex Reasoning: Preguntas que requieren inferencia
4. OCR/Text Understanding: Leer y entender texto en imagen
5. Spatial Reasoning: Posiciones relativas de objetos
```

```python
from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class VisualInstruction:
    """Estructura de una instruccion visual."""
    image_path: str
    conversations: List[Dict[str, str]]
    task_type: str  # "conversation", "description", "reasoning", etc.


class VisualInstructionDataset:
    """
    Dataset para visual instruction tuning.
    """

    def __init__(
        self,
        data_path: str,
        processor,
        max_length: int = 2048
    ):
        self.processor = processor
        self.max_length = max_length

        # Cargar datos
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retorna un ejemplo procesado."""
        item = self.data[idx]

        # Cargar imagen
        image = Image.open(item['image_path']).convert('RGB')

        # Construir texto de conversacion
        text = self._format_conversation(item['conversations'])

        # Procesar
        encoding = self.processor(
            images=image,
            text=text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)  # Para causal LM
        }

    def _format_conversation(
        self,
        conversations: List[Dict[str, str]]
    ) -> str:
        """Formatea conversacion para el modelo."""
        text = ""
        for turn in conversations:
            if turn['from'] == 'human':
                text += f"USER: {turn['value']}\n"
            else:
                text += f"ASSISTANT: {turn['value']}\n"
        return text


def generate_security_instructions() -> List[Dict]:
    """
    Genera instrucciones de entrenamiento para seguridad.
    En produccion: usar GPT-4 para generar datos de calidad.
    """
    templates = [
        {
            "task": "phishing_detection",
            "conversations": [
                {"from": "human", "value": "Analyze this webpage screenshot for signs of phishing."},
                {"from": "assistant", "value": "I'll analyze this screenshot for phishing indicators:\n\n1. **URL Analysis**: [describe URL if visible]\n2. **Visual Elements**: [describe logos, design]\n3. **Form Fields**: [list sensitive data requested]\n4. **Red Flags**: [list suspicious elements]\n\n**Verdict**: [phishing/legitimate with confidence]"}
            ]
        },
        {
            "task": "malware_ui_analysis",
            "conversations": [
                {"from": "human", "value": "What type of application interface is shown? Is it potentially malicious?"},
                {"from": "assistant", "value": "This interface appears to be [type]. Key observations:\n\n1. **Interface Type**: [description]\n2. **Suspicious Elements**: [list if any]\n3. **Risk Assessment**: [low/medium/high]"}
            ]
        },
        {
            "task": "credential_harvesting",
            "conversations": [
                {"from": "human", "value": "Does this page appear to be collecting user credentials? What brand is it impersonating?"},
                {"from": "assistant", "value": "Analysis of credential collection attempt:\n\n1. **Target Brand**: [brand name if identifiable]\n2. **Fields Requested**: [list fields]\n3. **Legitimacy Indicators**: [describe]\n4. **Recommendation**: [action to take]"}
            ]
        }
    ]

    return templates
```

## Aplicaciones en Seguridad

### Document Analysis for Threat Intelligence

```python
class SecurityDocumentAnalyzer:
    """
    Analiza documentos para threat intelligence usando VLMs.
    Casos de uso:
    - Analisis de PDFs sospechosos (screenshots)
    - Extraccion de IOCs de reportes
    - Clasificacion de documentos sensibles
    """

    def __init__(self, vlm: LLaVAWrapper):
        self.vlm = vlm

    def analyze_document_screenshot(
        self,
        screenshot: Image.Image
    ) -> Dict[str, Any]:
        """
        Analiza screenshot de documento para seguridad.
        """
        analyses = {}

        # 1. Identificar tipo de documento
        doc_type_prompt = """
        Analyze this document screenshot and identify:
        1. What type of document is this? (invoice, contract, report, email, etc.)
        2. Is there any branding or company logos visible?
        3. What is the apparent purpose of this document?
        """
        analyses['document_type'] = self.vlm.generate(screenshot, doc_type_prompt)

        # 2. Extraer IOCs potenciales
        ioc_prompt = """
        Extract any potential Indicators of Compromise (IOCs) visible in this document:
        - IP addresses
        - Domain names
        - Email addresses
        - File hashes (MD5, SHA1, SHA256)
        - URLs
        - File paths
        List each IOC found with its type.
        """
        analyses['iocs'] = self.vlm.generate(screenshot, ioc_prompt)

        # 3. Detectar informacion sensible
        sensitive_prompt = """
        Identify any sensitive information visible in this document:
        - Personal Identifiable Information (PII)
        - Financial data
        - Credentials or passwords
        - API keys or tokens
        - Internal company information
        """
        analyses['sensitive_data'] = self.vlm.generate(screenshot, sensitive_prompt)

        # 4. Evaluacion de riesgo
        risk_prompt = """
        Based on the content of this document, assess the security risk:
        1. Is this document legitimate or potentially malicious?
        2. What is the risk level (low/medium/high)?
        3. What actions should be taken?
        """
        analyses['risk_assessment'] = self.vlm.generate(screenshot, risk_prompt)

        return analyses


class VisualMalwareAnalyzer:
    """
    Analiza interfaces de malware usando VLMs.
    """

    def __init__(self, vlm: LLaVAWrapper):
        self.vlm = vlm

        self.malware_categories = [
            "ransomware",
            "banking_trojan",
            "spyware",
            "adware",
            "fake_antivirus",
            "tech_support_scam",
            "cryptominer",
            "legitimate_software"
        ]

    def analyze_ui(
        self,
        screenshot: Image.Image
    ) -> Dict[str, Any]:
        """
        Analiza UI de software potencialmente malicioso.
        """
        # Clasificacion
        classification_prompt = f"""
        Analyze this software interface screenshot and classify it into one of these categories:
        {', '.join(self.malware_categories)}

        Provide:
        1. The category
        2. Confidence level (low/medium/high)
        3. Key indicators that led to this classification
        """
        classification = self.vlm.generate(screenshot, classification_prompt)

        # Comportamiento esperado
        behavior_prompt = """
        Based on this interface, describe:
        1. What actions does this software appear to perform?
        2. What permissions or access might it request?
        3. What data might it collect or modify?
        """
        behavior = self.vlm.generate(screenshot, behavior_prompt)

        # Tacticas de ingenieria social
        social_prompt = """
        Identify any social engineering tactics used in this interface:
        - Fear tactics (urgency, threats)
        - Authority appeals (fake official branding)
        - Trust indicators (fake reviews, certifications)
        - Deceptive UI patterns (dark patterns)
        """
        social_engineering = self.vlm.generate(screenshot, social_prompt)

        return {
            'classification': classification,
            'expected_behavior': behavior,
            'social_engineering_tactics': social_engineering
        }


class PhishingPageAnalyzer:
    """
    Analisis detallado de paginas de phishing.
    """

    def __init__(self, vlm: LLaVAWrapper, blip_retrieval: BLIPRetrieval):
        self.vlm = vlm
        self.retrieval = blip_retrieval

    def full_analysis(
        self,
        screenshot: Image.Image,
        url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analisis completo de pagina sospechosa.
        """
        results = {}

        # 1. Identificar marca objetivo
        brand_prompt = """
        Identify the brand or organization this page appears to impersonate:
        1. What brand/company is shown?
        2. What visual elements indicate this brand?
        3. Is the branding consistent with the official brand?
        """
        results['brand_analysis'] = self.vlm.generate(screenshot, brand_prompt)

        # 2. Analisis de formulario
        form_prompt = """
        Analyze any forms on this page:
        1. What information is being requested?
        2. Are there unusual fields for this type of page?
        3. Is there a submit button and what does it say?
        """
        results['form_analysis'] = self.vlm.generate(screenshot, form_prompt)

        # 3. Indicadores visuales de phishing
        indicators_prompt = """
        List specific visual indicators of phishing:
        1. URL bar (if visible) - is it suspicious?
        2. Certificate/security indicators
        3. Grammar or spelling errors
        4. Low quality images or inconsistent design
        5. Missing or fake security badges
        """
        results['phishing_indicators'] = self.vlm.generate(screenshot, indicators_prompt)

        # 4. OCR de texto relevante
        ocr_prompt = """
        Read and transcribe:
        1. The URL in the address bar (if visible)
        2. Any error messages or warnings
        3. Form field labels
        4. Button text
        """
        results['text_extraction'] = self.vlm.generate(screenshot, ocr_prompt)

        # 5. Scoring final
        scoring_prompt = """
        Based on all observations, provide a phishing score from 0-100 where:
        - 0-30: Likely legitimate
        - 31-60: Suspicious, needs investigation
        - 61-100: Likely phishing

        Provide the score and a brief justification.
        """
        results['phishing_score'] = self.vlm.generate(screenshot, scoring_prompt)

        # 6. Agregar URL analysis si se proporciona
        if url:
            results['provided_url'] = url
            url_prompt = f"""
            The URL of this page is: {url}

            Analyze if the URL is consistent with the brand shown in the image.
            Is this a legitimate domain for this brand?
            """
            results['url_analysis'] = self.vlm.generate(screenshot, url_prompt)

        return results
```

## Metricas de Evaluacion para VLMs

```python
from typing import Dict, List
import numpy as np


class VLMEvaluator:
    """
    Evaluador para Vision-Language Models.
    """

    @staticmethod
    def compute_captioning_metrics(
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """
        Computa metricas de captioning: BLEU, METEOR, CIDEr, etc.

        Args:
            predictions: Lista de captions generados
            references: Lista de listas de referencias por imagen
        """
        from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
        from nltk.translate.meteor_score import meteor_score
        import nltk

        # Tokenizar
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split() for ref in refs] for refs in references]

        # BLEU scores
        bleu_1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0))
        bleu_4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))

        # METEOR (promedio)
        meteor_scores = []
        for pred, refs in zip(predictions, references):
            score = max(meteor_score([ref], pred) for ref in refs)
            meteor_scores.append(score)
        meteor = np.mean(meteor_scores)

        return {
            'bleu_1': bleu_1,
            'bleu_4': bleu_4,
            'meteor': meteor
        }

    @staticmethod
    def compute_vqa_accuracy(
        predictions: List[str],
        ground_truths: List[List[str]]
    ) -> Dict[str, float]:
        """
        Computa accuracy para VQA.
        Usa soft accuracy: min(#humans_that_said_answer / 3, 1)
        """
        accuracies = []

        for pred, gts in zip(predictions, ground_truths):
            pred_normalized = pred.lower().strip()
            gt_normalized = [gt.lower().strip() for gt in gts]

            # Contar cuantos ground truths coinciden
            matches = sum(1 for gt in gt_normalized if pred_normalized == gt)

            # Soft accuracy
            accuracy = min(matches / 3, 1.0)
            accuracies.append(accuracy)

        return {
            'vqa_accuracy': np.mean(accuracies),
            'exact_match': np.mean([acc == 1.0 for acc in accuracies])
        }

    @staticmethod
    def compute_retrieval_metrics(
        image_to_text_ranks: List[int],
        text_to_image_ranks: List[int]
    ) -> Dict[str, float]:
        """
        Metricas de retrieval: Recall@K, MRR.
        """
        def recall_at_k(ranks: List[int], k: int) -> float:
            return np.mean([1 if r <= k else 0 for r in ranks])

        def mrr(ranks: List[int]) -> float:
            return np.mean([1.0 / r for r in ranks])

        i2t = {
            'i2t_r@1': recall_at_k(image_to_text_ranks, 1),
            'i2t_r@5': recall_at_k(image_to_text_ranks, 5),
            'i2t_r@10': recall_at_k(image_to_text_ranks, 10),
            'i2t_mrr': mrr(image_to_text_ranks)
        }

        t2i = {
            't2i_r@1': recall_at_k(text_to_image_ranks, 1),
            't2i_r@5': recall_at_k(text_to_image_ranks, 5),
            't2i_r@10': recall_at_k(text_to_image_ranks, 10),
            't2i_mrr': mrr(text_to_image_ranks)
        }

        return {**i2t, **t2i}
```

## Resumen

```
VISION-LANGUAGE MODELS - KEY TAKEAWAYS:

┌─────────────────────────────────────────────────────────────────┐
│                     ARQUITECTURAS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. BLIP / BLIP-2                                               │
│     ├── Unified pre-training (ITC + ITM + LM)                  │
│     ├── Q-Former para conexion eficiente vision-LLM            │
│     └── Excelente para captioning y retrieval                  │
│                                                                 │
│  2. LLaVA                                                       │
│     ├── Simple: Vision encoder + Linear + LLM                  │
│     ├── Visual instruction tuning                              │
│     └── Bueno para conversaciones y razonamiento               │
│                                                                 │
│  3. Flamingo                                                    │
│     ├── Perceiver Resampler para compresion                    │
│     ├── Gated cross-attention en LLM                           │
│     └── Soporta multiples imagenes intercaladas                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     TAREAS                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  - Image Captioning: Generar descripcion de imagen             │
│  - VQA: Responder preguntas sobre imagenes                     │
│  - Visual Reasoning: Inferencia compleja                       │
│  - OCR: Leer texto en imagenes                                 │
│  - Image-Text Retrieval: Busqueda cross-modal                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                APLICACIONES SEGURIDAD                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  - Phishing detection: Analisis visual de paginas              │
│  - Malware UI analysis: Clasificacion de interfaces            │
│  - Document analysis: Extraccion de IOCs                       │
│  - Brand impersonation: Deteccion de suplantacion              │
│  - Social engineering: Identificar tacticas                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

MEJORES PRACTICAS:
- BLIP-2 para eficiencia (Q-Former reduce tokens)
- LLaVA para conversaciones y tareas complejas
- Cuantizacion (4-bit, 8-bit) para recursos limitados
- Prompt engineering crucial para resultados
- Fine-tuning con datos especificos del dominio
```

## Referencias

1. "BLIP: Bootstrapping Language-Image Pre-training" - Li et al., 2022
2. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" - Li et al., 2023
3. "Visual Instruction Tuning" (LLaVA) - Liu et al., 2023
4. "Flamingo: a Visual Language Model for Few-Shot Learning" - Alayrac et al., 2022
5. "Qwen-VL: A Versatile Vision-Language Model" - Bai et al., 2023
