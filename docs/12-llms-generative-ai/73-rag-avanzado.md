# 73. RAG Avanzado - Retrieval-Augmented Generation

## Tabla de Contenidos

1. [Introduccion a RAG](#introduccion)
2. [Arquitectura RAG Completa](#arquitectura)
3. [Chunking Strategies](#chunking)
4. [Embedding Models](#embeddings)
5. [Vector Databases](#vector-dbs)
6. [Retrieval Avanzado](#retrieval-avanzado)
7. [Reranking](#reranking)
8. [Evaluacion con RAGAS](#evaluacion)
9. [Guardrails y Seguridad](#guardrails)
10. [Implementacion con LangChain](#implementacion)
11. [Aplicaciones en Ciberseguridad](#ciberseguridad)

---

## 1. Introduccion a RAG {#introduccion}

RAG permite a los LLMs acceder a conocimiento externo actualizado, reduciendo alucinaciones y mejorando precision.

### Por Que RAG?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      POR QUE RAG?                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LIMITACIONES DE LLMs:                                                     │
│  - Knowledge cutoff (datos hasta fecha de entrenamiento)                   │
│  - No acceso a datos privados/propietarios                                 │
│  - Alucinaciones cuando no saben                                           │
│  - Fine-tuning costoso y lento                                             │
│                                                                             │
│  RAG SOLUCIONA:                                                            │
│  + Acceso a datos actualizados                                             │
│  + Datos privados sin fine-tuning                                          │
│  + Respuestas con fuentes verificables                                     │
│  + Actualizacion facil (solo cambiar documentos)                           │
│                                                                             │
│  COMPARACION:                                                               │
│  ┌───────────────┬──────────────┬──────────────┬───────────────┐           │
│  │ Aspecto       │ Solo LLM     │ Fine-tuning  │ RAG           │           │
│  ├───────────────┼──────────────┼──────────────┼───────────────┤           │
│  │ Actualizacion │ Imposible    │ Costoso      │ Facil         │           │
│  │ Datos privados│ No           │ Si (riesgo)  │ Si (seguro)   │           │
│  │ Citas/fuentes │ No           │ No           │ Si            │           │
│  │ Alucinaciones │ Alto         │ Medio        │ Bajo          │           │
│  │ Costo         │ Bajo         │ Alto         │ Medio         │           │
│  └───────────────┴──────────────┴──────────────┴───────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Arquitectura RAG Completa {#arquitectura}

### Pipeline RAG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ARQUITECTURA RAG COMPLETA                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FASE 1: INDEXING (Offline)                                                │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  [Documentos] ──► [Chunking] ──► [Embedding] ──► [Vector DB]    │       │
│  │                                                                  │       │
│  │  PDF, HTML,       Split en      Modelo de       Qdrant,         │       │
│  │  TXT, etc.        fragmentos    embeddings      Chroma, etc.    │       │
│  │                                                                  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  FASE 2: RETRIEVAL + GENERATION (Online)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  [Query] ──► [Query Embedding] ──► [Vector Search]              │       │
│  │                                           │                      │       │
│  │                                           ▼                      │       │
│  │                                    [Top-K Chunks]                │       │
│  │                                           │                      │       │
│  │                                           ▼                      │       │
│  │                                    [Reranking] (opcional)        │       │
│  │                                           │                      │       │
│  │                                           ▼                      │       │
│  │  [Prompt] = System + Context + Query                            │       │
│  │                                           │                      │       │
│  │                                           ▼                      │       │
│  │                                    [LLM Generation]              │       │
│  │                                           │                      │       │
│  │                                           ▼                      │       │
│  │                                    [Response + Sources]          │       │
│  │                                                                  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  COMPONENTES AVANZADOS:                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  - Query Rewriting: Reformular query para mejor retrieval       │       │
│  │  - Hybrid Search: Combinar vector + keyword search              │       │
│  │  - Multi-Query: Generar queries alternativas                    │       │
│  │  - Self-RAG: Modelo decide cuando usar retrieval                │       │
│  │  - CRAG: Corrective RAG (verificar relevancia)                  │       │
│  │  - Guardrails: Validar input/output                             │       │
│  │                                                                  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chunking Strategies {#chunking}

### Estrategias de Chunking

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ESTRATEGIAS DE CHUNKING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. FIXED SIZE CHUNKING                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ Divide por numero fijo de caracteres/tokens                     │       │
│  │                                                                  │       │
│  │ [====500 chars====][====500 chars====][====500 chars====]       │       │
│  │                     ↑ overlap 50 ↑                              │       │
│  │                                                                  │       │
│  │ Pros: Simple, predecible                                        │       │
│  │ Cons: Puede cortar oraciones/conceptos                          │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  2. RECURSIVE CHARACTER SPLITTING                                          │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ Divide por separadores jerarquicos: \n\n -> \n -> . ->          │       │
│  │                                                                  │       │
│  │ Intenta primero por parrafos, luego oraciones, luego palabras   │       │
│  │                                                                  │       │
│  │ Pros: Respeta estructura del documento                          │       │
│  │ Cons: Tamanos variables                                         │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  3. SEMANTIC CHUNKING                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ Divide basandose en similitud semantica entre oraciones         │       │
│  │                                                                  │       │
│  │ [Oracion 1, 2, 3] → similares → chunk 1                        │       │
│  │ [Oracion 4, 5]    → similares → chunk 2                        │       │
│  │                                                                  │       │
│  │ Pros: Chunks coherentes semanticamente                          │       │
│  │ Cons: Mas lento, requiere embeddings                           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  4. DOCUMENT-SPECIFIC CHUNKING                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ Usa estructura del documento:                                   │       │
│  │                                                                  │       │
│  │ - Markdown: por headers (#, ##, ###)                           │       │
│  │ - HTML: por tags (<section>, <article>)                        │       │
│  │ - Code: por funciones/clases                                   │       │
│  │ - PDF: por paginas/secciones                                   │       │
│  │                                                                  │       │
│  │ Pros: Chunks significativos                                     │       │
│  │ Cons: Requiere parsers especificos                             │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  TAMANOS RECOMENDADOS:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │ Tipo de contenido    │ Chunk size │ Overlap │ Notas            │       │
│  │──────────────────────┼────────────┼─────────┼──────────────────│       │
│  │ Documentos generales │ 500-1000   │ 50-100  │ Balance standard │       │
│  │ Codigo fuente        │ 200-500    │ 0-50    │ Por funcion      │       │
│  │ Legal/tecnico        │ 1000-2000  │ 200     │ Contexto largo   │       │
│  │ Q&A corto            │ 200-500    │ 50      │ Respuestas cortas│       │
│  │ Chat history         │ Por turno  │ 0       │ Mantener turnos  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion de Chunking

```python
from typing import Iterator
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """Un fragmento de documento."""
    content: str
    metadata: dict
    start_index: int
    end_index: int


class ChunkingStrategy:
    """Estrategias de chunking."""

    @staticmethod
    def fixed_size(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> list[Chunk]:
        """Chunking de tamano fijo."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(Chunk(
                content=chunk_text,
                metadata={"method": "fixed_size"},
                start_index=start,
                end_index=min(end, len(text))
            ))

            start = end - overlap

        return chunks

    @staticmethod
    def recursive_split(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: list[str] = None
    ) -> list[Chunk]:
        """Chunking recursivo por separadores."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        def split_text(text: str, separators: list[str]) -> list[str]:
            if not separators:
                return [text]

            sep = separators[0]
            if not sep:
                return list(text)

            splits = text.split(sep)
            result = []

            for split in splits:
                if len(split) <= chunk_size:
                    result.append(split)
                else:
                    result.extend(split_text(split, separators[1:]))

            return result

        raw_chunks = split_text(text, separators)

        # Combinar chunks pequenos y manejar overlap
        chunks = []
        current_chunk = ""
        current_start = 0

        for raw in raw_chunks:
            if len(current_chunk) + len(raw) <= chunk_size:
                current_chunk += raw
            else:
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk,
                        metadata={"method": "recursive"},
                        start_index=current_start,
                        end_index=current_start + len(current_chunk)
                    ))
                    # Overlap
                    overlap_text = current_chunk[-overlap:] if overlap else ""
                    current_start += len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + raw
                else:
                    current_chunk = raw

        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk,
                metadata={"method": "recursive"},
                start_index=current_start,
                end_index=current_start + len(current_chunk)
            ))

        return chunks

    @staticmethod
    def by_markdown_headers(text: str) -> list[Chunk]:
        """Chunking por headers de Markdown."""
        pattern = r'^(#{1,6})\s+(.+)$'
        chunks = []
        current_header = ""
        current_content = []
        current_start = 0

        for i, line in enumerate(text.split('\n')):
            match = re.match(pattern, line)
            if match:
                # Guardar chunk anterior
                if current_content:
                    chunks.append(Chunk(
                        content='\n'.join(current_content),
                        metadata={
                            "method": "markdown_headers",
                            "header": current_header
                        },
                        start_index=current_start,
                        end_index=current_start + len('\n'.join(current_content))
                    ))
                current_header = match.group(2)
                current_content = [line]
                current_start = i
            else:
                current_content.append(line)

        # Ultimo chunk
        if current_content:
            chunks.append(Chunk(
                content='\n'.join(current_content),
                metadata={
                    "method": "markdown_headers",
                    "header": current_header
                },
                start_index=current_start,
                end_index=len(text)
            ))

        return chunks


class SemanticChunker:
    """Chunking basado en similitud semantica."""

    def __init__(self, embedding_model, similarity_threshold: float = 0.7):
        self.embedding_model = embedding_model
        self.threshold = similarity_threshold

    def chunk(self, text: str) -> list[Chunk]:
        """Divide texto en chunks semanticamente coherentes."""
        import numpy as np

        # Dividir en oraciones
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Obtener embeddings
        embeddings = self.embedding_model.encode(sentences)

        # Agrupar por similitud
        chunks = []
        current_group = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            similarity = np.dot(current_embedding, embeddings[i]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
            )

            if similarity >= self.threshold:
                current_group.append(sentences[i])
                # Actualizar embedding promedio
                current_embedding = np.mean(embeddings[i-len(current_group)+1:i+1], axis=0)
            else:
                # Nuevo chunk
                chunks.append(Chunk(
                    content=' '.join(current_group),
                    metadata={"method": "semantic"},
                    start_index=0,
                    end_index=0
                ))
                current_group = [sentences[i]]
                current_embedding = embeddings[i]

        # Ultimo grupo
        if current_group:
            chunks.append(Chunk(
                content=' '.join(current_group),
                metadata={"method": "semantic"},
                start_index=0,
                end_index=0
            ))

        return chunks
```

---

## 4. Embedding Models {#embeddings}

### Comparativa de Modelos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODELOS DE EMBEDDINGS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Modelo                │ Dim  │ Max Tokens │ Performance │ Notas           │
│  ──────────────────────┼──────┼────────────┼─────────────┼─────────────────│
│  text-embedding-3-large│ 3072 │ 8191       │ Excelente   │ OpenAI, costoso │
│  text-embedding-3-small│ 1536 │ 8191       │ Muy bueno   │ OpenAI, barato  │
│  voyage-2              │ 1024 │ 4000       │ Excelente   │ Voyage AI       │
│  BGE-large-en-v1.5     │ 1024 │ 512        │ Muy bueno   │ Open source     │
│  E5-large-v2           │ 1024 │ 512        │ Muy bueno   │ Open source     │
│  GTE-large             │ 1024 │ 512        │ Muy bueno   │ Alibaba         │
│  all-MiniLM-L6-v2      │ 384  │ 256        │ Bueno       │ Rapido, pequeno │
│  nomic-embed-text      │ 768  │ 8192       │ Muy bueno   │ Open source     │
│                                                                             │
│  PARA CIBERSEGURIDAD:                                                      │
│  - SecureBERT: Pre-entrenado en texto de seguridad                        │
│  - CyBERT: Especializado en cyber threat intelligence                     │
│  - Fine-tuned BGE en datos de seguridad                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion

```python
from abc import ABC, abstractmethod
import numpy as np


class EmbeddingModel(ABC):
    """Interfaz para modelos de embeddings."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Genera embeddings para lista de textos."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension de los embeddings."""
        pass


class OpenAIEmbeddings(EmbeddingModel):
    """Embeddings de OpenAI."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self._dimension = 1536 if "small" in model else 3072

    def encode(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return np.array([e.embedding for e in response.data])

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerEmbeddings(EmbeddingModel):
    """Embeddings con sentence-transformers (local)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    @property
    def dimension(self) -> int:
        return self._dimension


class HuggingFaceEmbeddings(EmbeddingModel):
    """Embeddings con modelos de HuggingFace."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self._dimension = self.model.config.hidden_size

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return embeddings.cpu().numpy()

    @property
    def dimension(self) -> int:
        return self._dimension
```

---

## 5. Vector Databases {#vector-dbs}

### Comparativa

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASES                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Database     │ Tipo      │ Escala      │ Mejor para                       │
│  ─────────────┼───────────┼─────────────┼──────────────────────────────────│
│  Chroma       │ Embedded  │ Pequena     │ Prototipos, desarrollo           │
│  FAISS        │ Library   │ Grande      │ Research, alto rendimiento       │
│  Qdrant       │ Server    │ Grande      │ Produccion, features avanzados   │
│  Pinecone     │ Cloud     │ Grande      │ Managed, facil escalado          │
│  Weaviate     │ Server    │ Grande      │ Hybrid search, GraphQL           │
│  Milvus       │ Server    │ Muy grande  │ Enterprise, alta escala          │
│  pgvector     │ Extension │ Media       │ Si ya usas PostgreSQL            │
│                                                                             │
│  METRICAS DE SIMILITUD:                                                    │
│  - Cosine: Mejor para texto (ignora magnitud)                             │
│  - Euclidean (L2): Distancia geometrica                                   │
│  - Dot Product: Rapido, requiere normalizacion                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion

```python
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    """Resultado de busqueda vectorial."""
    id: str
    content: str
    metadata: dict
    score: float


class VectorStore(ABC):
    """Interfaz para vector stores."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict], contents: list[str]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass


class ChromaVectorStore(VectorStore):
    """Vector store con ChromaDB."""

    def __init__(self, collection_name: str, persist_directory: Optional[str] = None):
        import chromadb

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict], contents: list[str]) -> None:
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=contents
        )

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=1 - results['distances'][0][i]  # Convertir distancia a similitud
            ))

        return search_results

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)


class QdrantVectorStore(VectorStore):
    """Vector store con Qdrant."""

    def __init__(self, collection_name: str, host: str = "localhost", port: int = 6333, dimension: int = 1536):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        # Crear coleccion si no existe
        collections = self.client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )

    def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict], contents: list[str]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={"content": contents[i], **metadatas[i]}
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k
        )

        return [
            SearchResult(
                id=str(r.id),
                content=r.payload.get("content", ""),
                metadata={k: v for k, v in r.payload.items() if k != "content"},
                score=r.score
            )
            for r in results
        ]

    def delete(self, ids: list[str]) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={"points": [int(id) for id in ids]}
        )
```

---

## 6. Retrieval Avanzado {#retrieval-avanzado}

### Tecnicas Avanzadas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TECNICAS DE RETRIEVAL AVANZADO                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. HYBRID SEARCH (Vector + Keyword)                                       │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  Query ──┬──► Vector Search ──► Results 1                       │       │
│  │          │                                                       │       │
│  │          └──► BM25 Search   ──► Results 2                       │       │
│  │                     │                                            │       │
│  │                     ▼                                            │       │
│  │              Reciprocal Rank Fusion                             │       │
│  │                     │                                            │       │
│  │                     ▼                                            │       │
│  │              Final Ranked Results                               │       │
│  │                                                                  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  2. MULTI-QUERY RETRIEVAL                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  Original Query ──► LLM ──► Query Variations                    │       │
│  │                              │                                   │       │
│  │                     ┌────────┼────────┐                         │       │
│  │                     ▼        ▼        ▼                         │       │
│  │                  Query 1  Query 2  Query 3                      │       │
│  │                     │        │        │                         │       │
│  │                     ▼        ▼        ▼                         │       │
│  │                [Results] [Results] [Results]                    │       │
│  │                     │        │        │                         │       │
│  │                     └────────┼────────┘                         │       │
│  │                              ▼                                   │       │
│  │                    Merge + Deduplicate                          │       │
│  │                                                                  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  3. CONTEXTUAL COMPRESSION                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                                                                  │       │
│  │  Retrieved Chunks ──► LLM Compressor ──► Relevant Excerpts      │       │
│  │                                                                  │       │
│  │  "Extract only the parts relevant to: {query}"                  │       │
│  │                                                                  │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion

```python
from typing import Callable
import numpy as np


class HybridRetriever:
    """Combina busqueda vectorial con BM25."""

    def __init__(self, vector_store: VectorStore, documents: list[str]):
        self.vector_store = vector_store
        self.documents = documents

        # Inicializar BM25
        from rank_bm25 import BM25Okapi
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 10,
        alpha: float = 0.5  # Balance entre vector y BM25
    ) -> list[SearchResult]:
        """Busqueda hibrida."""
        # Vector search
        vector_results = self.vector_store.search(query_embedding, k=k*2)

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalizar scores
        vector_scores = {r.id: r.score for r in vector_results}
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

        # Reciprocal Rank Fusion
        combined_scores = {}

        for result in vector_results:
            combined_scores[result.id] = alpha * result.score

        for i, score in enumerate(bm25_scores):
            doc_id = str(i)
            normalized_bm25 = score / max_bm25
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * normalized_bm25
            else:
                combined_scores[doc_id] = (1 - alpha) * normalized_bm25

        # Ordenar y retornar top-k
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]

        results = []
        for doc_id in sorted_ids:
            idx = int(doc_id)
            results.append(SearchResult(
                id=doc_id,
                content=self.documents[idx] if idx < len(self.documents) else "",
                metadata={},
                score=combined_scores[doc_id]
            ))

        return results


class MultiQueryRetriever:
    """Genera multiples queries y combina resultados."""

    def __init__(self, llm, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.llm = llm
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def generate_queries(self, original_query: str, n: int = 3) -> list[str]:
        """Genera variaciones de la query."""
        prompt = f"""Generate {n} different search queries that could help answer this question.
Each query should approach the question from a different angle.

Original question: {original_query}

Generate {n} alternative search queries (one per line):"""

        response = self.llm.generate(prompt)
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        return [original_query] + queries[:n]

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Busqueda con multiples queries."""
        queries = self.generate_queries(query)

        all_results = {}
        for q in queries:
            embedding = self.embedding_model.encode([q])[0]
            results = self.vector_store.search(embedding, k=k)

            for result in results:
                if result.id not in all_results:
                    all_results[result.id] = result
                else:
                    # Boost score si aparece en multiples queries
                    all_results[result.id].score += result.score

        # Ordenar por score combinado
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:k]


class ContextualCompressor:
    """Comprime contexto para solo incluir partes relevantes."""

    def __init__(self, llm):
        self.llm = llm

    def compress(self, query: str, documents: list[str]) -> list[str]:
        """Extrae solo las partes relevantes de cada documento."""
        compressed = []

        for doc in documents:
            prompt = f"""Given the following document and question, extract ONLY the sentences
that are directly relevant to answering the question. If nothing is relevant, respond with "NOT_RELEVANT".

Question: {query}

Document:
{doc}

Relevant excerpts:"""

            response = self.llm.generate(prompt)

            if "NOT_RELEVANT" not in response:
                compressed.append(response.strip())

        return compressed
```

---

## 7. Reranking {#reranking}

### Modelos de Reranking

```python
class Reranker:
    """Reranker basado en cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        top_k: int = 5
    ) -> list[SearchResult]:
        """Reordena documentos usando cross-encoder."""
        pairs = [(query, doc.content) for doc in documents]
        scores = self.model.predict(pairs)

        # Actualizar scores y ordenar
        for i, doc in enumerate(documents):
            doc.score = float(scores[i])

        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)
        return sorted_docs[:top_k]


class CohereReranker:
    """Reranker usando Cohere API."""

    def __init__(self, api_key: str):
        import cohere
        self.client = cohere.Client(api_key)

    def rerank(
        self,
        query: str,
        documents: list[SearchResult],
        top_k: int = 5
    ) -> list[SearchResult]:
        """Reordena usando Cohere Rerank."""
        results = self.client.rerank(
            query=query,
            documents=[doc.content for doc in documents],
            top_n=top_k,
            model="rerank-english-v2.0"
        )

        reranked = []
        for r in results:
            doc = documents[r.index]
            doc.score = r.relevance_score
            reranked.append(doc)

        return reranked
```

---

## 8. Evaluacion con RAGAS {#evaluacion}

### Metricas de Evaluacion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    METRICAS DE EVALUACION RAG                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  METRICAS DE RETRIEVAL:                                                    │
│  - Context Precision: Proporcion de chunks relevantes recuperados          │
│  - Context Recall: Chunks relevantes / total relevantes disponibles        │
│  - MRR (Mean Reciprocal Rank): Posicion del primer resultado relevante    │
│  - NDCG: Normalized Discounted Cumulative Gain                             │
│                                                                             │
│  METRICAS DE GENERATION:                                                   │
│  - Faithfulness: Respuesta basada en el contexto (no alucinacion)         │
│  - Answer Relevancy: Respuesta relevante a la pregunta                     │
│  - Answer Correctness: Respuesta correcta vs ground truth                  │
│                                                                             │
│  METRICAS END-TO-END:                                                      │
│  - RAGAS Score: Combinacion de faithfulness, relevancy, etc.              │
│  - Human Evaluation: Evaluacion manual de calidad                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion con RAGAS

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGEvalSample:
    """Muestra para evaluacion RAG."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None


class RAGASEvaluator:
    """Evaluador usando metricas RAGAS."""

    def __init__(self, llm):
        self.llm = llm

    def evaluate_faithfulness(self, sample: RAGEvalSample) -> float:
        """
        Evalua si la respuesta es fiel al contexto.
        0 = Alucinacion, 1 = Completamente basado en contexto
        """
        prompt = f"""Given the context and answer, determine if the answer is faithful to the context.
The answer should only contain information present in the context.

Context:
{' '.join(sample.contexts)}

Answer:
{sample.answer}

Score from 0 to 1 (0 = contains information not in context, 1 = fully faithful):
Score:"""

        response = self.llm.generate(prompt)
        try:
            return float(response.strip().split()[-1])
        except:
            return 0.5

    def evaluate_relevancy(self, sample: RAGEvalSample) -> float:
        """
        Evalua si la respuesta es relevante a la pregunta.
        """
        prompt = f"""Given the question and answer, rate how relevant the answer is to the question.

Question: {sample.question}
Answer: {sample.answer}

Score from 0 to 1 (0 = completely irrelevant, 1 = perfectly relevant):
Score:"""

        response = self.llm.generate(prompt)
        try:
            return float(response.strip().split()[-1])
        except:
            return 0.5

    def evaluate_context_precision(self, sample: RAGEvalSample) -> float:
        """
        Evalua precision del contexto recuperado.
        """
        relevant_count = 0

        for ctx in sample.contexts:
            prompt = f"""Is this context relevant to answering the question?

Question: {sample.question}
Context: {ctx}

Answer only 'yes' or 'no':"""

            response = self.llm.generate(prompt).strip().lower()
            if 'yes' in response:
                relevant_count += 1

        return relevant_count / len(sample.contexts) if sample.contexts else 0

    def evaluate(self, samples: list[RAGEvalSample]) -> dict:
        """Evalua multiples muestras."""
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []

        for sample in samples:
            faithfulness_scores.append(self.evaluate_faithfulness(sample))
            relevancy_scores.append(self.evaluate_relevancy(sample))
            precision_scores.append(self.evaluate_context_precision(sample))

        return {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
            "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores),
            "context_precision": sum(precision_scores) / len(precision_scores),
            "ragas_score": (
                sum(faithfulness_scores) / len(faithfulness_scores) * 0.4 +
                sum(relevancy_scores) / len(relevancy_scores) * 0.3 +
                sum(precision_scores) / len(precision_scores) * 0.3
            )
        }
```

---

## 9. Guardrails y Seguridad {#guardrails}

```python
from typing import Optional
import re


class RAGGuardrails:
    """Guardrails para RAG pipelines."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.min_relevance_score = self.config.get("min_relevance_score", 0.5)

    def validate_query(self, query: str) -> tuple[bool, str]:
        """Valida query de entrada."""
        # Check longitud
        if len(query) > 1000:
            return False, "Query too long"

        # Check injection patterns
        injection_patterns = [
            r"ignore\s+.*instructions",
            r"system\s*:",
            r"<\s*script",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Potential injection detected"

        return True, "Valid"

    def validate_context(self, contexts: list[SearchResult]) -> list[SearchResult]:
        """Filtra contextos por relevancia y seguridad."""
        filtered = []

        for ctx in contexts:
            # Filtrar por score minimo
            if ctx.score < self.min_relevance_score:
                continue

            # Check contenido sensible
            if self._contains_sensitive_data(ctx.content):
                ctx.content = self._redact_sensitive(ctx.content)

            filtered.append(ctx)

        return filtered

    def _contains_sensitive_data(self, text: str) -> bool:
        """Detecta datos sensibles."""
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        return any(re.search(p, text) for p in patterns)

    def _redact_sensitive(self, text: str) -> str:
        """Redacta datos sensibles."""
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', text)
        text = re.sub(r'\b\d{16}\b', '[CARD REDACTED]', text)
        return text

    def validate_response(self, response: str, contexts: list[str]) -> tuple[bool, str]:
        """Valida respuesta generada."""
        # Check que no revela system prompt
        if "system:" in response.lower() or "instruction:" in response.lower():
            return False, "Response may contain system information"

        # Check alucinacion basica (respuesta muy diferente del contexto)
        context_words = set(' '.join(contexts).lower().split())
        response_words = set(response.lower().split())

        # Si menos del 20% de palabras de respuesta estan en contexto, posible alucinacion
        overlap = len(response_words & context_words) / len(response_words) if response_words else 0
        if overlap < 0.2:
            return False, "Possible hallucination detected"

        return True, "Valid"
```

---

## 10. Implementacion con LangChain {#implementacion}

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGPipeline:
    """Pipeline RAG completo con LangChain."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = None

    def index_documents(self, documents: list[str], persist_directory: str = "./chroma_db"):
        """Indexa documentos."""
        chunks = self.text_splitter.create_documents(documents)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )

    def create_qa_chain(self, k: int = 4):
        """Crea chain de QA."""
        if not self.vector_store:
            raise ValueError("Index documents first")

        prompt_template = """Use the following context to answer the question.
If you don't know the answer based on the context, say "I don't have enough information to answer."

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, question: str) -> dict:
        """Ejecuta query."""
        chain = self.create_qa_chain()
        result = chain({"query": question})

        return {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }
```

---

## 11. Aplicaciones en Ciberseguridad {#ciberseguridad}

```python
class SecurityRAG:
    """RAG especializado para ciberseguridad."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def index_security_docs(self, sources: dict[str, list[str]]):
        """
        Indexa documentos de seguridad de multiples fuentes.

        sources: {
            "cve": ["CVE descriptions..."],
            "mitre": ["MITRE ATT&CK descriptions..."],
            "playbooks": ["Incident response playbooks..."],
            "policies": ["Security policies..."]
        }
        """
        all_docs = []
        for source_type, docs in sources.items():
            for doc in docs:
                # Anadir metadata de tipo
                all_docs.append(f"[Source: {source_type}]\n{doc}")

        self.pipeline.index_documents(all_docs)

    def threat_lookup(self, query: str) -> dict:
        """Busca informacion de amenazas."""
        result = self.pipeline.query(
            f"What information do you have about: {query}? Include CVEs, MITRE techniques, and mitigations."
        )
        return result

    def incident_guidance(self, incident_type: str, details: str) -> dict:
        """Obtiene guia de respuesta a incidentes."""
        query = f"""I'm responding to a {incident_type} incident.
Details: {details}
What are the recommended response steps according to our playbooks?"""
        return self.pipeline.query(query)

    def policy_check(self, action: str) -> dict:
        """Verifica si una accion cumple con politicas."""
        query = f"According to our security policies, is the following action allowed? Action: {action}"
        return self.pipeline.query(query)


# Ejemplo de uso
SECURITY_DOCS = {
    "cve": [
        "CVE-2024-1234: Remote code execution in Apache Log4j. CVSS: 10.0. Affects versions 2.0-2.14.1. Mitigation: Update to 2.17.0 or later.",
        "CVE-2024-5678: SQL injection in WordPress plugin. CVSS: 7.5. Mitigation: Update plugin or disable.",
    ],
    "mitre": [
        "T1190 - Exploit Public-Facing Application: Adversaries may attempt to take advantage of a weakness in an Internet-facing computer.",
        "T1059.001 - PowerShell: Adversaries may abuse PowerShell commands and scripts for execution.",
    ],
    "playbooks": [
        "Ransomware Response Playbook: 1. Isolate affected systems. 2. Preserve evidence. 3. Identify patient zero. 4. Begin recovery from backups.",
    ]
}
```

---

## Resumen

Este capitulo cubrio RAG avanzado:

1. **Arquitectura**: Pipeline completo de indexing y retrieval
2. **Chunking**: Estrategias fijas, recursivas, semanticas
3. **Embeddings**: Modelos y seleccion
4. **Vector DBs**: Chroma, Qdrant, comparativa
5. **Retrieval avanzado**: Hybrid search, multi-query
6. **Reranking**: Cross-encoders para mejorar precision
7. **Evaluacion**: RAGAS y metricas
8. **Guardrails**: Seguridad en RAG
9. **Ciberseguridad**: Aplicaciones especializadas

### Recursos

- LangChain: https://langchain.com
- LlamaIndex: https://llamaindex.ai
- RAGAS: https://github.com/explodinggradients/ragas
- Chroma: https://www.trychroma.com

---

*Siguiente: [74. Agents y Tool Use](./74-agents-tool-use.md)*
