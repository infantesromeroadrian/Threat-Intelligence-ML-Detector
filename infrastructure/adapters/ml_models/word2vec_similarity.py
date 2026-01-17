"""
Word2Vec-based similarity search adapter.

Uses gensim Word2Vec to find similar documents and words in the threat intelligence corpus.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from ....domain.entities import ThreatIntel
from ....domain.ports.modelers import SimilaritySearchPort
from ...config.logging_config import get_logger

logger = get_logger(__name__)

# Reproducibility
SEED = 42
np.random.seed(SEED)


class Word2VecSimilaritySearch:
    """
    Word2Vec similarity search implementation.

    Uses gensim Word2Vec to:
    1. Train word embeddings from threat intelligence documents
    2. Find semantically similar documents
    3. Find similar words (e.g., "ransomware" → "malware", "trojan")
    4. Generate document vectors by averaging word vectors
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
    ) -> None:
        """
        Initialize Word2Vec model.

        Args:
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            epochs: Training epochs
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

        self.model: Word2Vec | None = None
        self.doc_vectors: dict[str, np.ndarray] = {}  # document_id -> vector

        # Try to load pre-trained model
        # __file__ is in infrastructure/adapters/ml_models/, need to go up to threat_intelligence_aggregator/
        model_path = (
            Path(__file__).parent.parent.parent.parent / "models" / "word2vec" / "word2vec.model"
        )
        if model_path.exists():
            logger.info(
                f"📥 Loading pre-trained Word2Vec model from {model_path}",
                source="Word2VecSimilaritySearch",
            )
            try:
                self.load_model(str(model_path))
                logger.info(
                    "✅ Pre-trained Word2Vec model loaded successfully",
                    source="Word2VecSimilaritySearch",
                )
            except Exception as e:
                logger.warning(
                    f"⚠️  Could not load pre-trained model: {e}. Will train on first use.",
                    source="Word2VecSimilaritySearch",
                )
        else:
            logger.info(
                f"ℹ️  No pre-trained Word2Vec model found at {model_path}. Will train on first use.",
                source="Word2VecSimilaritySearch",
            )

        logger.info(
            "✅ Word2Vec similarity search initialized",
            source="Word2VecSimilaritySearch",
            vector_size=vector_size,
            window=window,
        )

    def _preprocess_text(self, text: str) -> list[str]:
        """
        Preprocess text for Word2Vec.

        Args:
            text: Raw text

        Returns:
            List of tokens
        """
        # Remove URLs, emails, special chars
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # Tokenize (gensim simple_preprocess: lowercase, min_len=2, max_len=15)
        tokens = simple_preprocess(text, deacc=True)

        return tokens

    def train(self, documents: list[ThreatIntel]) -> None:
        """
        Train Word2Vec model on documents.

        Args:
            documents: List of threat intelligence documents
        """
        if not documents:
            logger.warning(
                "⚠️ No documents provided for training",
                source="Word2VecSimilaritySearch",
            )
            return

        logger.info(
            "ℹ️ Training Word2Vec model...",
            source="Word2VecSimilaritySearch",
            num_documents=len(documents),
        )

        # Preprocess documents
        sentences = []
        for doc in documents:
            text = f"{doc.title} {doc.content}"
            tokens = self._preprocess_text(text)
            sentences.append(tokens)

        # Train Word2Vec
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=SEED,
        )

        # Build document vectors (average of word vectors)
        self._build_document_vectors(documents)

        vocab_size = len(self.model.wv)
        logger.info(
            "✅ Word2Vec training complete",
            source="Word2VecSimilaritySearch",
            vocab_size=vocab_size,
            num_documents=len(documents),
        )

    def _build_document_vectors(self, documents: list[ThreatIntel]) -> None:
        """
        Build document vectors by averaging word vectors.

        Args:
            documents: List of documents
        """
        if self.model is None:
            logger.error(
                "❌ Model not trained",
                source="Word2VecSimilaritySearch",
            )
            return

        self.doc_vectors = {}

        for doc in documents:
            text = f"{doc.title} {doc.content}"
            tokens = self._preprocess_text(text)

            # Get word vectors for tokens in vocabulary
            word_vecs = []
            for token in tokens:
                if token in self.model.wv:
                    word_vecs.append(self.model.wv[token])

            # Average vectors (or zero vector if no valid words)
            if word_vecs:
                doc_vector = np.mean(word_vecs, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)

            self.doc_vectors[doc.document_id] = doc_vector

        logger.info(
            "✅ Document vectors built",
            source="Word2VecSimilaritySearch",
            num_vectors=len(self.doc_vectors),
        )

    def find_similar_documents(
        self, query: ThreatIntel, top_n: int = 10
    ) -> list[tuple[str, float]]:
        """
        Find similar documents based on content.

        Args:
            query: Query document
            top_n: Number of similar documents to return

        Returns:
            List of (document_id, similarity_score) tuples, sorted by similarity
        """
        if self.model is None or not self.doc_vectors:
            logger.warning(
                "⚠️ Model not trained or no document vectors",
                source="Word2VecSimilaritySearch",
            )
            return []

        # Get query vector
        text = f"{query.title} {query.content}"
        tokens = self._preprocess_text(text)

        word_vecs = []
        for token in tokens:
            if token in self.model.wv:
                word_vecs.append(self.model.wv[token])

        if not word_vecs:
            logger.warning(
                "⚠️ No valid words in query",
                source="Word2VecSimilaritySearch",
            )
            return []

        query_vector = np.mean(word_vecs, axis=0)

        # Calculate cosine similarity with all documents
        similarities = []
        for doc_id, doc_vector in self.doc_vectors.items():
            if doc_id == query.document_id:
                continue  # Skip self

            # Cosine similarity
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)

            if norm_query == 0 or norm_doc == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query_vector, doc_vector) / (norm_query * norm_doc))

            similarities.append((doc_id, similarity))

        # Sort by similarity (descending) and take top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_n]

        logger.info(
            "✅ Found similar documents",
            source="Word2VecSimilaritySearch",
            query_id=query.document_id,
            num_results=len(results),
        )

        return results

    def find_similar_words(self, word: str, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Find similar words based on embeddings.

        Args:
            word: Query word
            top_n: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples
        """
        if self.model is None:
            logger.warning(
                "⚠️ Model not trained",
                source="Word2VecSimilaritySearch",
            )
            return []

        word_lower = word.lower()

        if word_lower not in self.model.wv:
            logger.warning(
                "⚠️ Word not in vocabulary",
                source="Word2VecSimilaritySearch",
                word=word,
            )
            return []

        try:
            # Get most similar words
            similar = self.model.wv.most_similar(word_lower, topn=top_n)

            logger.info(
                "✅ Found similar words",
                source="Word2VecSimilaritySearch",
                word=word,
                num_results=len(similar),
            )

            return similar

        except Exception as e:
            logger.error(
                "❌ Error finding similar words",
                source="Word2VecSimilaritySearch",
                word=word,
                error=str(e),
            )
            return []

    def get_word_vector(self, word: str) -> list[float] | None:
        """
        Get word embedding vector.

        Args:
            word: Word to vectorize

        Returns:
            Word vector or None if word not in vocabulary
        """
        if self.model is None:
            logger.warning(
                "⚠️ Model not trained",
                source="Word2VecSimilaritySearch",
            )
            return None

        word_lower = word.lower()

        if word_lower not in self.model.wv:
            return None

        vector = self.model.wv[word_lower]
        return vector.tolist()

    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.

        Args:
            path: Path to save model (can be file or directory)
        """
        if self.model is None:
            logger.error(
                "❌ No model to save",
                source="Word2VecSimilaritySearch",
            )
            return

        path_obj = Path(path)

        # Determine if path is directory or file
        if path_obj.suffix == ".model":
            # Path is already a file (e.g., "models/word2vec/word2vec.model")
            model_path = path_obj
            save_dir = path_obj.parent
        else:
            # Path is a directory (e.g., "models/word2vec/")
            save_dir = path_obj
            model_path = path_obj / "word2vec.model"

        # Ensure directory exists
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save Word2Vec model
        self.model.save(str(model_path))

        # Save document vectors (always in same directory as model)
        vectors_path = save_dir / "doc_vectors.npy"
        np.save(str(vectors_path), self.doc_vectors)

        logger.info(
            "✅ Model saved",
            source="Word2VecSimilaritySearch",
            path=str(model_path),
        )

    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.

        Args:
            path: Path to load model from (can be file or directory)
        """
        path_obj = Path(path)

        # Determine if path is directory or file
        if path_obj.suffix == ".model":
            # Path is already a file (e.g., "models/word2vec/word2vec.model")
            model_path = path_obj
            load_dir = path_obj.parent
        else:
            # Path is a directory (e.g., "models/word2vec/")
            load_dir = path_obj
            model_path = path_obj / "word2vec.model"

        if not model_path.exists():
            logger.error(
                "❌ Word2Vec model file not found",
                source="Word2VecSimilaritySearch",
                path=str(model_path),
            )
            return

        self.model = Word2Vec.load(str(model_path))

        # Load document vectors (always in same directory as model)
        vectors_path = load_dir / "doc_vectors.npy"
        if vectors_path.exists():
            self.doc_vectors = np.load(str(vectors_path), allow_pickle=True).item()
        else:
            logger.warning(
                "⚠️ Document vectors not found, will need to rebuild",
                source="Word2VecSimilaritySearch",
            )
            self.doc_vectors = {}

        logger.info(
            "✅ Model loaded",
            source="Word2VecSimilaritySearch",
            path=str(model_path),
            vocab_size=len(self.model.wv),
            num_doc_vectors=len(self.doc_vectors),
        )
