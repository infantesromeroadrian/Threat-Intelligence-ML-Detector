"""
ML Modeler ports (interfaces) for machine learning models.

These are abstract interfaces that infrastructure adapters must implement.
"""

from __future__ import annotations

from typing import Protocol

from ..entities import ThreatIntel, Topic


class TopicModelerPort(Protocol):
    """Interface for topic modeling (LDA)."""

    def train(self, documents: list[ThreatIntel]) -> list[Topic]:
        """
        Train topic model on documents.

        Args:
            documents: List of threat intelligence documents

        Returns:
            List of discovered topics
        """
        ...

    def predict_topics(self, document: ThreatIntel, top_n: int = 3) -> list[tuple[int, float]]:
        """
        Predict topic distribution for a document.

        Args:
            document: Document to analyze
            top_n: Number of top topics to return

        Returns:
            List of (topic_number, probability) tuples
        """
        ...

    def get_topic_keywords(self, topic_number: int, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Get top keywords for a topic.

        Args:
            topic_number: Topic number
            top_n: Number of keywords to return

        Returns:
            List of (word, probability) tuples
        """
        ...

    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        ...

    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        ...

    def get_coherence_score(self) -> float:
        """Get model coherence score."""
        ...


class SeverityClassifierPort(Protocol):
    """Interface for severity classification (BERT)."""

    def train(
        self,
        texts: list[str],
        labels: list[str],
        validation_split: float = 0.2,
        epochs: int = 3,
    ) -> dict[str, float]:
        """
        Train severity classifier.

        Args:
            texts: Training texts
            labels: Training labels (severity levels)
            validation_split: Validation data fraction
            epochs: Number of training epochs

        Returns:
            Training metrics
        """
        ...

    def predict(self, text: str) -> str:
        """
        Predict severity for text.

        Args:
            text: Text to classify

        Returns:
            Predicted severity level
        """
        ...

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Predict severity with confidence score.

        Args:
            text: Text to classify

        Returns:
            Tuple of (severity, confidence)
        """
        ...

    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        ...

    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        ...


class SimilaritySearchPort(Protocol):
    """Interface for similarity search (Word2Vec)."""

    def train(self, documents: list[ThreatIntel]) -> None:
        """
        Train word embedding model.

        Args:
            documents: List of documents for training
        """
        ...

    def find_similar_documents(
        self, query: ThreatIntel, top_n: int = 10
    ) -> list[tuple[str, float]]:
        """
        Find similar documents based on content.

        Args:
            query: Query document
            top_n: Number of similar documents to return

        Returns:
            List of (document_id, similarity_score) tuples
        """
        ...

    def find_similar_words(self, word: str, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Find similar words based on embeddings.

        Args:
            word: Query word
            top_n: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples
        """
        ...

    def get_word_vector(self, word: str) -> list[float] | None:
        """
        Get word embedding vector.

        Args:
            word: Word to vectorize

        Returns:
            Word vector or None if word not in vocabulary
        """
        ...

    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        ...

    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        ...
