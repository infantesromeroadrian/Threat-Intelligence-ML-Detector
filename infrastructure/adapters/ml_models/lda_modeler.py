"""
LDA Topic Modeler implementation using gensim.

Implements TopicModelerPort for discovering topics in threat intelligence documents.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ....domain.entities import ThreatIntel, Topic, TopicWord
from ....domain.ports.modelers import TopicModelerPort
from ....infrastructure.config.logging_config import get_logger
from ....infrastructure.config.settings import settings

logger = get_logger(__name__)


class LDATopicModeler:
    """
    LDA Topic Modeler using gensim.

    Discovers latent topics in threat intelligence documents using
    Latent Dirichlet Allocation (LDA).
    """

    def __init__(self) -> None:
        """Initialize LDA topic modeler."""
        self.num_topics = settings.lda_num_topics
        self.passes = settings.lda_passes
        self.iterations = settings.lda_iterations

        self.model: Any = None
        self.dictionary: Any = None
        self.corpus: Any = None

        # Try to load pre-trained model
        # __file__ is in infrastructure/adapters/ml_models/, need to go up to threat_intelligence_aggregator/
        model_path = (
            Path(__file__).parent.parent.parent.parent / "models" / "lda" / "lda_model.gensim"
        )
        if model_path.exists():
            logger.info(
                f"📥 Loading pre-trained LDA model from {model_path}",
                source="LDATopicModeler",
            )
            try:
                self.load_model(str(model_path))
                logger.info(
                    "✅ Pre-trained LDA model loaded successfully",
                    source="LDATopicModeler",
                )
            except Exception as e:
                logger.warning(
                    f"⚠️  Could not load pre-trained model: {e}. Will train on first use.",
                    source="LDATopicModeler",
                )
        else:
            logger.info(
                f"ℹ️  No pre-trained LDA model found at {model_path}. Will train on first use.",
                source="LDATopicModeler",
            )

        # Stop words (common words to ignore)
        self.stop_words = {
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "i",
            "it",
            "for",
            "not",
            "on",
            "with",
            "he",
            "as",
            "you",
            "do",
            "at",
            "this",
            "but",
            "his",
            "by",
            "from",
            "they",
            "we",
            "say",
            "her",
            "she",
            "or",
            "an",
            "will",
            "my",
            "one",
            "all",
            "would",
            "there",
            "their",
            "what",
            "so",
            "up",
            "out",
            "if",
            "about",
            "who",
            "get",
            "which",
            "go",
            "me",
            "when",
            "make",
            "can",
            "like",
            "time",
            "no",
            "just",
            "him",
            "know",
            "take",
            "people",
            "into",
            "year",
            "your",
            "good",
            "some",
            "could",
            "them",
            "see",
            "other",
            "than",
            "then",
            "now",
            "look",
            "only",
            "come",
            "its",
            "over",
            "think",
            "also",
            "back",
            "after",
            "use",
            "two",
            "how",
            "our",
            "work",
        }

        logger.info(
            f"✅ Initialized LDA Topic Modeler",
            source="LDATopicModeler",
            num_topics=self.num_topics,
            passes=self.passes,
        )

    def train(self, documents: list[ThreatIntel]) -> list[Topic]:
        """
        Train LDA model on threat intelligence documents.

        Args:
            documents: List of threat intelligence documents

        Returns:
            List of discovered topics
        """
        logger.info(
            f"🧠 Training LDA model on {len(documents)} documents",
            source="LDATopicModeler",
            num_docs=len(documents),
        )

        if len(documents) < 2:
            logger.warning(
                "⚠️  Need at least 2 documents for topic modeling",
                source="LDATopicModeler",
            )
            return []

        try:
            import gensim
            from gensim import corpora
            from gensim.models import LdaModel
        except ImportError:
            logger.error(
                "❌ gensim not installed - cannot train LDA model",
                source="LDATopicModeler",
            )
            return []

        # Step 1: Preprocess documents
        logger.info("📝 Preprocessing documents...")
        texts = [self._preprocess_text(doc.content) for doc in documents]

        # Step 2: Create dictionary
        logger.info("📚 Creating dictionary...")
        self.dictionary = corpora.Dictionary(texts)

        # Filter extremes (words that appear in <5% or >50% of docs)
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)

        # Step 3: Create corpus (bag of words)
        logger.info("🗂️  Creating corpus...")
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

        # Step 4: Train LDA model
        logger.info(f"🎓 Training LDA with {self.num_topics} topics...")
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            iterations=self.iterations,
            random_state=42,
            per_word_topics=True,
        )

        # Step 5: Calculate coherence score
        logger.info("📊 Calculating coherence score...")
        coherence_score = self._calculate_coherence(texts)

        # Step 6: Extract topics
        logger.info("🔍 Extracting topics...")
        topics = self._extract_topics(documents, coherence_score)

        logger.info(
            f"✅ Training complete - discovered {len(topics)} topics",
            source="LDATopicModeler",
            num_topics=len(topics),
            coherence=coherence_score,
        )

        return topics

    def predict_topics(self, document: ThreatIntel, top_n: int = 3) -> list[tuple[int, float]]:
        """
        Predict topic distribution for a document.

        Args:
            document: Document to analyze
            top_n: Number of top topics to return

        Returns:
            List of (topic_number, probability) tuples
        """
        if self.model is None or self.dictionary is None:
            logger.warning(
                "⚠️  Model not trained - call train() first",
                source="LDATopicModeler",
            )
            return []

        # Preprocess
        text = self._preprocess_text(document.content)
        bow = self.dictionary.doc2bow(text)

        # Get topic distribution
        topic_dist = self.model.get_document_topics(bow)

        # Sort by probability and return top N
        sorted_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)
        return sorted_topics[:top_n]

    def get_topic_keywords(self, topic_number: int, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Get top keywords for a topic.

        Args:
            topic_number: Topic number
            top_n: Number of keywords to return

        Returns:
            List of (word, probability) tuples
        """
        if self.model is None:
            logger.warning(
                "⚠️  Model not trained",
                source="LDATopicModeler",
            )
            return []

        return self.model.show_topic(topic_number, topn=top_n)

    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            logger.warning("⚠️  No model to save", source="LDATopicModeler")
            return

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(model_path))
        self.dictionary.save(str(model_path.with_suffix(".dict")))

        logger.info(
            f"💾 Saved LDA model to {path}",
            source="LDATopicModeler",
        )

    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        try:
            from gensim.models import LdaModel
            from gensim import corpora

            model_path = Path(path)
            self.model = LdaModel.load(str(model_path))
            self.dictionary = corpora.Dictionary.load(str(model_path.with_suffix(".dict")))

            logger.info(
                f"✅ Loaded LDA model from {path}",
                source="LDATopicModeler",
            )
        except Exception as e:
            logger.error(
                f"❌ Failed to load model: {e}",
                source="LDATopicModeler",
            )

    def get_coherence_score(self) -> float:
        """Get model coherence score."""
        if self.model is None or self.corpus is None:
            return 0.0

        return self._calculate_coherence([])

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _preprocess_text(self, text: str) -> list[str]:
        """
        Preprocess text for topic modeling.

        Args:
            text: Raw text

        Returns:
            List of tokens
        """
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove emails
        text = re.sub(r"\S+@\S+", "", text)

        # Remove special characters (keep letters and spaces)
        text = re.sub(r"[^a-z\s]", " ", text)

        # Tokenize
        tokens = text.split()

        # Remove stop words and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        return tokens

    def _calculate_coherence(self, texts: list[list[str]]) -> float:
        """Calculate topic coherence score."""
        if self.model is None or self.corpus is None:
            return 0.0

        try:
            from gensim.models import CoherenceModel

            coherence_model = CoherenceModel(
                model=self.model,
                texts=texts if texts else [],
                dictionary=self.dictionary,
                coherence="c_v",
            )

            return coherence_model.get_coherence()
        except Exception as e:
            logger.warning(
                f"⚠️  Could not calculate coherence: {e}",
                source="LDATopicModeler",
            )
            return 0.5  # Default value

    def _extract_topics(self, documents: list[ThreatIntel], coherence: float) -> list[Topic]:
        """Extract Topic entities from trained model."""
        from datetime import datetime

        topics = []

        for topic_num in range(self.num_topics):
            # Get keywords
            keywords_tuples = self.get_topic_keywords(topic_num, top_n=10)
            keywords = [TopicWord(word=word, probability=prob) for word, prob in keywords_tuples]

            # Count documents for this topic
            doc_count = self._count_documents_for_topic(topic_num)

            topic = Topic(
                topic_id=f"topic_{topic_num}",
                topic_number=topic_num,
                keywords=keywords,
                coherence_score=coherence,
                document_count=doc_count,
                discovery_date=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            topics.append(topic)

        return topics

    def _count_documents_for_topic(self, topic_num: int, threshold: float = 0.3) -> int:
        """Count documents where this topic has significant weight."""
        if self.model is None or self.corpus is None:
            return 0

        count = 0
        for doc_bow in self.corpus:
            topic_dist = self.model.get_document_topics(doc_bow)
            for topic_id, prob in topic_dist:
                if topic_id == topic_num and prob >= threshold:
                    count += 1
                    break

        return count
