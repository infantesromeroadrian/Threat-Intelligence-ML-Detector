"""
Topic Discovery Domain Service.

Business logic for topic modeling and analysis.
Pure Python - no infrastructure dependencies.
"""

from __future__ import annotations

from ..entities import ThreatIntel, Topic


class TopicDiscoveryService:
    """
    Domain service for topic discovery logic.

    Coordinates topic modeling and analysis.
    Infrastructure-agnostic - uses ports for actual ML operations.
    """

    def validate_topic_quality(self, topic: Topic) -> bool:
        """
        Validate topic quality based on business rules.

        Args:
            topic: Topic to validate

        Returns:
            True if topic meets quality standards
        """
        # Business rules:
        # 1. Must have at least 3 keywords
        # 2. Must have coherence score > 0.3 (if available)
        # 3. Must be associated with at least 2 documents

        has_enough_keywords = len(topic.keywords) >= 3
        has_good_coherence = topic.coherence_score is None or topic.coherence_score > 0.3
        has_enough_docs = topic.document_count >= 2

        return has_enough_keywords and has_good_coherence and has_enough_docs

    def filter_significant_topics(self, topics: list[Topic]) -> list[Topic]:
        """
        Filter topics to only significant ones.

        Args:
            topics: List of topics

        Returns:
            Filtered list of significant topics
        """
        return [topic for topic in topics if topic.is_significant]

    def rank_topics_by_importance(self, topics: list[Topic]) -> list[Topic]:
        """
        Rank topics by importance (document count + coherence).

        Args:
            topics: List of topics

        Returns:
            Sorted list of topics (most important first)
        """

        def importance_score(topic: Topic) -> float:
            # Weight document count more than coherence
            doc_score = topic.document_count * 0.7
            coherence_score = (topic.coherence_score or 0.5) * 0.3
            return doc_score + coherence_score

        return sorted(topics, key=importance_score, reverse=True)

    def find_related_topics(
        self, topic: Topic, all_topics: list[Topic], threshold: float = 0.3
    ) -> list[Topic]:
        """
        Find topics related to a given topic based on keyword overlap.

        Args:
            topic: Source topic
            all_topics: All available topics
            threshold: Minimum overlap ratio (0.0-1.0)

        Returns:
            List of related topics
        """
        related = []
        source_keywords = set(topic.top_keywords)

        for other_topic in all_topics:
            if other_topic.topic_id == topic.topic_id:
                continue

            other_keywords = set(other_topic.top_keywords)
            overlap = len(source_keywords & other_keywords)
            overlap_ratio = overlap / len(source_keywords) if source_keywords else 0

            if overlap_ratio >= threshold:
                related.append(other_topic)

        return related

    def suggest_topic_labels(self, topic: Topic) -> list[str]:
        """
        Suggest human-readable labels for a topic based on keywords.

        Args:
            topic: Topic to label

        Returns:
            List of suggested labels
        """
        # Get top 3 keywords
        top_keywords = topic.top_keywords[:3]

        # Create label suggestions
        suggestions = [
            " + ".join(top_keywords[:2]),  # "ransomware + malware"
            f"{top_keywords[0].capitalize()} Threats",
            f"{top_keywords[0].capitalize()} & {top_keywords[1].capitalize()}",
        ]

        return suggestions

    def calculate_topic_trends(self, topics: list[Topic]) -> dict[str, int | float]:
        """
        Calculate trend statistics about topics.

        Args:
            topics: List of topics

        Returns:
            Dictionary with trend statistics
        """
        if not topics:
            return {
                "total_topics": 0,
                "avg_document_count": 0.0,
                "avg_coherence": 0.0,
                "labeled_topics": 0,
            }

        return {
            "total_topics": len(topics),
            "significant_topics": sum(1 for t in topics if t.is_significant),
            "avg_document_count": sum(t.document_count for t in topics) / len(topics),
            "avg_coherence": sum(t.coherence_score or 0.0 for t in topics) / len(topics),
            "labeled_topics": sum(1 for t in topics if t.is_labeled),
            "unlabeled_topics": sum(1 for t in topics if not t.is_labeled),
        }
