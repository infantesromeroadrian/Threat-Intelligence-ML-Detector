"""
BERT Severity Classifier implementation.

Implements SeverityClassifierPort using transformers (BERT).
Classifies threat severity based on text description.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....domain.ports.modelers import SeverityClassifierPort
from ....infrastructure.config.logging_config import get_logger
from ....infrastructure.config.settings import settings

logger = get_logger(__name__)


class BERTSeverityClassifier:
    """
    BERT-based severity classifier.

    Classifies threat descriptions into severity levels:
    - CRITICAL
    - HIGH
    - MEDIUM
    - LOW
    - INFO
    """

    # Severity label mapping
    LABEL_MAP = {
        0: "CRITICAL",
        1: "HIGH",
        2: "MEDIUM",
        3: "LOW",
        4: "INFO",
    }

    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    def __init__(self) -> None:
        """Initialize BERT classifier."""
        self.model_name = settings.bert_model
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = "cpu"

        # Check if CUDA is available
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass

        logger.info(
            f"✅ Initialized BERT Classifier",
            source="BERTSeverityClassifier",
            model=self.model_name,
            device=self.device,
        )

    def train(
        self,
        texts: list[str],
        labels: list[str],
        validation_split: float = 0.2,
        epochs: int = 3,
    ) -> dict[str, float]:
        """
        Train BERT classifier.

        Args:
            texts: Training texts
            labels: Training labels (severity levels)
            validation_split: Validation data fraction
            epochs: Number of training epochs

        Returns:
            Training metrics
        """
        logger.info(
            f"🧠 Training BERT classifier on {len(texts)} samples",
            source="BERTSeverityClassifier",
            num_samples=len(texts),
            epochs=epochs,
        )

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
            from sklearn.model_selection import train_test_split
            import numpy as np
        except ImportError as e:
            logger.error(
                f"❌ Required library not installed: {e}",
                source="BERTSeverityClassifier",
            )
            return {"error": "Missing dependencies"}

        # Convert labels to integers
        label_ids = [self.REVERSE_LABEL_MAP.get(label.upper(), 2) for label in labels]

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, label_ids, test_size=validation_split, random_state=42
        )

        # Load tokenizer and model
        logger.info(f"📥 Loading model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.LABEL_MAP),
        )

        # Move to device
        self.model.to(self.device)

        # Tokenize
        logger.info("🔤 Tokenizing texts...")
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        # Create datasets
        class SimpleDataset(torch.utils.data.Dataset):  # type: ignore
            def __init__(self, encodings: Any, labels: list[int]) -> None:
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx: int) -> dict[str, Any]:
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self) -> int:
                return len(self.labels)

        train_dataset = SimpleDataset(train_encodings, train_labels)
        val_dataset = SimpleDataset(val_encodings, val_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./bert_severity_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train
        logger.info(f"🎓 Training for {epochs} epochs...")
        train_result = trainer.train()

        # Evaluate
        logger.info("📊 Evaluating model...")
        eval_result = trainer.evaluate()

        metrics = {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result.get("eval_loss", 0.0),
            "epochs": epochs,
            "num_samples": len(texts),
        }

        logger.info(
            f"✅ Training complete",
            source="BERTSeverityClassifier",
            metrics=metrics,
        )

        return metrics

    def predict(self, text: str) -> str:
        """
        Predict severity for text.

        Args:
            text: Text to classify

        Returns:
            Predicted severity level
        """
        severity, _ = self.predict_with_confidence(text)
        return severity

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Predict severity with confidence score.

        Args:
            text: Text to classify

        Returns:
            Tuple of (severity, confidence)
        """
        if self.model is None or self.tokenizer is None:
            logger.warning(
                "⚠️  Model not trained - using heuristic",
                source="BERTSeverityClassifier",
            )
            return self._heuristic_classify(text)

        try:
            import torch
            import torch.nn.functional as F

            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)

                confidence, predicted = torch.max(probs, dim=-1)

                severity = self.LABEL_MAP[predicted.item()]
                conf_score = confidence.item()

            logger.info(
                f"🔮 Predicted severity: {severity} ({conf_score:.2f})",
                source="BERTSeverityClassifier",
            )

            return severity, conf_score

        except Exception as e:
            logger.error(
                f"❌ Prediction failed: {e}",
                source="BERTSeverityClassifier",
            )
            return self._heuristic_classify(text)

    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None or self.tokenizer is None:
            logger.warning(
                "⚠️  No model to save",
                source="BERTSeverityClassifier",
            )
            return

        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))

        logger.info(
            f"💾 Saved BERT model to {path}",
            source="BERTSeverityClassifier",
        )

    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            model_path = Path(path)

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.model.to(self.device)

            logger.info(
                f"✅ Loaded BERT model from {path}",
                source="BERTSeverityClassifier",
            )
        except Exception as e:
            logger.error(
                f"❌ Failed to load model: {e}",
                source="BERTSeverityClassifier",
            )

    # =========================================================================
    # Private Methods
    # =========================================================================

    @staticmethod
    def _heuristic_classify(text: str) -> tuple[str, float]:
        """
        Simple heuristic-based classification (fallback).

        Uses keyword matching when model is not available.
        """
        text_lower = text.lower()

        # Critical keywords
        if any(
            word in text_lower
            for word in [
                "critical",
                "severe",
                "emergency",
                "exploit",
                "zero-day",
                "remote code execution",
                "rce",
                "arbitrary code",
            ]
        ):
            return "CRITICAL", 0.8

        # High keywords
        if any(
            word in text_lower
            for word in [
                "high",
                "dangerous",
                "serious",
                "vulnerability",
                "attack",
                "malware",
                "ransomware",
                "backdoor",
            ]
        ):
            return "HIGH", 0.7

        # Medium keywords
        if any(
            word in text_lower
            for word in ["medium", "moderate", "warning", "issue", "flaw", "weakness", "exposure"]
        ):
            return "MEDIUM", 0.6

        # Low keywords
        if any(
            word in text_lower
            for word in ["low", "minor", "information", "disclosure", "informational"]
        ):
            return "LOW", 0.6

        # Default
        return "MEDIUM", 0.5
