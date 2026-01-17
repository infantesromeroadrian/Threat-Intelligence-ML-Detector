#!/usr/bin/env python3
"""
Train ML models for Threat Intelligence Aggregator.

Trains LDA Topic Model and Word2Vec Similarity Search with mock threat intelligence data.
Pre-trained models reduce startup time from ~40 minutes to ~30 seconds.

Usage:
    python scripts/train_ml_models.py --model lda
    python scripts/train_ml_models.py --model word2vec
    python scripts/train_ml_models.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from threat_intelligence_aggregator.domain.entities import ThreatIntel, ThreatType, ThreatSeverity
from threat_intelligence_aggregator.infrastructure.adapters.ml_models import (
    LDATopicModeler,
    Word2VecSimilaritySearch,
)
from threat_intelligence_aggregator.infrastructure.config.logging_config import get_logger

logger = get_logger(__name__)

# Model paths
MODELS_DIR = project_root / "src" / "threat_intelligence_aggregator" / "models"
LDA_MODEL_PATH = MODELS_DIR / "lda" / "lda_model.gensim"
LDA_DICT_PATH = MODELS_DIR / "lda" / "lda_dictionary.dict"
LDA_METADATA_PATH = MODELS_DIR / "lda" / "metadata.json"

WORD2VEC_MODEL_PATH = MODELS_DIR / "word2vec" / "word2vec.model"
WORD2VEC_METADATA_PATH = MODELS_DIR / "word2vec" / "metadata.json"


def generate_mock_threat_intel(count: int = 300) -> list[ThreatIntel]:
    """
    Generate realistic mock threat intelligence documents.

    Creates synthetic threat intel with realistic cybersecurity content
    covering malware, phishing, APT, ransomware, and exploits.
    """
    logger.info(f"🎲 Generating {count} mock threat intelligence documents...")

    # Templates for realistic threat intel
    malware_templates = [
        "New {variant} malware discovered targeting {target}. Uses {technique} for persistence.",
        "{family} ransomware campaign detected encrypting files with {algo} algorithm.",
        "Trojan {name} drops backdoor using {method} injection technique.",
        "Banking trojan {variant} steals credentials from {browser} browsers.",
        "Cryptominer {name} exploits {vuln} to gain system access.",
    ]

    phishing_templates = [
        "Phishing campaign impersonating {brand} targets {industry} sector.",
        "Credential harvesting attack uses fake {service} login pages.",
        "Business email compromise targeting {role} with invoice fraud.",
        "Spear phishing campaign delivers {malware} via weaponized {filetype} attachments.",
        "Whaling attack targets C-level executives at {companies}.",
    ]

    apt_templates = [
        "APT{num} group launches cyber espionage campaign against {target}.",
        "{group} threat actor exploits zero-day in {software} for initial access.",
        "Nation-state actor {name} deploys custom backdoor in {sector} infrastructure.",
        "Advanced persistent threat targeting {industry} using {tool} framework.",
        "Cyber espionage group {name} steals intellectual property from {targets}.",
    ]

    exploit_templates = [
        "CVE-{year}-{num} critical vulnerability in {software} allows remote code execution.",
        "Zero-day exploit targeting {product} grants SYSTEM privileges.",
        "Buffer overflow vulnerability in {service} enables arbitrary code execution.",
        "SQL injection flaw in {app} allows database compromise.",
        "Privilege escalation exploit in {os} kernel grants root access.",
    ]

    ransomware_templates = [
        "{name} ransomware encrypts files and demands {amount} Bitcoin payment.",
        "Double extortion attack by {group} threatens data leak if ransom not paid.",
        "Ransomware-as-a-Service {name} sold on dark web forums.",
        "{variant} targets healthcare facilities causing service disruptions.",
        "Wiper malware disguised as ransomware destroys data at {targets}.",
    ]

    # Variables for randomization
    variants = ["Alpha", "Beta", "Gamma", "v2.0", "Pro", "Elite", "Dark", "Shadow"]
    families = ["LockBit", "BlackCat", "Conti", "REvil", "DarkSide", "Ryuk"]
    targets = [
        "financial institutions",
        "healthcare",
        "manufacturing",
        "energy sector",
        "government agencies",
    ]
    techniques = ["registry modification", "scheduled tasks", "DLL hijacking", "service creation"]
    methods = ["process hollowing", "reflective DLL", "thread injection"]
    browsers = ["Chrome", "Firefox", "Edge", "Safari"]
    vulns = ["CVE-2024-1234", "CVE-2023-5678", "unpatched vulnerability"]
    brands = ["Microsoft", "Amazon", "PayPal", "Bank of America", "DHL"]
    industries = ["finance", "retail", "healthcare", "manufacturing", "education"]
    roles = ["CFO", "IT Manager", "HR Director"]
    services = ["Office 365", "Google Workspace", "DocuSign"]
    groups = ["APT28", "APT29", "Lazarus", "FIN7", "Sandworm"]
    software = ["Apache", "Windows", "Linux kernel", "Adobe Reader"]

    documents = []

    for i in range(count):
        # Randomly select template type
        template_type = i % 5

        if template_type == 0:  # Malware
            import random

            template = random.choice(malware_templates)
            content = template.format(
                variant=random.choice(variants),
                target=random.choice(targets),
                technique=random.choice(techniques),
                family=random.choice(families),
                algo="AES-256",
                name=f"Malware{i}",
                method=random.choice(methods),
                browser=random.choice(browsers),
                vuln=random.choice(vulns),
            )
            threat_type = ThreatType.MALWARE
            title = f"Malware Campaign #{i}: {random.choice(families)}"

        elif template_type == 1:  # Phishing
            import random

            template = random.choice(phishing_templates)
            content = template.format(
                brand=random.choice(brands),
                industry=random.choice(industries),
                service=random.choice(services),
                role=random.choice(roles),
                malware=f"Trojan{i}",
                filetype=random.choice(["PDF", "DOCX", "XLSX"]),
                companies=random.choice(targets),
            )
            threat_type = ThreatType.PHISHING
            title = f"Phishing Campaign targeting {random.choice(industries)}"

        elif template_type == 2:  # APT
            import random

            template = random.choice(apt_templates)
            content = template.format(
                num=random.randint(28, 40),
                group=random.choice(groups),
                target=random.choice(targets),
                software=random.choice(software),
                sector=random.choice(industries),
                name=f"APT{random.randint(28, 40)}",
                industry=random.choice(industries),
                tool=random.choice(["Cobalt Strike", "Metasploit", "Empire"]),
                targets=random.choice(targets),
            )
            threat_type = ThreatType.APT
            title = f"APT Campaign: {random.choice(groups)}"

        elif template_type == 3:  # Exploit
            import random

            template = random.choice(exploit_templates)
            year = 2024
            num = random.randint(1000, 9999)
            content = template.format(
                year=year,
                num=num,
                software=random.choice(software),
                product=random.choice(["Windows Server", "Apache Tomcat", "nginx"]),
                service=random.choice(["SSH", "RDP", "HTTP"]),
                app=random.choice(["WordPress", "Joomla", "Drupal"]),
                os=random.choice(["Windows", "Linux", "macOS"]),
            )
            threat_type = ThreatType.VULNERABILITY
            title = f"CVE-{year}-{num} Critical Vulnerability"

        else:  # Ransomware
            import random

            template = random.choice(ransomware_templates)
            content = template.format(
                name=random.choice(families),
                amount=random.choice(["5", "10", "50", "100"]),
                group=random.choice(families),
                variant=random.choice(families),
                targets=random.choice(targets),
            )
            threat_type = ThreatType.RANSOMWARE
            title = f"{random.choice(families)} Ransomware Attack"

        # Create ThreatIntel entity
        pub_date = datetime(2024, 1, (i % 28) + 1)
        doc = ThreatIntel(
            document_id=f"mock_{i}",
            title=title,
            content=content,
            source="Mock Data Generator",
            threat_type=threat_type,
            severity=random.choice(
                [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH, ThreatSeverity.MEDIUM]
            ),
            published_date=pub_date,
            collected_at=pub_date,  # Same as published for mock data
            tags=[threat_type.value.lower(), "mock", random.choice(industries)],
        )

        documents.append(doc)

    logger.info(f"✅ Generated {len(documents)} threat intelligence documents")

    # Log distribution
    type_counts = {}
    for doc in documents:
        type_counts[doc.threat_type.value] = type_counts.get(doc.threat_type.value, 0) + 1

    for threat_type, count in type_counts.items():
        logger.info(f"  - {threat_type}: {count} documents")

    return documents


def train_lda_model(documents: list[ThreatIntel]) -> None:
    """Train LDA topic model and save to disk."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("  🧠 Training LDA Topic Model")
    logger.info("=" * 80)
    logger.info("")

    # Initialize modeler
    modeler = LDATopicModeler()

    # Train
    logger.info(f"📚 Training with {len(documents)} documents...")
    topics = modeler.train(documents)

    # Save model
    logger.info(f"💾 Saving LDA model to {LDA_MODEL_PATH}...")
    modeler.save_model(str(LDA_MODEL_PATH))

    # Save metadata
    metadata = {
        "num_topics": modeler.num_topics,
        "num_documents": len(documents),
        "coherence_score": modeler.get_coherence_score(),
        "trained_at": datetime.utcnow().isoformat(),
        "model_version": "1.0",
        "data_source": "mock",
    }

    with open(LDA_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"📄 Saved metadata to {LDA_METADATA_PATH}")
    logger.info("")
    logger.info(f"✅ LDA model training complete!")
    logger.info(f"   Topics discovered: {len(topics)}")
    logger.info(f"   Coherence score: {metadata['coherence_score']:.4f}")
    logger.info("")

    # Display top topics
    logger.info("🔍 Top 5 Topics:")
    for i, topic in enumerate(topics[:5]):
        keywords = ", ".join([kw.word for kw in topic.keywords[:5]])
        logger.info(f"  Topic {i}: {keywords}")
    logger.info("")


def train_word2vec_model(documents: list[ThreatIntel]) -> None:
    """Train Word2Vec model and save to disk."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("  🔤 Training Word2Vec Similarity Search")
    logger.info("=" * 80)
    logger.info("")

    # Initialize model
    w2v = Word2VecSimilaritySearch(
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
    )

    # Train
    logger.info(f"📚 Training with {len(documents)} documents...")
    w2v.train(documents)

    # Save model
    logger.info(f"💾 Saving Word2Vec model to {WORD2VEC_MODEL_PATH}...")
    w2v.save_model(str(WORD2VEC_MODEL_PATH))

    # Save metadata
    metadata = {
        "vector_size": w2v.vector_size,
        "window": w2v.window,
        "min_count": w2v.min_count,
        "epochs": w2v.epochs,
        "num_documents": len(documents),
        "vocabulary_size": len(w2v.model.wv) if w2v.model else 0,
        "trained_at": datetime.utcnow().isoformat(),
        "model_version": "1.0",
        "data_source": "mock",
    }

    with open(WORD2VEC_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"📄 Saved metadata to {WORD2VEC_METADATA_PATH}")
    logger.info("")
    logger.info(f"✅ Word2Vec model training complete!")
    logger.info(f"   Vocabulary size: {metadata['vocabulary_size']}")
    logger.info(f"   Vector dimensions: {metadata['vector_size']}")
    logger.info("")

    # Test similarity
    if w2v.model:
        try:
            similar_words = w2v.find_similar_words("ransomware", top_n=5)
            logger.info("🔍 Words similar to 'ransomware':")
            for word, score in similar_words:
                logger.info(f"  - {word}: {score:.4f}")
            logger.info("")
        except:
            logger.info("⚠️  Could not compute similar words (word not in vocabulary)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train ML models for Threat Intelligence Aggregator"
    )
    parser.add_argument(
        "--model",
        choices=["lda", "word2vec", "all"],
        default="all",
        help="Model to train (default: all)",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=300,
        help="Number of mock documents to generate (default: 300)",
    )

    args = parser.parse_args()

    logger.info("")
    logger.info("╔═══════════════════════════════════════════════════════════════╗")
    logger.info("║  🚀 Threat Intelligence Aggregator - ML Model Training       ║")
    logger.info("╚═══════════════════════════════════════════════════════════════╝")
    logger.info("")

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "lda").mkdir(exist_ok=True)
    (MODELS_DIR / "word2vec").mkdir(exist_ok=True)

    # Generate mock data
    documents = generate_mock_threat_intel(count=args.num_docs)

    # Train models
    if args.model in ["lda", "all"]:
        train_lda_model(documents)

    if args.model in ["word2vec", "all"]:
        train_word2vec_model(documents)

    logger.info("=" * 80)
    logger.info("✅ Model training complete!")
    logger.info("")
    logger.info(f"📂 Models saved to: {MODELS_DIR}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Models will be loaded automatically on app startup")
    logger.info("  2. To retrain with real data: python scripts/train_ml_models.py --all")
    logger.info("  3. View metadata: cat threat_intelligence_aggregator/models/*/metadata.json")
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)
