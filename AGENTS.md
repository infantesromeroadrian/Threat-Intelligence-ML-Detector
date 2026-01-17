# Agent Guidelines for AI-RedTeam-Course

## Project Overview

**AI/ML Security & Red Team Course** - A comprehensive educational program to master Machine Learning and Artificial Intelligence engineering from fundamentals to state-of-the-art, with cybersecurity applications.

**Purpose**: Training material to become a **Senior ML/AI Engineer (top 1%)**  
**Format**: 
- 87 technical documents (107K+ lines of theory)
- Jupyter notebooks with practical implementations
- Trained models and datasets
- 22 Cursor rules for engineering excellence

**Python**: 3.10+ | **Environment**: `ml-course-venv/`

---

## 📚 Course Structure (15 Modules)

### Fundamentals (Modules 1-3)
1. **Regression** (7 docs): Linear, polynomial, Ridge/Lasso, gradient descent
2. **Classification** (6 docs): Logistic regression, trees, RF, SVM, XGBoost, Naive Bayes
3. **Clustering** (5 docs): K-means, DBSCAN, hierarchical, GMM

### Core Deep Learning (Modules 4-8)
4. **Neural Networks** (6 docs): MLP, CNN, RNN, autoencoders, transformers, attention
5. **Time Series** (6 docs): ARIMA, Prophet, DL for sequences, anomaly detection
6. **Dimensionality Reduction** (5 docs): PCA, t-SNE, UMAP, feature selection
7. **NLP** (8 docs): Text preprocessing, TF-IDF, embeddings, BERT, NER, sentiment
8. **Computer Vision** (8 docs): CNNs, ResNet/VGG, transfer learning, object detection, ViT, GANs

### Production & Advanced (Modules 9-11)
9. **MLOps** (7 docs): MLflow, feature stores, serving, CI/CD, monitoring, drift, A/B testing
10. **Mathematics** (4 docs): Convex optimization, Bayesian statistics, information theory
11. **Reinforcement Learning** (6 docs): Q-learning, DQN, policy gradient, multi-agent, security apps

### State-of-the-Art (Modules 12-15)
12. **LLMs & Generative AI** (6 docs): Architecture, fine-tuning, RLHF, prompt engineering, RAG, **agents**
13. **Generative Models** (4 docs): Diffusion models, Stable Diffusion, VAEs, Flow models
14. **Graph Neural Networks** (4 docs): GCN, GAT, GraphSAGE, cybersecurity applications
15. **Multimodal AI** (4 docs): CLIP, vision-language models, audio-multimodal

**Evaluation**: Metrics for classifiers, regression, ranking

---

## 🎯 How Agents Should Use This Repo

### Primary Role: Teaching & Explaining
When a student asks about a concept, you MUST:

1. **Reference the docs first**: Check `docs/XX-topic/` for existing theory
2. **Explain with visual diagrams**: The docs use ASCII art extensively - replicate this style
3. **Connect theory to practice**: Link concepts to corresponding notebooks
4. **Progressive complexity**: Start simple, build up (like docs do)
5. **Use cybersecurity examples**: This is the course's domain focus

### Teaching Style (Mirror the Docs)
The documentation follows a specific pedagogical pattern:

```
1. Visual diagrams (ASCII art boxes and trees)
2. Intuition BEFORE math
3. Clear notation tables (symbols, meanings, examples)
4. Code examples (Python with type hints)
5. Practical cybersecurity applications
6. Common pitfalls (❌ DON'T / ✅ DO)
```

**Example structure** (from regression docs):
```
# Topic Title

## 1. Intuition & Motivation
[ASCII diagram showing concept visually]
[Real-world cybersecurity example]

## 2. Mathematical Foundation
[Notation table with symbols]
[Equations with explanations]

## 3. Implementation
[Python code with comments]
[Step-by-step walkthrough]

## 4. Evaluation & Pitfalls
[When to use / when NOT to use]
[Common mistakes]
```

---

## 🛠️ Build, Run & Test Commands

### Environment Setup
```bash
# Activate virtual environment
source ml-course-venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
# Or with uv (preferred):
uv sync
```

### Running Notebooks
```bash
# Launch Jupyter Lab (recommended)
jupyter lab

# Or Jupyter Notebook
jupyter notebook

# Execute notebook from CLI
jupyter nbconvert --execute --to notebook --inplace \
  notebooks/05-CNN-MalwareClassification.ipynb

# Execute and convert to HTML
jupyter nbconvert --to html --execute \
  notebooks/02-RegresionLogistica-DeteccionSPAM-ConPersistencia.ipynb
```

### Code Quality (Python Scripts)
```bash
# Format code (replaces Black + isort)
ruff format .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Type checking (strict mode required)
mypy <script_name>.py --strict
```

### Testing (If test suite exists)
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_malware_detector.py

# Run single test function
pytest tests/test_malware_detector.py::test_model_prediction

# Run with markers
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## 📝 Code Style Guidelines

### Python (PEP 8 + Modern 3.10+)

#### Imports
```python
# Standard library
import os
from pathlib import Path

# Third-party (data & ML)
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Local modules (if any)
from utils.preprocessing import clean_text
```
- Group: stdlib → third-party → local (blank lines between)
- **NO wildcard imports** (`from x import *`)
- Absolute imports from project root

#### Type Hints (Required for scripts)
```python
from __future__ import annotations

import numpy as np

def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from file."""
    ...

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 10
) -> dict[str, float]:
    """Train model and return metrics."""
    ...
```
- Modern syntax: `list[str]`, `dict[str, Any]` (not `List`, `Dict`)
- Unions: `str | None` (not `Optional[str]`)
- Always add `-> None` for void functions

#### Error Handling
```python
# ✅ Specific exceptions
try:
    model = torch.load(model_path)
except FileNotFoundError:
    logger.error("❌ Model file not found: %s", model_path)
    raise
except RuntimeError as e:
    logger.error("❌ Failed to load model: %s", e)
    raise

# ❌ NEVER bare except
try:
    model = torch.load(model_path)
except:  # DON'T
    pass
```

#### Logging (Not print!)
```python
import logging

logger = logging.getLogger(__name__)

logger.info("ℹ️ Training started with %d samples", len(X_train))
logger.warning("⚠️ Class imbalance detected: %s", class_dist)
logger.error("❌ Failed to save model: %s", error)
```
- Use emoji prefixes: `ℹ️` `⚠️` `❌`
- **NO `print()` in production code** (notebooks OK for exploration)
- **Never log secrets, API keys, PII**

---

## 🧠 ML/AI Specific Rules

### Data Splitting (Critical!)
```python
# ✅ Proper train/val/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# ❌ NEVER evaluate on training data
model.fit(X, y)
print(f"Accuracy: {model.score(X, y)}")  # Wrong!
```

### Reproducibility
```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

### Model Saving with Metadata
```python
from datetime import datetime
from pathlib import Path

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = Path(f"models/malware_classifier_{timestamp}.pth")

torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'accuracy': test_accuracy,
    'config': config_dict,
    'seed': SEED,
    'dataset_version': 'v1.0'
}, model_path)
```

### Metrics (Never just accuracy!)
- **Imbalanced data**: Precision, recall, F1, confusion matrix, ROC-AUC
- **Always compare to baseline** (random, majority class, simple model)
- **Document why you chose these metrics**

---

## 🔐 Security & Best Practices

### MUST ✅
- Externalize config (env vars, config files)
- **Never hardcode secrets, API keys, credentials**
- Use structured logging (no `print()` in production)
- Apply type hints consistently
- Set random seeds for reproducibility
- Document model hyperparameters

### MUST NOT ❌
- Commit secrets or `.env` files
- Use wildcard imports
- Swallow exceptions silently
- Log sensitive data (PII, passwords, tokens)
- Use mutable default arguments (`def f(x=[]):`)
- Evaluate models on training data
- Train without holdout test set

---

## 📂 Project Structure

```
AI-RedTeam-Course/
├── docs/                   # 87 theory documents (107K+ lines)
│   ├── 01regresion/       # Regression fundamentals (7 docs)
│   ├── 02-clasificacion/  # Classification algorithms (6 docs)
│   ├── 03-clustering/     # Unsupervised learning (5 docs)
│   ├── 04-redes-neuronales/ # Neural networks (6 docs)
│   ├── 05-series-temporales/ # Time series (6 docs)
│   ├── 06-reduccion-dimensionalidad/ # PCA, t-SNE (5 docs)
│   ├── 07-NLP/            # Natural Language Processing (8 docs)
│   ├── 08-Computer-Vision/ # CNNs, ViT, GANs (8 docs)
│   ├── 09-mlops-produccion/ # MLOps practices (7 docs)
│   ├── 10-matematicas-ml/ # Math foundations (4 docs)
│   ├── 11-reinforcement-learning/ # RL (6 docs)
│   ├── 12-llms-generative-ai/ # LLMs, RAG, agents (6 docs)
│   ├── 13-modelos-generativos/ # Diffusion, VAEs (4 docs)
│   ├── 14-graph-neural-networks/ # GNNs (4 docs)
│   ├── 15-multimodal-ai/ # CLIP, VLMs (4 docs)
│   └── evaluacion/        # Metrics & evaluation
├── notebooks/             # Jupyter notebooks (6 notebooks)
│   ├── 01-*.ipynb        # Regression examples
│   ├── 02-*.ipynb        # SPAM detection
│   ├── 03-*.ipynb        # Phishing detection
│   ├── 04-*.ipynb        # Anomaly detection (Random Forest)
│   ├── 05-*.ipynb        # Malware classification (CNN)
│   └── 06-*.ipynb        # Sentiment analysis (NLP)
├── data/                  # Datasets (gitignored)
│   ├── email.csv
│   ├── phishing_email.csv
│   ├── KDD+.txt
│   └── malimg_paper_dataset_imgs/
├── models/                # Trained models (.pth, .keras, .joblib)
├── .cursor/rules/         # 22 engineering rules
├── ml-course-venv/        # Virtual environment
├── CLAUDE.md              # Code review rules (Gentleman-AI)
└── README.md
```

---

## 🎓 Cursor Rules Reference

22 comprehensive rules in `.cursor/rules/` - **Key ones for teaching**:

- `00_master_workflow_rule.mdc` - Workflow orchestration
- `01_security_baseline_rule.mdc` - Security & privacy (ALWAYS ACTIVE)
- `07_clean_code_principles.mdc` - Code structure, size, responsibilities
- `08_cod_style.mdc` - Python style (PEP 8 + modern)
- `09_general_rules.mdc` - ML/MLOps engineering practices
- `13_workflow_modern.mdc` - ML workflow and experimentation
- `19_python_logging_rule.mdc` - Logging structure
- `20_ai_security_rule.mdc` - AI/LLM-specific security
- `21_cybersecurity_engineering_rule.mdc` - Cybersecurity practices

---

## 🤖 Teaching Agent Behavior

### When a Student Asks "Explain X":
1. Check if `docs/` has content on topic X
2. If YES: Summarize the doc's approach, use its visual style
3. If NO: Explain from first principles, create ASCII diagrams
4. Link to related notebooks for hands-on practice
5. Provide cybersecurity use case

### When a Student Asks "How do I implement X":
1. Check if a notebook already covers it
2. Provide working code with type hints
3. Explain each step (don't just dump code)
4. Add comments explaining "why", not just "what"
5. Include proper train/test splits and evaluation

### When Reviewing Student Code:
1. Follow `CLAUDE.md` rules (code review standards)
2. Check for ML pitfalls (data leakage, no test set, etc.)
3. Verify reproducibility (seeds set?)
4. Ensure proper logging (no `print()` in production)
5. Validate security (no hardcoded secrets)

---

## 📊 Quick Reference

| Task | Command |
|------|---------|
| Activate venv | `source ml-course-venv/bin/activate` |
| Launch Jupyter | `jupyter lab` |
| Run notebook | `jupyter nbconvert --execute --to notebook --inplace <file.ipynb>` |
| Format Python | `ruff format .` |
| Lint Python | `ruff check --fix .` |
| Type check | `mypy <script>.py --strict` |
| Run tests | `pytest` |
| Single test | `pytest tests/test_file.py::test_function` |

---

**Version**: 2.0  
**Last Updated**: 2026-01-11  
**Maintained by**: Adrian Infantes Romero  
**Course Focus**: Top 1% ML/AI Engineering with Cybersecurity Applications

**Note to Agents**: This is an educational project. Your role is to **teach, explain, and guide**, not just generate code. Always reference the 87 theory documents in `docs/` when explaining concepts.
