# ML Models Directory

This directory contains pre-trained machine learning models for the Threat Intelligence Aggregator.

## Models

### 1. LDA Topic Model (`lda/`)

**Purpose**: Discovers latent topics in threat intelligence documents.

**Files**:
- `lda_model.gensim` - Trained LDA model
- `lda_dictionary.dict` - Vocabulary dictionary
- `metadata.json` - Training metadata (coherence, num_topics, date)

**Usage**:
```python
from infrastructure.adapters.ml_models import LDATopicModeler

modeler = LDATopicModeler()  # Auto-loads from lda/lda_model.gensim
topics = modeler.predict_topics(document)
```

**Retrain**:
```bash
make train-lda
```

---

### 2. Word2Vec Similarity Search (`word2vec/`)

**Purpose**: Finds semantically similar threat intelligence documents.

**Files**:
- `word2vec.model` - Trained Word2Vec embeddings
- `metadata.json` - Training metadata (vector_size, vocabulary, date)

**Usage**:
```python
from infrastructure.adapters.ml_models import Word2VecSimilaritySearch

w2v = Word2VecSimilaritySearch()  # Auto-loads from word2vec/word2vec.model
similar = w2v.find_similar_documents(document, top_n=5)
```

**Retrain**:
```bash
make train-word2vec
```

---

## Training

### Initial Setup (Mock Data)

Train models with synthetic threat intelligence data:

```bash
# Train both models
make train-models

# Or train individually
make train-lda
make train-word2vec
```

This generates ~300 realistic mock threat intel documents and trains the models.

### Retraining with Real Data

Once you have real threat intelligence data from NVD and OTX:

1. **Configure API keys** in `.env`:
   ```bash
   NVD_API_KEY=your_nvd_key
   OTX_API_KEY=your_otx_key
   ```

2. **Scrape real data**:
   ```bash
   make scrape-cves
   ```

3. **Retrain models**:
   ```bash
   make train-models
   ```

### Manual Training

For more control:

```bash
# Train with custom number of documents
python scripts/train_ml_models.py --all --num-docs 500

# Train only LDA
python scripts/train_ml_models.py --model lda

# Train only Word2Vec
python scripts/train_ml_models.py --model word2vec
```

---

## Model Metadata

Each model directory contains a `metadata.json` file with training information:

```json
{
  "num_topics": 10,
  "num_documents": 300,
  "coherence_score": 0.4532,
  "trained_at": "2026-01-17T02:30:00Z",
  "model_version": "1.0",
  "data_source": "mock"
}
```

---

## Performance

### Without Pre-trained Models
- **Startup time**: ~40 minutes (train LDA + Word2Vec each time)
- **Memory**: High (training in RAM)

### With Pre-trained Models (current setup)
- **Startup time**: ~30 seconds (load from disk)
- **Memory**: Low (models loaded once)

---

## Storage

**Size estimates**:
- LDA model: ~5-20 MB (depends on vocabulary size)
- Word2Vec model: ~10-50 MB (depends on vocabulary + vector dimensions)
- **Total**: ~15-70 MB

---

## Version Control

Models can be version controlled in Git:

**Option A**: Commit models (if <50 MB total)
```bash
git add src/threat_intelligence_aggregator/models/
git commit -m "feat: add pre-trained LDA and Word2Vec models"
```

**Option B**: Use Git LFS (if >50 MB)
```bash
git lfs track "*.gensim" "*.model" "*.dict"
git add .gitattributes
git add src/threat_intelligence_aggregator/models/
git commit -m "feat: add pre-trained models (Git LFS)"
```

**Option C**: External storage (production)
- Upload to S3/GCS/Azure Blob
- Download during deployment
- Use DVC for data versioning

---

## Troubleshooting

### Models not loading

If you see:
```
ℹ️  No pre-trained LDA model found at .../models/lda/lda_model.gensim
```

**Solution**: Train the models first:
```bash
make train-models
```

### Training takes too long

**Mock data** (default): 5-10 minutes  
**Real data** (1000+ docs): 20-40 minutes

**Speed up**:
- Reduce `--num-docs` for mock data
- Use faster hardware (more CPU cores, SSD)
- Consider pre-trained models (commit to repo)

### Out of memory

**Solution**:
- Reduce batch size in training script
- Increase system RAM
- Train on separate machine with more memory

---

## API Endpoints (Future)

Planned API endpoints for model management:

```
POST   /api/models/lda/train       - Trigger LDA training
GET    /api/models/lda/status      - Get LDA model info
POST   /api/models/word2vec/train  - Trigger Word2Vec training
GET    /api/models/word2vec/status - Get Word2Vec info
```

---

**Last Updated**: 2026-01-17  
**Model Version**: 1.0  
**Data Source**: Mock (synthetic threat intelligence)
