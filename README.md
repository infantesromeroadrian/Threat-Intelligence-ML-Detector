# 🛡️ Threat Intelligence Aggregator

> Production-ready AI-powered cybersecurity threat intelligence platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Threat Intelligence Aggregator** is an enterprise-grade platform for collecting, analyzing, and managing cybersecurity threat intelligence using state-of-the-art AI/ML techniques.

---

## ✨ Features

### 🔐 CVE Management
- Automated CVE scraping from NVD
- CVSS score tracking
- Advanced filtering (severity, vendor, product, date range)
- Real-time statistics dashboard

### 🎯 IOC Extraction
- Automatic extraction using spaCy NER
- Support for IPs, domains, URLs, emails, file hashes
- Confidence scoring
- Context preservation

### ⚠️ Threat Intelligence
- AlienVault OTX integration
- Multi-source aggregation
- BERT-based severity classification
- Topic modeling with LDA

### 📚 Topic Discovery
- Automatic topic extraction from threat documents
- Coherence score calculation
- Manual topic labeling
- Significant topic filtering

### 🚨 Smart Alerts
- Intelligent alert generation
- Complete workflow management (NEW → ACKNOWLEDGED → RESOLVED)
- False positive tracking
- Actionable recommendations

### 🤖 ML Models
- **NER IOC Extractor**: spaCy + regex patterns
- **LDA Topic Modeler**: gensim-based topic discovery
- **BERT Classifier**: Severity classification (5 levels)
- **Word2Vec**: Document/word similarity search

### 🌐 REST API
- **35 endpoints** with full CRUD operations
- Auto-generated documentation (Swagger/ReDoc)
- Pagination & advanced filtering
- CORS-enabled

### 🎨 Web Dashboard
- Modern responsive UI
- Real-time API status
- Interactive filtering
- Alert workflow management

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (recommended)

### Run with Docker

```bash
# 1. Generate test data
python scripts/test_pipeline.py

# 2. Start all services
docker compose up -d --build

# 3. Access the application
open http://localhost              # Web Dashboard
open http://localhost:8000/docs    # API Documentation
```

### Run Locally

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate test data
python scripts/test_pipeline.py

# 4. Start API
uvicorn threat_intelligence_aggregator.infrastructure.api.main:app --reload

# 5. Serve frontend (new terminal)
cd frontend && python -m http.server 8080
```

---

## 📊 Architecture

```
Frontend (Nginx) → FastAPI → Application Layer → Domain Layer → Infrastructure
                    ↓                                              ↓
                  REST API                                    Adapters
                    ↓                                              ↓
              Pydantic DTOs                          ML Models | Repositories
                    ↓                                              ↓
            35 Endpoints                                      SQLite DB
```

**Key Principles**:
- ✅ Hexagonal Architecture (Ports & Adapters)
- ✅ Domain-Driven Design
- ✅ SOLID principles
- ✅ 100% type hints (mypy --strict ready)
- ✅ Zero infrastructure dependencies in domain

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern async web framework
- **SQLAlchemy** - ORM
- **Pydantic** - Data validation
- **spaCy** - NLP/NER
- **gensim** - Topic modeling
- **transformers** - BERT classification
- **PyTorch** - Deep learning

### Frontend
- **HTML5/CSS3** - Modern web standards
- **Vanilla JavaScript** - No frameworks
- **Nginx** - Web server & reverse proxy

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Orchestration
- **uv** - Fast Python package manager

---

## 📁 Project Structure

```
src/threat_intelligence_aggregator/
├── domain/              # Pure business logic (NO dependencies)
│   ├── entities/       # CVE, IOC, ThreatIntel, Topic, Alert
│   ├── ports/          # Interfaces (Protocol)
│   └── services/       # Business rules
├── application/        # Use cases
└── infrastructure/     # Technical implementation
    ├── adapters/      # ML, repos, scrapers
    ├── api/           # FastAPI routes
    └── config/        # Settings, logging

frontend/               # Web dashboard
├── index.html
├── css/main.css
└── js/
    ├── api.js         # API client
    └── app.js         # App logic
```

---

## 🧪 Testing

```bash
# Generate test data
python scripts/test_pipeline.py

# Verify database
sqlite3 threat_intel_test.db
> SELECT COUNT(*) FROM cves;
> SELECT COUNT(*) FROM iocs;

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/cves/stats
curl http://localhost:8000/api/alerts/active
```

---

## 📖 API Documentation

**Interactive Testing**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Schema: http://localhost:8000/openapi.json

**Key Endpoints**:
```
GET  /api/cves              # List CVEs
GET  /api/cves/stats        # CVE statistics
GET  /api/cves/critical     # Critical CVEs

GET  /api/iocs              # List IOCs
GET  /api/iocs/active       # Active IOCs

GET  /api/threats           # Threat intelligence
GET  /api/threats/high-severity  # High severity threats

GET  /api/topics            # Discovered topics
GET  /api/topics/significant  # Significant topics

GET  /api/alerts            # Security alerts
POST /api/alerts/{id}/acknowledge  # Acknowledge alert
POST /api/alerts/{id}/resolve      # Resolve alert
```

---

## 🔐 Security

- ✅ Input validation (Pydantic)
- ✅ SQL injection prevention (ORM)
- ✅ CORS configured
- ✅ Secret management (environment variables)
- ✅ Security headers (Nginx)
- ✅ Dependency scanning ready

---

## 📊 Statistics

- **9,600+** lines of code
- **40** Python files
- **35** REST API endpoints
- **5** domain entities
- **4** ML models
- **5** data repositories
- **100%** type coverage

---

## 🎯 Use Cases

### Security Operations Center (SOC)
- Monitor incoming CVEs
- Track IOCs across sources
- Manage security alerts
- Discover emerging threat patterns

### Threat Intelligence Team
- Aggregate threat feeds
- Extract IOCs automatically
- Classify threats by severity
- Generate actionable reports

### Security Researchers
- Analyze threat trends
- Discover topic clusters
- Correlate CVEs with threats
- Track vulnerability evolution

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `README_THREAT_INTEL.md` | **This file** - Getting started |
| `PROJECT_COMPLETE.md` | Complete project summary |
| `PROGRESS_SESSION4.md` | API implementation details |
| `AGENTS.md` | Development guidelines |

---

## 🤝 Contributing

This is an educational/demo project. For improvements:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **FastAPI** - Awesome async web framework
- **spaCy** - Industrial-strength NLP
- **gensim** - Topic modeling made easy
- **Hugging Face** - Transformers library
- **AlienVault** - OTX threat intelligence platform

---

## 📧 Contact

For questions, feedback, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/yourusername/threat-intel-aggregator/issues)
- Email: your.email@example.com

---

**Built with ❤️ for the cybersecurity community**

🛡️ Stay secure! 🛡️
