# 📧 SPAM & PHISHING Detector - Full Stack ML Application

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18-61dafb.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Production-ready full-stack email threat detection system** using Machine Learning with modern React frontend and FastAPI backend.

## 🏗️ Project Structure

```
spam-phishing-detector/
├── src/
│   ├── backend/          # FastAPI + ML models (Python 3.12)
│   │   ├── spam_detector/  # Python package (flat layout)
│   │   ├── tests/
│   │   ├── models/       # Git LFS tracked
│   │   └── README.md     # Backend docs
│   └── frontend/         # React + TypeScript + Vite
│       ├── src/
│       ├── public/
│       └── README.md     # Frontend docs
├── docs/                 # Documentation
│   └── README_FULL_STACK.md
├── docker-compose.yml    # Full-stack deployment
├── LICENSE
└── README.md             # This file
```

## ✨ Features

### 🎯 ML Capabilities
- **Dual Detection**: Simultaneous SPAM and PHISHING classification
- **High Accuracy**: ~95% SPAM, ~92% PHISHING detection
- **Fast Inference**: <10ms per email
- **Model Versioning**: MLflow + Git LFS

### 🚀 Interfaces
- **Modern Web UI**: React with dark glassmorphism theme
- **REST API**: FastAPI with automatic OpenAPI docs
- **CLI Tool**: Rich terminal interface

### 🏛️ Architecture
- **Backend**: Hexagonal/Clean Architecture
- **Frontend**: Component-based React with TypeScript
- **Type Safety**: End-to-end with Pydantic + TypeScript
- **Testing**: Comprehensive test suites

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (backend)
- **Node.js 18+** (frontend)
- **uv** (recommended): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker** (optional, for containerized deployment)

### Option 1: Development Setup

#### Backend

```bash
cd src/backend

# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv sync

# Run API server
spam-detector-api
# → http://localhost:8000
# → Docs: http://localhost:8000/docs

# Or use CLI
spam-detector predict "URGENT! You won a lottery!"
```

#### Frontend

```bash
cd src/frontend

# Install dependencies
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Start dev server
npm run dev
# → http://localhost:5173
```

### Option 2: Docker Compose (Full Stack)

```bash
# Build and run both services
docker-compose up --build

# Access:
# - Frontend: http://localhost:5173
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Backend README](src/backend/README.md) | Backend setup, API docs, CLI usage |
| [Frontend README](src/frontend/README.md) | Frontend development, components, deployment |
| [Full-Stack Guide](docs/README_FULL_STACK.md) | Complete setup and architecture guide |

## 🎯 Tech Stack

### Backend
- **Framework**: FastAPI
- **ML**: scikit-learn, NLTK
- **Validation**: Pydantic
- **Testing**: pytest
- **Tooling**: uv, ruff, mypy

### Frontend
- **Framework**: React 18 + TypeScript
- **Build**: Vite
- **Styling**: Tailwind CSS
- **Animation**: Framer Motion
- **Charts**: Chart.js
- **State**: React Query
- **HTTP**: Axios

## 📊 ML Models

| Model | Algorithm | Accuracy | Samples | Features |
|-------|-----------|----------|---------|----------|
| SPAM | Logistic Regression | ~95% | 5,572 | TF-IDF (5000) |
| PHISHING | Logistic Regression | ~92% | 11,430 | TF-IDF (5000) |

Models are versioned with **Git LFS** and tracked with **MLflow**.

## 🔌 API Usage

### Classify Email

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "email_text": "URGENT! You won a lottery! Click here now!"
  }'
```

**Response:**

```json
{
  "verdict": "PHISHING",
  "risk_level": "HIGH",
  "is_malicious": true,
  "spam_probability": 0.505,
  "phishing_probability": 0.985,
  "execution_time_ms": 1.81,
  "threat_report": {
    "risk_score": 84,
    "iocs": [...],
    "recommendations": [...]
  }
}
```

See [API documentation](http://localhost:8000/docs) for complete endpoints.

## 🧪 Testing

### Backend

```bash
cd src/backend
pytest                           # Run all tests
pytest --cov=spam_detector      # With coverage
pytest tests/unit               # Only unit tests
```

### Frontend

```bash
cd src/frontend
npm run lint                    # Lint code
npm run build                   # Build for production
```

## 🚢 Deployment

### Backend Options
- **Docker**: Use `src/backend/Dockerfile`
- **Railway/Render**: Connect GitHub repo
- **AWS ECS/EKS**: Push to ECR, deploy container

### Frontend Options
- **Vercel**: Connect GitHub, auto-deploy
- **Netlify**: Connect repo, set build command
- **Cloudflare Pages**: Similar to Vercel/Netlify
- **Static**: Build and serve via nginx/CDN

### Full-Stack
```bash
docker-compose -f docker-compose.yml up -d
```

See [deployment guide](docs/README_FULL_STACK.md#deployment) for details.

## 🎨 Frontend Preview

The modern React frontend features:
- 🌑 Dark glassmorphism cybersecurity theme
- 📊 Dual gauge charts for threat visualization
- 🎭 Smooth animations with Framer Motion
- 🎯 Color-coded risk levels (green → red)
- 📱 Responsive mobile-friendly design
- ⚡ Fast loading with Vite HMR

## 🔐 Security

- ✅ Input validation (Pydantic)
- ✅ CORS configured
- ✅ No secrets in code
- ✅ Type safety (mypy strict)
- ✅ Dependency scanning ready
- ⚠️ Rate limiting (TODO for production)
- ⚠️ Authentication (TODO for production)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes
4. Run tests: `cd src/backend && pytest`
5. Commit: `git commit -m "feat: add feature"`
6. Push: `git push origin feature/my-feature`
7. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file

## 🏆 Credits

**Built with:**
- Clean Architecture principles
- Hexagonal/Ports & Adapters pattern
- Modern React best practices
- Type-driven development

**Author**: Adrian Infantes Romero

---

**⚡ Built for production ML systems**

For detailed setup and architecture information, see [Full-Stack Guide](docs/README_FULL_STACK.md).
