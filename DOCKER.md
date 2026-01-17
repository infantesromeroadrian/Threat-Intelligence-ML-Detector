# 🐳 Docker Setup - Threat Intelligence Aggregator

Complete guide for running the application with Docker.

---

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum (8GB recommended for ML models)
- 10GB disk space

---

## 🚀 Quick Start (Production)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env with your API keys
nano .env  # or your preferred editor

# 3. Build and start services
make build
make up

# 4. Verify services are running
make ps
make health
```

**Access:**
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

## 🔧 Development Mode (Hot-Reload)

```bash
# 1. Build development images
make dev-build

# 2. Start with hot-reload
make dev-up

# 3. View logs
make dev-logs

# 4. Stop services
make dev-down
```

**Changes to `src/` will automatically reload!**

---

## 📦 Services

| Service | Port | Description |
|---------|------|-------------|
| **api** | 8000 | FastAPI REST API |
| **dashboard** | 8501 | Streamlit Dashboard |
| **db** | - | SQLite (persistent volume) |
| **cli** | - | CLI commands (profile) |

---

## 🔨 Common Commands

### Using Makefile (Recommended)

```bash
# Start services
make up

# View logs
make logs              # All services
make logs-api          # API only
make logs-dash         # Dashboard only

# Stop services
make down

# Restart
make restart

# Run CLI commands
make cli CMD="scrape-cves --days 7"
make scrape-cves       # Shortcut
make extract-iocs      # Shortcut
make discover-topics   # Shortcut

# Open shell in container
make shell             # API container
make shell-dash        # Dashboard container

# Cleanup
make clean             # Stop and remove containers + volumes
make clean-all         # Also remove images
```

### Using docker-compose directly

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Run CLI command
docker-compose run --rm cli python -m src.threat_intelligence_aggregator.infrastructure.cli.commands scrape-cves --days 7

# Execute in running container
docker-compose exec api bash
```

---

## 🏗️ Docker Architecture

### Multi-Stage Build

```
┌──────────────────────────────────────────────────┐
│ Stage 1: base                                    │
│ - Python 3.10 slim                               │
│ - System dependencies                            │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│ Stage 2: dependencies                            │
│ - Install requirements.txt                       │
│ - Download spaCy model                           │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│ Stage 3: app (PRODUCTION)                        │
│ - Copy source code                               │
│ - Create non-root user                           │
│ - Setup directories                              │
└──────────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│ Stage 4: development (OPTIONAL)                  │
│ - Install dev dependencies                       │
│ - Enable hot-reload                              │
└──────────────────────────────────────────────────┘
```

### Volume Mounts

| Volume | Purpose | Persistent |
|--------|---------|------------|
| `threat-intel-data` | Database, scraped data | ✅ Yes |
| `threat-intel-models` | Trained ML models | ✅ Yes |
| `threat-intel-cache` | API caches | ✅ Yes |
| `./src` (dev only) | Source code hot-reload | ❌ No |

---

## 🔐 Security Features

### ✅ Implemented

- **Non-root user**: Container runs as `appuser` (UID 1000)
- **Read-only volumes**: Source code mounted read-only in dev mode
- **No secrets in image**: `.env` file never copied to image
- **Minimal base image**: `python:3.10-slim` (smaller attack surface)
- **Health checks**: Automatic container health monitoring
- **.dockerignore**: Excludes unnecessary files from build context

### 🔒 Environment Variables Security

**❌ NEVER commit `.env` to Git!**

```bash
# Good: Use .env.example as template
cp .env.example .env
# Edit .env with real values

# Good: Pass via docker-compose
docker-compose up

# Good: Pass via CLI
docker run -e NVD_API_KEY=xxx threat-intel-aggregator

# BAD: Hardcode in Dockerfile
ENV NVD_API_KEY=xxx  # DON'T DO THIS!
```

---

## 📊 Monitoring & Logs

### View Logs

```bash
# All services (real-time)
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api

# Since timestamp
docker-compose logs --since 2026-01-17T10:00:00 api
```

### Health Check

```bash
# Check API health
curl http://localhost:8000/health

# Check container health status
docker ps

# Detailed health info
docker inspect threat-intel-api | grep Health -A 10
```

### Resource Usage

```bash
# Monitor resource usage
docker stats

# View resource limits
docker inspect threat-intel-api | grep -i memory
```

---

## 🐛 Troubleshooting

### Service Won't Start

```bash
# Check logs
make logs-api

# Check container status
docker ps -a

# Rebuild from scratch
make clean
make build
make up
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process (if safe)
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8080:8000"  # Use 8080 on host
```

### Database Issues

```bash
# Reset database (⚠️ data loss!)
docker volume rm threat-intel-data
make up
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > 8GB

# Or limit container memory in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Permission Errors

```bash
# Ensure volumes have correct permissions
docker-compose exec api ls -la /app/data

# Fix permissions (if needed)
docker-compose exec --user root api chown -R appuser:appuser /app/data
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build & Push

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t threat-intel-aggregator:latest .
      
      - name: Run tests in container
        run: |
          docker run --rm threat-intel-aggregator:latest \
            pytest src/threat_intelligence_aggregator/tests
      
      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push threat-intel-aggregator:latest
```

---

## 🚀 Production Deployment

### AWS ECS Example

```bash
# 1. Build and tag for ECR
docker build -t threat-intel-aggregator:latest .
docker tag threat-intel-aggregator:latest <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/threat-intel-aggregator:latest

# 2. Push to ECR
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
docker push <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/threat-intel-aggregator:latest

# 3. Deploy to ECS (use task definition)
```

### Docker Swarm Example

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml threat-intel

# Scale services
docker service scale threat-intel_api=3

# View services
docker stack services threat-intel
```

### Kubernetes (Helm) - Coming Soon

---

## 📈 Performance Optimization

### Build Cache

```bash
# Use BuildKit for better caching
export DOCKER_BUILDKIT=1
docker build --cache-from threat-intel-aggregator:latest .
```

### Multi-Platform Build

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t threat-intel-aggregator:latest .
```

### Layer Optimization

- ✅ Copy `requirements.txt` before source code (better caching)
- ✅ Use `.dockerignore` to exclude unnecessary files
- ✅ Combine `RUN` commands to reduce layers
- ✅ Clean up apt cache in same layer

---

## 📋 Maintenance

### Update Dependencies

```bash
# 1. Update requirements locally
uv pip compile pyproject.toml -o requirements.txt --upgrade

# 2. Rebuild images
make clean
make build
make up
```

### Backup Data

```bash
# Backup volumes
docker run --rm -v threat-intel-data:/data -v $(pwd):/backup alpine tar czf /backup/threat-intel-data-backup.tar.gz /data

# Restore volumes
docker run --rm -v threat-intel-data:/data -v $(pwd):/backup alpine tar xzf /backup/threat-intel-data-backup.tar.gz -C /
```

### Database Migration

```bash
# Export SQLite database
docker-compose exec api sqlite3 /app/data/threat_intel.db .dump > backup.sql

# Import to new database
cat backup.sql | docker-compose exec -T api sqlite3 /app/data/threat_intel_new.db
```

---

## 🆘 Getting Help

- **Documentation**: See `README.md` and `DEPENDENCIES.md`
- **Issues**: Open an issue on GitHub
- **Logs**: Always check logs first: `make logs`

---

**Last Updated**: 2026-01-17  
**Docker Version**: 20.10+  
**Compose Version**: 2.0+
