#!/bin/bash
# =============================================================================
# Docker Setup Validation Script
# Checks that Docker setup is correct before building
# =============================================================================

set -e  # Exit on error

echo "🔍 Threat Intelligence Aggregator - Docker Setup Check"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track errors
ERRORS=0

# Function to print status
check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# =============================================================================
# 1. Check Docker is installed
# =============================================================================
echo "1️⃣  Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    check_pass "Docker installed: $DOCKER_VERSION"
else
    check_fail "Docker is not installed"
fi
echo ""

# =============================================================================
# 2. Check Docker Compose is installed
# =============================================================================
echo "2️⃣  Checking Docker Compose..."
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short)
    check_pass "Docker Compose installed: $COMPOSE_VERSION"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}' | sed 's/,//')
    check_warn "Using legacy docker-compose: $COMPOSE_VERSION (consider upgrading)"
else
    check_fail "Docker Compose is not installed"
fi
echo ""

# =============================================================================
# 3. Check Docker daemon is running
# =============================================================================
echo "3️⃣  Checking Docker daemon..."
if docker info &> /dev/null; then
    check_pass "Docker daemon is running"
else
    check_fail "Docker daemon is not running"
fi
echo ""

# =============================================================================
# 4. Check required files exist
# =============================================================================
echo "4️⃣  Checking required files..."
REQUIRED_FILES=(
    "Dockerfile"
    "docker-compose.yml"
    ".dockerignore"
    "requirements.txt"
    "requirements-dev.txt"
    "pyproject.toml"
    ".env.example"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$file exists"
    else
        check_fail "$file is missing"
    fi
done
echo ""

# =============================================================================
# 5. Check .env file
# =============================================================================
echo "5️⃣  Checking environment configuration..."
if [ -f ".env" ]; then
    check_pass ".env file exists"
    
    # Check for placeholder values
    if grep -q "your-api-key-here" .env 2>/dev/null; then
        check_warn ".env contains placeholder values - update with real API keys"
    fi
    
    if grep -q "change-me-in-production" .env 2>/dev/null; then
        check_warn ".env contains insecure SECRET_KEY - generate a secure one"
    fi
else
    check_warn ".env file not found (will use .env.example defaults)"
    echo "    Run: cp .env.example .env"
fi
echo ""

# =============================================================================
# 6. Check source code structure
# =============================================================================
echo "6️⃣  Checking source code structure..."
if [ -d "src/threat_intelligence_aggregator" ]; then
    check_pass "Source directory exists"
    
    # Check key directories
    DIRS=(
        "src/threat_intelligence_aggregator/domain"
        "src/threat_intelligence_aggregator/infrastructure"
        "src/threat_intelligence_aggregator/application"
    )
    
    for dir in "${DIRS[@]}"; do
        if [ -d "$dir" ]; then
            check_pass "$dir exists"
        else
            check_fail "$dir is missing"
        fi
    done
else
    check_fail "Source directory src/threat_intelligence_aggregator not found"
fi
echo ""

# =============================================================================
# 7. Check Docker resources
# =============================================================================
echo "7️⃣  Checking Docker resources..."
if docker info &> /dev/null; then
    TOTAL_MEM=$(docker info --format '{{.MemTotal}}' 2>/dev/null | numfmt --to=iec-i --suffix=B 2>/dev/null || echo "Unknown")
    
    if [ "$TOTAL_MEM" != "Unknown" ]; then
        check_pass "Docker memory: $TOTAL_MEM"
        
        # Warn if less than 4GB
        MEM_GB=$(docker info --format '{{.MemTotal}}' 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
        if [ -n "$MEM_GB" ] && [ "$MEM_GB" -lt 4 ]; then
            check_warn "Docker has less than 4GB RAM - ML models may fail"
        fi
    fi
fi
echo ""

# =============================================================================
# 8. Validate Dockerfile syntax
# =============================================================================
echo "8️⃣  Validating Dockerfile..."
if docker build --dry-run -f Dockerfile . &> /dev/null 2>&1 || docker build -f Dockerfile --target base . &> /dev/null 2>&1; then
    check_pass "Dockerfile syntax is valid"
else
    check_warn "Could not validate Dockerfile (may require docker buildx)"
fi
echo ""

# =============================================================================
# 9. Validate docker-compose syntax
# =============================================================================
echo "9️⃣  Validating docker-compose.yml..."
if docker compose config &> /dev/null; then
    check_pass "docker-compose.yml syntax is valid"
elif docker-compose config &> /dev/null; then
    check_pass "docker-compose.yml syntax is valid (legacy compose)"
else
    check_fail "docker-compose.yml has syntax errors"
fi
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=================================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Ready to build.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Update .env with your API keys (if needed)"
    echo "  2. Build images: make build"
    echo "  3. Start services: make up"
    echo "  4. Check health: make health"
    exit 0
else
    echo -e "${RED}❌ $ERRORS check(s) failed${NC}"
    echo ""
    echo "Please fix the errors above before building."
    exit 1
fi
