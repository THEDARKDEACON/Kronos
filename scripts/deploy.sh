#!/bin/bash
# Kronos Production Deployment Script

set -e

echo "🚀 Kronos Quant Pipeline - Deployment Script"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating from template...${NC}"
    cp .env.example .env
    echo -e "${RED}Please edit .env file with your API keys before continuing.${NC}"
    exit 1
fi

# Parse arguments
MODE=${1:-paper}
ACTION=${2:-deploy}

echo -e "${GREEN}Mode: $MODE${NC}"
echo -e "${GREEN}Action: $ACTION${NC}"

# Functions
deploy() {
    echo -e "${YELLOW}Building and deploying...${NC}"
    
    # Create necessary directories
    mkdir -p data/cache data/processed experiments research logs models
    
    # Build images
    docker-compose build
    
    # Start services
    docker-compose up -d
    
    echo -e "${GREEN}Deployment complete!${NC}"
    echo ""
    echo "Services available at:"
    echo "  Dashboard:    http://localhost:8501"
    echo "  Prometheus:   http://localhost:9090"
    echo "  Grafana:      http://localhost:3000 (admin/admin)"
    echo "  MLflow:       http://localhost:5000"
    echo ""
}

stop() {
    echo -e "${YELLOW}Stopping services...${NC}"
    docker-compose down
    echo -e "${GREEN}Services stopped.${NC}"
}

restart() {
    echo -e "${YELLOW}Restarting services...${NC}"
    docker-compose restart
    echo -e "${GREEN}Services restarted.${NC}"
}

logs() {
    echo -e "${YELLOW}Showing logs...${NC}"
    docker-compose logs -f
}

health() {
    echo -e "${YELLOW}Running health check...${NC}"
    docker-compose exec kronos-pipeline python scripts/health_check.py
}

update() {
    echo -e "${YELLOW}Updating to latest version...${NC}"
    git pull
    docker-compose build
    docker-compose up -d
    echo -e "${GREEN}Update complete!${NC}"
}

backup() {
    echo -e "${YELLOW}Creating backup...${NC}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    tar -czf "backup_$TIMESTAMP.tar.gz" data/ experiments/ logs/
    echo -e "${GREEN}Backup created: backup_$TIMESTAMP.tar.gz${NC}"
}

# Execute action
case $ACTION in
    deploy|start)
        deploy
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    health)
        health
        ;;
    update)
        update
        ;;
    backup)
        backup
        ;;
    *)
        echo "Usage: $0 [mode] [action]"
        echo ""
        echo "Modes:"
        echo "  backtest  - Run with historical data"
        echo "  paper     - Paper trading (default)"
        echo "  live      - Live trading (⚠️  Real money)"
        echo ""
        echo "Actions:"
        echo "  deploy    - Deploy all services"
        echo "  stop      - Stop all services"
        echo "  restart   - Restart services"
        echo "  logs      - View logs"
        echo "  health    - Run health check"
        echo "  update    - Update to latest version"
        echo "  backup    - Backup data"
        echo ""
        echo "Examples:"
        echo "  $0 paper deploy   # Deploy paper trading"
        echo "  $0 live deploy    # ⚠️  Deploy live trading"
        echo "  $0 paper stop     # Stop services"
        echo "  $0 paper logs     # View logs"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
