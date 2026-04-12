# Kronos Quant Pipeline - Deployment Guide

## Quick Start

```bash
# 1. Clone and setup
cd /home/gareth-joel/Downloads/Kronos
cp .env.example .env
# Edit .env with your API keys

# 2. Local deployment (Docker)
docker-compose up -d

# 3. Verify deployment
curl http://localhost:8501  # Dashboard
curl http://localhost:9090  # Prometheus
curl http://localhost:3000  # Grafana (admin/admin)
```

## Deployment Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Dashboard     │     │  Pipeline Runner │     │    Scheduler    │
│  (Streamlit)    │◄────│   (Python)       │────►│   (Cron)        │
│    :8501        │     │   Main Logic     │     │   Rebalancing   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
           │                       │
           ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│     Redis       │     │    MLflow        │
│   Cache/Queue   │     │  Experiment      │
│     :6379       │     │   Tracking       │
└─────────────────┘     └──────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│              Monitoring Stack                   │
│  Prometheus (:9090)  ◄───  Grafana (:3000)     │
└─────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `KRONOS_MODE` | Yes | `backtest`, `paper`, or `live` |
| `POLYGON_API_KEY` | No | Alternative to yFinance |
| `NEWSAPI_KEY` | No | Real news sentiment |
| `ALPACA_API_KEY` | For paper/live | Broker API key |
| `ALPACA_SECRET_KEY` | For paper/live | Broker secret |
| `S3_BUCKET` | No | Cloud data storage |
| `SLACK_WEBHOOK_URL` | No | Error alerts |

### Modes

| Mode | Data | Broker | Risk |
|------|------|--------|------|
| `backtest` | Historical | Simulated | None |
| `paper` | Real-time | Paper trading | Fake money |
| `live` | Real-time | Real broker | **Real money** |

## Cloud Deployment

### AWS (Recommended)

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name kronos-quant

# 2. Build and push
docker build -t kronos-quant .
docker tag kronos-quant:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/kronos-quant:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/kronos-quant:latest

# 3. Deploy to ECS or EC2
# Use provided terraform/ or cloudformation/ templates
```

### GCP

```bash
# 1. Build and push to GCR
docker build -t gcr.io/$PROJECT_ID/kronos-quant .
docker push gcr.io/$PROJECT_ID/kronos-quant

# 2. Deploy to Cloud Run
gcloud run deploy kronos-pipeline \
  --image gcr.io/$PROJECT_ID/kronos-quant \
  --set-env-vars KRONOS_MODE=paper
```

### Azure

```bash
# 1. Build and push to ACR
az acr build --registry $ACR_NAME --image kronos-quant .

# 2. Deploy to Container Instances
az container create \
  --resource-group $RG \
  --name kronos-pipeline \
  --image $ACR_NAME.azurecr.io/kronos-quant \
  --cpu 2 --memory 4
```

## Monitoring

### Health Checks

```bash
# System health
python scripts/health_check.py

# Component status
python -c "from scripts.health_check import HealthChecker; c = HealthChecker(); print(c.run_all_checks())"
```

### Prometheus Metrics

| Metric | Description |
|--------|-------------|
| `kronos_health_status` | 1 if healthy, 0 if not |
| `kronos_cpu_usage` | CPU percent |
| `kronos_memory_usage` | Memory percent |
| `kronos_data_sources_up` | Number of working data sources |

### Alerts

Configure in `alertmanager.yml`:

- Pipeline fails 3 times in a row
- Data source down > 15 minutes
- Memory usage > 90%
- No trades executed in 24h (live mode)

## Troubleshooting

### Common Issues

**FinBERT sentiment is 0**
- Normal with mock news (neutral headlines)
- Add NEWSAPI_KEY for real sentiment

**yFinance timeout**
- Add POLYGON_API_KEY for reliable data
- Check internet connection

**Model not loading**
- Ensure GPU available: `nvidia-smi`
- Check disk space: `df -h`

**Docker fails to start**
- Check logs: `docker-compose logs kronos-pipeline`
- Verify .env file exists and is readable

## Security Checklist

- [ ] API keys in .env (not committed)
- [ ] .env in .gitignore
- [ ] Encryption key set for sensitive data
- [ ] Paper trading tested before live
- [ ] IP whitelist for dashboard
- [ ] HTTPS for production dashboard
- [ ] Database backups configured

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  kronos-pipeline-1:
    ...
  kronos-pipeline-2:
    ...
  
  # Use Redis to coordinate
```

### Database Scaling

- S3 for historical data storage
- RDS/Cloud SQL for structured data
- ElastiCache for Redis

## Maintenance

### Daily
- Check dashboard for errors
- Verify positions match target
- Review execution logs

### Weekly
- Review performance metrics
- Check alpha decay
- Update universe if needed

### Monthly
- Model retraining
- Risk parameter review
- Cost analysis
- Strategy evaluation

## Support

For issues:
1. Check logs: `docker-compose logs`
2. Run health check: `python scripts/health_check.py`
3. Review: `docs/TROUBLESHOOTING.md`
