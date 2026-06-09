# Kronos Quant Pipeline - Production Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-slim.txt requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-slim.txt

COPY . .

RUN mkdir -p data/cache cache logs

HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import pipeline.core; print('OK')" || exit 1

CMD ["python3", "scripts/run_production.py"]
