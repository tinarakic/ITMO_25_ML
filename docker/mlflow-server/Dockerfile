FROM python:3.12-slim

# Install system dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow with S3 & SQL support
RUN pip install --no-cache-dir \
    mlflow[extras]==3.1.0 \
    boto3 \
    psycopg2-binary
