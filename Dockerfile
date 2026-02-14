# Dockerfile simplificado para testes - Modo Python
FROM python:3.10-slim

# Instalar dependências de runtime
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Criar diretórios
WORKDIR /app
RUN mkdir -p /app/models /app/Datasets

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY api/ /app/

# Copiar datasets  
COPY Datasets/ /app/Datasets/

# Exposar porta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando padrão
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
