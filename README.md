# DynFrs Service

## Descrição

Implementação do framework DynFrs (Dynamic Random Forest Serialized) como serviço REST, permitindo:
- Treinamento de modelos Random Forest
- **Machine Unlearning**: Remoção eficiente de amostras

## Quick Start

### 1. Deploy
```bash
chmod +x deploy.sh
./deploy.sh
```

### 2. Teste
```bash
./test_api.sh
```

### 3. Acesso
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

---

## Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Treinar Modelo
```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"dataset": "Adult", "num_trees": 100}'
```

### Machine Unlearning
```bash
curl -X POST http://localhost:8000/model/unlearn \
  -H "Content-Type: application/json" \
  -d '{"unlearn_count": 100}'
```

---

## Estrutura do Projeto

```
DynFrsPrivacyService/
├── api/                  # API REST (FastAPI)
│   ├── main.py           # Endpoints
│   └── model_manager.py  # Gerenciador de modelos
│
├── Datasets/             # Datasets
│   ├── Adult/            
│   ├── Vaccine/
│   └── NoShow/
│
├── DynFrs.h              # Código C++ original
├── main_serializada.cpp  # Implementação DynFrs
└── docker-compose.yml    # Orquestração
```

---

## Comandos Docker

```bash
# Iniciar
docker-compose up -d

# Logs (ver métricas)
docker-compose logs -f dynfrs-privacy-service

# Parar
docker-compose down

# Rebuild
docker-compose up --build --force-recreate -d
```

---

## Exemplo de Uso (Python)

```python
import requests

BASE = "http://localhost:8000"

# 1. Treinar
r = requests.post(f"{BASE}/model/train", 
    json={"dataset": "Adult", "num_trees": 100})
print(f"Model ID: {r.json()['model_id']}")

# 2. Unlearning
r = requests.post(f"{BASE}/model/unlearn",
    json={"unlearn_count": 100})
print(f"Status: {r.json()['status']}")

```

---

Desenvolvido por Milena Curtinhas Santos e Marcela Carpenter da Paixao - Trabalho T2 - Segurança Computacional
