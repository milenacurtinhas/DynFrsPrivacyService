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

# Ou testar ataque de inferência
./test_attack.sh

# Ou testar todos os datasets (Adult, Vaccine, NoShow)
./test_all_datasets.sh
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

### Ataque de Inferência de Membro
```bash
curl -X POST http://localhost:8000/attack/inference \
  -H "Content-Type: application/json" \
  -d '{"dataset": "Adult", "sample_size": 5000, "unlearn_count": 500}'
```

---

## Ataque de Inferência de Membro (MIA)

O serviço inclui um endpoint para testar a vulnerabilidade do modelo contra ataques de inferência de membro. Este ataque tenta determinar se uma amostra específica foi usada no treinamento do modelo.

### Como funciona o ataque

1. **Preparação**: Carrega dados de membros (treino) e não-membros (teste)
2. **Treino**: Inicializa e treina o modelo alvo
3. **Coleta**: Obtém probabilidades de predição para ambos os grupos
4. **Ataque Pré-Unlearning**: Treina um classificador para distinguir membros de não-membros
5. **Unlearning**: Remove amostras do modelo
6. **Ataque Pós-Unlearning**: Reavalia a acurácia do ataque

### Exemplo de uso via API

```python
import requests

response = requests.post(
    "http://localhost:8000/attack/inference",
    json={
        "dataset": "Adult",
        "sample_size": 5000,
        "unlearn_count": 500
    }
)

result = response.json()
print(f"Acurácia modelo PRÉ: {result['model_metrics']['accuracy_pre_unlearning']:.2%}")
print(f"Acurácia modelo PÓS: {result['model_metrics']['accuracy_post_unlearning']:.2%}")
print(f"Acurácia ataque PRÉ: {result['attack_metrics']['accuracy_pre_unlearning']:.2%}")
print(f"Acurácia ataque PÓS: {result['attack_metrics']['accuracy_post_unlearning']:.2%}")
print(f"Melhoria de privacidade: {result['attack_metrics']['privacy_improvement_pct']:.2f}%")
```

### Executar script standalone

```bash
python script_ataque_inferencia.py
```

### Testar todos os datasets

Execute o ataque de inferência para todos os datasets (Adult, Vaccine, NoShow) automaticamente:

```bash
./test_all_datasets.sh
```

Este script irá:
- Executar o ataque de inferência para cada dataset
- Coletar as métricas de acurácia do modelo e do ataque (PRÉ e PÓS unlearning)
- Salvar os resultados em um arquivo `results_YYYYMMDD_HHMMSS.txt`
- Exibir um resumo formatado no terminal

**Exemplo de saída:**

```

DATASET: Adult

Acurácia Modelo (PRÉ):   86.35%
Acurácia Modelo (PÓS):   86.49%
Acurácia Ataque (PRÉ):   50.69%
Acurácia Ataque (PÓS):   50.81%
```

**Interpretação dos resultados:**
- **Acurácia do Modelo**: Deve se manter alta (~80%) após unlearning
- **Acurácia do Ataque**: Deve estar próxima de 50% (equivalente a chance aleatória)
- Se ataque ~50% = **Privacidade preservada** ✅

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

# 3. Ataque de Inferência
r = requests.post(f"{BASE}/attack/inference",
    json={"dataset": "Adult", "sample_size": 5000, "unlearn_count": 500})
print(f"Acurácia modelo PRÉ: {r.json()['model_metrics']['accuracy_pre_unlearning']:.2%}")
print(f"Acurácia ataque PRÉ: {r.json()['attack_metrics']['accuracy_pre_unlearning']:.2%}")
print(f"Acurácia ataque PÓS: {r.json()['attack_metrics']['accuracy_post_unlearning']:.2%}")

```

---

Desenvolvido por Milena Curtinhas Santos e Marcela Carpenter da Paixao - Trabalho T2 - Segurança Computacional
