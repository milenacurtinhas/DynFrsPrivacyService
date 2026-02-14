#!/bin/bash
# Deploy script para DynFrs Privacy Service

# NÃO usar set -e pois queremos continuar mesmo com erros de limpeza

echo "  DynFrs Privacy Service - Deploy Script"
echo "  Machine Unlearning"
echo ""

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "[FAIL] Docker não encontrado. Instale Docker primeiro."
    exit 1
fi

# Verificar Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "[FAIL] Docker Compose não encontrado. Instale Docker Compose primeiro."
    exit 1
fi

echo "[OK] Docker: $(docker --version)"
echo "[OK] Docker Compose: $(docker-compose --version)"
echo ""

# Criar diretórios necessários
echo "[DIR] Criando diretórios..."
mkdir -p models
echo "[OK] Diretórios criados"
echo ""

# Parar containers existentes (ignorar erros)
echo "[DOCKER] Limpando ambiente..."
docker-compose down 2>/dev/null || true
echo ""

# Build containers
echo "[DOCKER] Construindo imagem..."
docker-compose build 2>&1 | grep -v "DeprecationWarning\|CryptographyDeprecationWarning" || docker-compose build
echo ""

# Iniciar containers
echo "[DOCKER] Iniciando serviço..."
docker-compose up -d 2>&1 | grep -v "orphan container" | grep -v "DeprecationWarning\|CryptographyDeprecationWarning" || docker-compose up -d

# Aguardar containers iniciarem
echo ""
echo "[WAIT] Aguardando serviço iniciar..."
sleep 10

# Health check
echo ""
echo "[CHECK] Verificando saúde do serviço..."
MAX_RETRIES=10
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "[OK] Serviço está saudável!"
        break
    else
        RETRY=$((RETRY+1))
        echo "  Tentativa $RETRY/$MAX_RETRIES..."
        sleep 3
    fi
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo "[FAIL] Serviço não respondeu após $MAX_RETRIES tentativas"
    echo ""
    echo "Logs do container:"
    docker-compose logs dynfrs-api
    exit 1
fi

echo ""
echo "  [PASS] Deploy Concluído com Sucesso!"
echo ""
echo "[WEB] Serviço disponível em:"
echo "   API:          http://localhost:8000"
echo "   Docs:         http://localhost:8000/docs"
echo "   Health:       http://localhost:8000/health"
echo ""
echo "[DOCS] Comandos úteis:"
echo "   Ver logs:     docker-compose logs -f dynfrs-api"
echo "   Parar:        docker-compose down"
echo "   Restart:      docker-compose restart"
echo "   Status:       docker-compose ps"
echo ""
echo "[TEST] Próximos passos:"
echo "   1. Testar serviço:  python3 test_service.py"
echo "   2. Ver docs:        http://localhost:8000/docs"
echo "   3. Ler API docs:    cat API_DOCUMENTATION.md"
echo ""
