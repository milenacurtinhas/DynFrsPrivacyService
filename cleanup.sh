#!/bin/bash
# Script de limpeza completa para reconstruir o serviço

echo "=== LIMPEZA COMPLETA DO DYNFRS ==="
echo ""
echo "Este script vai limpar containers órfãos e liberar a porta 8000."
echo "Você pode precisar inserir sua senha de administrador."
echo ""

# 1. Parar TODOS os containers (incluindo os criados com sudo)
echo "[1/5] Parando containers..."
sudo docker stop $(sudo docker ps -aq --filter "name=dynfrs") 2>/dev/null || true
docker stop $(docker ps -aq --filter "name=dynfrs") 2>/dev/null || true
echo "✓ Containers parados"
echo ""

# 2. Remover TODOS os containers
echo "[2/5] Removendo containers..."
sudo docker rm $(sudo docker ps -aq --filter "name=dynfrs") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=dynfrs") 2>/dev/null || true
echo "✓ Containers removidos"
echo ""

# 3. Remover TODAS as imagens dynfrs
echo "[3/5] Removendo imagens antigas..."
sudo docker rmi $(sudo docker images -q "*dynfrs*") -f 2>/dev/null || true
docker rmi $(docker images -q "*dynfrs*") -f 2>/dev/null || true
echo "✓ Imagens removidas"
echo ""

# 4. Verificar se porta 8000 está livre
echo "[4/5] Verificando porta 8000..."
PORT_PID=$(sudo lsof -ti:8000 2>/dev/null || echo "")
if [ ! -z "$PORT_PID" ]; then
    echo "⚠ Porta 8000 em uso pelo PID: $PORT_PID"
    echo "Liberando porta..."
    sudo kill -9 $PORT_PID 2>/dev/null || true
    sleep 2
fi
echo "✓ Porta 8000 livre"
echo ""

# 5. Limpar cache Python
echo "[5/5] Limpando cache Python..."
find api -name "*.pyc" -delete 2>/dev/null || true
find api -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✓ Cache limpo"
echo ""

echo "=== LIMPEZA CONCLUÍDA ==="
echo ""
echo "Execute agora: ./deploy.sh"
