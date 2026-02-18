#!/bin/bash

# Script de teste para o ataque de inferência de membro
echo "==================================="
echo "Teste de Ataque de Inferência"
echo "==================================="
echo ""

# Verificar se o serviço está rodando
echo "1. Verificando health..."
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""

# Executar ataque de inferência
echo "2. Executando ataque de inferência..."
curl -X POST http://localhost:8000/attack/inference \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "Adult",
    "sample_size": 5000,
    "unlearn_count": 500
  }' | python3 -m json.tool

echo ""
echo "==================================="
echo "Teste concluído!"
echo "==================================="
