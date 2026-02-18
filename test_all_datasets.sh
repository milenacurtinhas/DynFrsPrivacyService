#!/bin/bash

# Script para testar ataque de inferência em todos os datasets
echo ""
echo "TESTE DE ATAQUE - TODOS OS DATASETS"
echo ""

# Aguardar serviço estar pronto
sleep 5

# Datasets para testar
DATASETS=("Adult" "Vaccine" "NoShow")

# Arquivo para salvar resultados
RESULTS_FILE="results_$(date +%Y%m%d_%H%M%S).txt"

echo "Resultados salvos em: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Dataset: $dataset"
    echo ""
    
    echo "" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    echo "DATASET: $dataset" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    
    # Executar ataque
    response=$(curl -s -X POST http://localhost:8000/attack/inference \
      -H "Content-Type: application/json" \
      -d "{
        \"dataset\": \"$dataset\",
        \"sample_size\": 5000,
        \"unlearn_count\": 500
      }")
    
    # Extrair métricas com jq (se disponível) ou python
    if command -v jq &> /dev/null; then
        model_pre=$(echo "$response" | jq -r '.model_metrics.accuracy_pre_unlearning')
        model_pos=$(echo "$response" | jq -r '.model_metrics.accuracy_post_unlearning')
        attack_pre=$(echo "$response" | jq -r '.attack_metrics.accuracy_pre_unlearning')
        attack_pos=$(echo "$response" | jq -r '.attack_metrics.accuracy_post_unlearning')
    else
        model_pre=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['model_metrics']['accuracy_pre_unlearning'])")
        model_pos=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['model_metrics']['accuracy_post_unlearning'])")
        attack_pre=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['attack_metrics']['accuracy_pre_unlearning'])")
        attack_pos=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['attack_metrics']['accuracy_post_unlearning'])")
    fi
    
    # Exibir e salvar resultados
    echo "Acurácia Modelo (PRÉ):   $model_pre" | tee -a "$RESULTS_FILE"
    echo "Acurácia Modelo (PÓS):   $model_pos" | tee -a "$RESULTS_FILE"
    echo "Acurácia Ataque (PRÉ):   $attack_pre" | tee -a "$RESULTS_FILE"
    echo "Acurácia Ataque (PÓS):   $attack_pos" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    
    # Salvar JSON completo
    echo "$response" | python3 -m json.tool >> "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    
    echo ""
done

echo ""
echo "TESTE CONCLUÍDO!"
echo "Resultados salvos em: $RESULTS_FILE"
echo ""
