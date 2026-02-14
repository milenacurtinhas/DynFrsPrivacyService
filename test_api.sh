#!/bin/bash
# Script de teste automatizado para DynFrs Service

BASE_URL="${BASE_URL:-http://localhost:8000}"
TESTS_PASSED=0
TESTS_FAILED=0

echo ""
echo "DynFrs Service - Teste Automatizado"
echo ""

test_endpoint() {
    local test_name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    local expected_field=$5
    
    echo "[TEST] ${test_name}"
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s "${BASE_URL}${endpoint}")
    else
        response=$(curl -s -X POST "${BASE_URL}${endpoint}" \
                   -H "Content-Type: application/json" \
                   -d "${data}")
    fi
    
    if echo "$response" | grep -q "\"${expected_field}\""; then
        echo "  [PASS] PASSOU"
        echo "  Response: $(echo $response | head -c 100)..."
        ((TESTS_PASSED++))
    else
        echo "  [FAIL] FALHOU"
        echo "  Response: $response"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Teste 1: Health Check
echo "━━━ Fase 1: Health Check ━━━"
test_endpoint "Health Check" "GET" "/health" "" "status"

# Teste 2: Service Info
echo "━━━ Fase 2: Service Info ━━━"
test_endpoint "Service Info" "GET" "/" "" "service"

# Teste 3: Model Training
echo "━━━ Fase 3: Model Training ━━━"
training_data='{"dataset": "Adult", "num_trees": 10, "max_depth": 10, "k": 5}'
test_endpoint "Train Model" "POST" "/model/train" "$training_data" "model_id"

echo "Aguardando treinamento..."
sleep 2

# Teste 4: Machine Unlearning
echo "━━━ Fase 4: Machine Unlearning ━━━"
unlearn_data='{"unlearn_count": 100}'
test_endpoint "Machine Unlearning" "POST" "/model/unlearn" "$unlearn_data" "unlearned_samples"

# Resumo
echo ""
echo "━━━ Resumo ━━━"
echo "  [PASS] Tests Passed: ${TESTS_PASSED}"
echo "  [FAIL] Tests Failed: ${TESTS_FAILED}"
echo ""

if [ $TESTS_FAILED -gt 0 ]; then
    echo "[FAIL] Some tests failed."
    exit 1
else
    echo "[SUCCESS] All tests passed!"
    echo ""
    echo "Para ver métricas detalhadas:"
    echo "  docker-compose logs -f dynfrs-privacy-service"
    exit 0
fi
