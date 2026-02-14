#!/usr/bin/env python3
"""
Script de teste completo do DynFrs Service
Executa workflow completo: treino e unlearning
"""

import requests
import json
import time
from typing import Dict

BASE_URL = "http://localhost:8000"

def print_section(title: str):
    """Imprime cabeçalho de seção"""
    print("\n" )
    print(f"  {title}")
    print("")

def test_health():
    """Testa health check"""
    print_section("1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    print(f"[OK] Status: {data['status']}")
    print(f"[OK] Model loaded: {data['model_loaded']}")
    print(f"[OK] Memory: {data['memory_usage']['rss_mb']:.2f} MB")

def train_model(dataset: str = "Adult") -> Dict:
    """Treina um modelo"""
    print_section("2. Treinamento do Modelo")
    print(f"Dataset: {dataset}")
    
    payload = {
        "dataset": dataset,
        "num_trees": 100,
        "max_depth": 15,
        "min_split_size": 10,
        "k": 10
    }
    
    print("Treinando... (pode levar alguns segundos)")
    start = time.time()
    response = requests.post(f"{BASE_URL}/model/train", json=payload)
    elapsed = time.time() - start
    
    data = response.json()
    print(f"[OK] Model ID: {data['model_id']}")
    print(f"[OK] Accuracy: {data['metrics']['train_accuracy']:.4f}")
    print(f"[OK] Samples: {data['metrics']['train_samples']}")
    print(f"[OK] Features: {data['metrics']['features']}")
    print(f"[OK] Tempo: {elapsed:.2f}s")
    
    return data

def perform_unlearning(unlearn_count: int = 100) -> Dict:
    """Realiza unlearning"""
    print_section("3. Machine Unlearning")
    
    payload = {
        "unlearn_count": unlearn_count
    }
    
    print(f"Removendo {unlearn_count} amostras aleatórias...")
    start = time.time()
    response = requests.post(f"{BASE_URL}/model/unlearn", json=payload)
    elapsed = time.time() - start
    
    data = response.json()
    metrics = data['metrics']
    
    print(f"[OK] Amostras removidas: {data['unlearned_samples']}")
    print(f"[OK] Accuracy antes: {metrics['accuracy_before']:.4f}")
    print(f"[OK] Accuracy depois: {metrics['accuracy_after']:.4f}")
    print(f"[OK] Drop: {metrics['accuracy_drop']:.4f}")
    print(f"[OK] Utility preservada: {metrics['utility_preserved_pct']:.2f}%")
    print(f"[OK] Tempo: {elapsed:.2f}s")
    
    return data

def main():
    """Executa teste completo"""
    print("")
    print("DynFrs Service Test")
    print("Machine Unlearning Workflow")
    print("")
    
    try:
        # 1. Health check
        test_health()
        
        # 2. Treinar modelo
        train_result = train_model()
        
        # 3. Unlearning
        unlearn_result = perform_unlearning(unlearn_count=100)
        
        # Resumo final
        print_section("RESUMO FINAL")
        
        utility_pct = unlearn_result['metrics']['utility_preserved_pct']
        
        print(f"\n{'Métrica':<30} {'Valor'}")
        print("-" * 60)
        print(f"{'Utility Preserved':<30} {utility_pct:.2f}%")
        print(f"{'Accuracy Drop':<30} {unlearn_result['metrics']['accuracy_drop']:.4f}")
        
        print("\n" + "="*60)
        print(" [PASS] TESTE CONCLUÍDO COM SUCESSO")
        print("="*60)
        print("\nPara ver métricas detalhadas no terminal:")
        print("  docker-compose logs -f dynfrs-privacy-service")
        
    except requests.exceptions.ConnectionError:
        print("\n[FAIL] Erro: Não foi possível conectar ao serviço.")
        print("   Certifique-se de que o Docker Compose está rodando:")
        print("   $ docker-compose up -d")
    except Exception as e:
        print(f"\n[FAIL] Erro: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
