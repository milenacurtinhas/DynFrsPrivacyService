"""
DynFrs Service - API REST para Machine Unlearning com DynFrs C++
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
import logging
from datetime import datetime
import json
import os

from model_manager import ModelManager

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DynFrs Service",
    description="Machine Unlearning com DynFrs C++ original",
    version="2.0.0"
)

# Gerenciador global do modelo
model_manager = ModelManager()

# Modelos de Dados (Pydantic)

class TrainRequest(BaseModel):
    dataset: str = Field(..., description="Nome do dataset (Adult, Vaccine, NoShow)")
    num_trees: int = Field(100, ge=1, le=500, description="Número de árvores")
    max_depth: int = Field(15, ge=1, le=30, description="Profundidade máxima")
    min_split_size: int = Field(10, ge=2, le=100, description="Tamanho mínimo para split")
    k: int = Field(10, ge=1, le=50, description="Número de árvores por amostra")



class UnlearnRequest(BaseModel):
    unlearn_count: int = Field(..., ge=1, le=10000, description="Número de amostras a desaprender")

class ModelInfo(BaseModel):
    model_id: str
    dataset: str
    num_trees: int
    num_samples: int
    num_features: int
    created_at: str
    unlearn_operations: int
    last_modified: str

# Endpoints Principais

@app.get("/")
async def root():
    """Informações sobre o serviço"""
    return {
        "service": "DynFrs Service",
        "version": "2.0.0",
        "description": "Machine Unlearning com DynFrs C++",
        "endpoints": {
            "health": "/health",
            "train": "POST /model/train",
            "unlearn": "POST /model/unlearn"
        }
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde do serviço"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager.is_loaded(),
        "memory_usage": model_manager.get_memory_usage()
    }

@app.post("/model/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Treina um novo modelo Random Forest
    
    - Carrega dataset especificado
    - Treina modelo DynFrs com serialização
    - Retorna métricas iniciais de acurácia
    """
    try:
        logger.info(f"Iniciando treinamento: dataset={request.dataset}, trees={request.num_trees}")
        
        # Carregar dataset
        dataset_path = f"/app/Datasets/{request.dataset}"
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset {request.dataset} não encontrado")
        
        # Treinar modelo
        result = model_manager.train(
            dataset_name=request.dataset,
            num_trees=request.num_trees,
            max_depth=request.max_depth,
            min_split_size=request.min_split_size,
            k=request.k
        )
        
        # Imprimir métricas no terminal
        print("\n" + "="*60)
        print("TREINAMENTO CONCLUÍDO")
        print("="*60)
        print(f"Model ID: {result['model_id']}")
        print(f"Dataset: {request.dataset}")
        print(f"Samples: {result['num_samples']}")
        print(f"Features: {result['num_features']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("="*60 + "\n")
        
        logger.info(f"Treinamento concluído: accuracy={result['accuracy']:.4f}")
        
        return {
            "status": "success",
            "model_id": result["model_id"],
            "dataset": request.dataset,
            "metrics": {
                "train_accuracy": result["accuracy"],
                "train_samples": result["num_samples"],
                "features": result["num_features"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/unlearn")
async def unlearn(request: UnlearnRequest):
    """
    Realiza machine unlearning - Remove N amostras aleatórias do modelo
    
    - Remove amostras aleatórias
    - Mostra métricas no terminal
    - Retorna impacto do unlearning
    """
    try:
        if not model_manager.is_loaded():
            raise HTTPException(status_code=400, detail="Nenhum modelo carregado")
        
        logger.info(f"Iniciando unlearning: {request.unlearn_count} amostras")
        
        # Métricas pré-unlearning
        metrics_before = model_manager.evaluate()
        
        # Realizar unlearning
        result = model_manager.unlearn_random(unlearn_count=request.unlearn_count)
        
        # Métricas pós-unlearning
        metrics_after = model_manager.evaluate()
        
        # Calcular impacto
        accuracy_drop = metrics_before["accuracy"] - metrics_after["accuracy"]
        utility_preserved = (metrics_after["accuracy"] / metrics_before["accuracy"]) * 100
        
        # Imprimir métricas no terminal
        print("\n" + "="*60)
        print("UNLEARNING CONCLUÍDO")
        print("="*60)
        print(f"Amostras removidas: {request.unlearn_count}")
        print(f"Accuracy antes: {metrics_before['accuracy']:.4f}")
        print(f"Accuracy depois: {metrics_after['accuracy']:.4f}")
        print(f"Dif accuracy: {accuracy_drop:.4f}")
        print(f"Utilidade preservada: {utility_preserved:.2f}%")
        print(f"Tempo: {result.get('unlearning_time', 0):.2f}s")
        print("="*60 + "\n")
        
        logger.info(f"Unlearning concluído: utility_preserved={utility_preserved:.2f}%")
        
        return {
            "status": "success",
            "unlearned_samples": request.unlearn_count,
            "metrics": {
                "accuracy_before": metrics_before["accuracy"],
                "accuracy_after": metrics_after["accuracy"],
                "accuracy_drop": accuracy_drop,
                "utility_preserved_pct": utility_preserved
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro no unlearning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
