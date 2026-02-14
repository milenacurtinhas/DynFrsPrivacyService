"""
Model Manager - Gerencia o modelo DynFrs C++ via Python
Responsável por treinamento, predição, unlearning e serialização
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import uuid
import os
import psutil
import json

# Import do wrapper C++ (será criado com pybind11)
try:
    import dynfrs_cpp
except ImportError:
    logging.warning("dynfrs_cpp não disponível. Usando modo simulação.")
    dynfrs_cpp = None

logger = logging.getLogger(__name__)


class ModelManager:
    """Gerenciador do modelo Random Forest dinâmico"""
    
    def __init__(self, models_dir: str = "/app/models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Estado do modelo
        self.model = None
        self.model_id = None
        self.dataset_name = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        
        # Metadados
        self.created_at = None
        self.last_modified = None
        self.unlearn_operations = []
        self.config = {}
        
        logger.info("ModelManager inicializado")
    
    def is_loaded(self) -> bool:
        """Verifica se há um modelo carregado"""
        return self.model is not None
    
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carrega dataset do diretório Datasets/
        
        Formatos esperados:
        - train.txt e test.txt com dados separados por espaço
        - Última coluna é o label (0 ou 1)
        """
        train_path = f"/app/Datasets/{dataset_name}/train.txt"
        test_path = f"/app/Datasets/{dataset_name}/test.txt"
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Dataset de treino não encontrado: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Dataset de teste não encontrado: {test_path}")
        
        # Carregar dados (skiprows=1 para pular metadados da primeira linha)
        train_data = np.loadtxt(train_path, skiprows=1)
        test_data = np.loadtxt(test_path, skiprows=1)
        
        # Separar features e labels
        X_train = train_data[:, :-1]
        Y_train = train_data[:, -1].astype(int)
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1].astype(int)
        
        logger.info(f"Dataset carregado: {dataset_name}")
        logger.info(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, Y_train, X_test, Y_test
    
    def train(self, 
              dataset_name: str,
              num_trees: int = 100,
              max_depth: int = 15,
              min_split_size: int = 10,
              k: int = 10) -> Dict:
        """
        Treina um novo modelo Random Forest
        
        Args:
            dataset_name: Nome do dataset (Adult, Vaccine, NoShow)
            num_trees: Número de árvores na floresta
            max_depth: Profundidade máxima das árvores
            min_split_size: Tamanho mínimo para split
            k: Número de árvores que cada amostra aparece
            
        Returns:
            Dict com métricas de treinamento
        """
        try:
            # Carregar dataset
            self.dataset_name = dataset_name
            self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_dataset(dataset_name)
            
            # Configuração
            self.config = {
                "num_trees": num_trees,
                "max_depth": max_depth,
                "min_split_size": min_split_size,
                "k": k,
                "dataset": dataset_name
            }
            
            # Treinar modelo C++
            if dynfrs_cpp:
                self.model = dynfrs_cpp.RandomForest(
                    X=self.X_train.tolist(),
                    Y=self.Y_train.tolist(),
                    T=num_trees,
                    k=k,
                    max_dep=max_depth,
                    min_split_size=min_split_size
                )
            else:
                # Modo simulação (para desenvolvimento sem C++)
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=num_trees,
                    max_depth=max_depth,
                    min_samples_split=min_split_size,
                    random_state=42
                )
                self.model.fit(self.X_train, self.Y_train)
            
            # Metadados
            self.model_id = str(uuid.uuid4())[:8]
            self.created_at = datetime.now().isoformat()
            self.last_modified = self.created_at
            self.unlearn_operations = []
            
            # Avaliar
            accuracy = self.evaluate()["accuracy"]
            
            logger.info(f"Modelo treinado: {self.model_id}, accuracy={accuracy:.4f}")
            
            return {
                "model_id": self.model_id,
                "accuracy": accuracy,
                "num_samples": len(self.X_train),
                "num_features": self.X_train.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {str(e)}")
            raise
    
    def predict(self, features: List[float]) -> np.ndarray:
        """
        Realiza predição para uma amostra
        
        Args:
            features: Lista de features
            
        Returns:
            Array de probabilidades [P(class_0), P(class_1)]
        """
        if not self.is_loaded():
            raise ValueError("Nenhum modelo carregado")
        
        X = np.array(features).reshape(1, -1)
        
        if dynfrs_cpp:
            # Chamar C++ qry
            probs = self.model.query(features)
            return np.array(probs)
        else:
            # Modo simulação
            probs = self.model.predict_proba(X)[0]
            return probs
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Prediz para múltiplas amostras"""
        if not self.is_loaded():
            raise ValueError("Nenhum modelo carregado")
        
        if dynfrs_cpp:
            # Predição batch via C++
            results = []
            for features in X:
                probs = self.model.query(features.tolist())
                results.append(probs)
            return np.array(results)
        else:
            return self.model.predict_proba(X)
    
    def unlearn_random(self, unlearn_count: int) -> Dict:
        """
        Remove N amostras aleatórias do modelo (compatível com DynFrs C++ -unl_cnt)
        
        Args:
            unlearn_count: Número de amostras a remover aleatoriamente
            
        Returns:
            Dict com informações da operação
        """
        if not self.is_loaded():
            raise ValueError("Nenhum modelo carregado")
        
        if unlearn_count <= 0 or unlearn_count > len(self.X_train):
            raise ValueError(f"unlearn_count deve estar entre 1 e {len(self.X_train)}")
        
        # Selecionar amostras aleatórias
        import time
        start_time = time.time()
        
        random_ids = np.random.choice(len(self.X_train), size=unlearn_count, replace=False)
        
        logger.info(f"Unlearning {unlearn_count} amostras aleatórias")
        
        # Modo Python: retreinar sem as amostras
        mask = np.ones(len(self.X_train), dtype=bool)
        mask[random_ids] = False
        X_new = self.X_train[mask]
        Y_new = self.Y_train[mask]
        
        self.model.fit(X_new, Y_new)
        self.X_train = X_new
        self.Y_train = Y_new
        
        unlearning_time = time.time() - start_time
        
        # Registrar operação
        operation = {
            "operation_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "samples_removed": unlearn_count,
            "sample_ids": random_ids.tolist(),
            "unlearning_time": unlearning_time
        }
        self.unlearn_operations.append(operation)
        self.last_modified = operation["timestamp"]
        
        logger.info(f"Unlearning concluído: operation_id={operation['operation_id']}, tempo={unlearning_time:.2f}s")
        
        return operation
    
    def evaluate(self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None) -> Dict:
        """
        Avalia o modelo no dataset de teste
        
        Returns:
            Dict com métricas (accuracy, precision, recall, f1)
        """
        if not self.is_loaded():
            raise ValueError("Nenhum modelo carregado")
        
        if X is None:
            X = self.X_test
            Y = self.Y_test
        
        # Predições
        probs = self.predict_batch(X)
        predictions = np.argmax(probs, axis=1)
        
        # Métricas
        accuracy = np.mean(predictions == Y)
        
        # Métricas por classe (binário)
        tp = np.sum((predictions == 1) & (Y == 1))
        fp = np.sum((predictions == 1) & (Y == 0))
        fn = np.sum((predictions == 0) & (Y == 1))
        tn = np.sum((predictions == 0) & (Y == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "test_samples": len(Y)
        }
    
    def serialize(self, filename: Optional[str] = None) -> str:
        """
        Serializa o modelo para arquivo binário
        
        Args:
            filename: Nome do arquivo (default: model_{model_id}.bin)
            
        Returns:
            Caminho completo do arquivo salvo
        """
        if not self.is_loaded():
            raise ValueError("Nenhum modelo carregado")
        
        if filename is None:
            filename = f"model_{self.model_id}.bin"
        
        filepath = os.path.join(self.models_dir, filename)
        
        if dynfrs_cpp:
            # Chamar serialização C++
            self.model.serialize(filepath)
        else:
            # Modo simulação: usar pickle
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'X_train': self.X_train,
                    'Y_train': self.Y_train,
                    'metadata': {
                        'model_id': self.model_id,
                        'dataset_name': self.dataset_name,
                        'created_at': self.created_at,
                        'config': self.config,
                        'unlearn_operations': self.unlearn_operations
                    }
                }, f)
        
        # Salvar metadados separadamente
        metadata_path = filepath.replace('.bin', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'model_id': self.model_id,
                'dataset_name': self.dataset_name,
                'created_at': self.created_at,
                'last_modified': self.last_modified,
                'config': self.config,
                'unlearn_operations': self.unlearn_operations
            }, f, indent=2)
        
        logger.info(f"Modelo serializado: {filepath}")
        return filepath
    
    def deserialize(self, filename: str):
        """
        Carrega modelo de arquivo serializado
        
        Args:
            filename: Nome ou caminho completo do arquivo
        """
        if not os.path.isabs(filename):
            filepath = os.path.join(self.models_dir, filename)
        else:
            filepath = filename
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        if dynfrs_cpp:
            # Carregar via C++
            self.model = dynfrs_cpp.RandomForest.deserialize(filepath)
        else:
            # Modo simulação
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.X_train = data['X_train']
                self.Y_train = data['Y_train']
                metadata = data['metadata']
                self.model_id = metadata['model_id']
                self.dataset_name = metadata['dataset_name']
                self.created_at = metadata['created_at']
                self.config = metadata['config']
                self.unlearn_operations = metadata['unlearn_operations']
        
        # Carregar metadados
        metadata_path = filepath.replace('.bin', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_id = metadata.get('model_id', self.model_id)
                self.dataset_name = metadata.get('dataset_name', self.dataset_name)
                self.created_at = metadata.get('created_at', self.created_at)
                self.last_modified = metadata.get('last_modified', self.created_at)
                self.config = metadata.get('config', self.config)
                self.unlearn_operations = metadata.get('unlearn_operations', [])
        
        # Recarregar dataset para teste
        if self.dataset_name:
            _, _, self.X_test, self.Y_test = self.load_dataset(self.dataset_name)
        
        logger.info(f"Modelo carregado: {self.model_id}")
    
    def get_info(self) -> Dict:
        """Retorna informações sobre o modelo"""
        if not self.is_loaded():
            raise ValueError("Nenhum modelo carregado")
        
        return {
            "model_id": self.model_id,
            "dataset": self.dataset_name,
            "num_trees": self.config.get("num_trees", 0),
            "num_samples": len(self.X_train) if self.X_train is not None else 0,
            "num_features": self.X_train.shape[1] if self.X_train is not None else 0,
            "created_at": self.created_at,
            "unlearn_operations": len(self.unlearn_operations),
            "last_modified": self.last_modified
        }
    
    def get_detailed_metrics(self) -> Dict:
        """Retorna métricas detalhadas do modelo"""
        metrics = self.evaluate()
        metrics.update({
            "unlearn_count": len(self.unlearn_operations),
            "model_id": self.model_id,
            "dataset": self.dataset_name
        })
        return metrics
    
    def get_unlearn_history(self) -> List[Dict]:
        """Retorna histórico de operações de unlearning"""
        return self.unlearn_operations
    
    def get_num_features(self) -> int:
        """Retorna número de features esperadas"""
        if self.X_train is not None:
            return self.X_train.shape[1]
        return 0
    
    def get_memory_usage(self) -> Dict:
        """Retorna uso de memória do processo"""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
