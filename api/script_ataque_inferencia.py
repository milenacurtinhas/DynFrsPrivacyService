import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class MembershipInferenceAttack():
    """
    Experimento de Inferência de Membro (MIA)
    """
    def __init__(self, dataset_name="Adult", base_url="http://localhost:8000", model_manager=None):
        self.dataset_name = dataset_name
        self.base_url = base_url
        self.model_manager = model_manager  # If provided, use directly instead of HTTP
        self.attacker_model = LogisticRegression(random_state = 42)

        self.X_members = None
        self.X_non_members = None
        self.y_attack = None

    def _communicate_api(self, metodo, endpoint, payload=None):
        """Método privado para enviar requisições para a API. Se o Docker estiver desligado, usa simulação com dados mockados"""
        url = f"{self.base_url}{endpoint}"
    
        try:
        # Tentativa de conectar com Docker
            if metodo == "POST":
                response = requests.post(url, json=payload)
            elif metodo == "GET":
                response = requests.get(url)

            response.raise_for_status()
            return response.json()
    
        except requests.exceptions.ConnectionError:
            # Modo simulação (para Docker desativado)
            logger.warning(f"[!] Docker desligado em {url}. Usando modo de simulação...")
        
            if endpoint == "/health":
                return {"status": "Simulacao_Ativa"}
            elif endpoint == "/model/train":
                return {"status": "success", "model_id": "mock_id_123", "metrics": {"train_accuracy": 0.98}}
            elif endpoint == "/model/predict":
                mock_proba = np.random.rand(len(payload["data"]), 2).tolist()
                return {"status": "success", "probabilities": mock_proba}
            elif endpoint == "/model/unlearn":
                return {"status": "success", "unlearned_samples": 500}
            
    def load_data(self, sample_size=5000):
        """Carrega os datasets locais e extrai uma amostra"""
        train_path = f"Datasets/{self.dataset_name}/train.txt"
        test_path = f"Datasets/{self.dataset_name}/test.txt"

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Dataset de treino não encontrado: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Dataset de teste não encontrado: {test_path}")
        
        # Carregar dados (skiprows=1 para pular metadados da primeira linha)
        train_data = np.loadtxt(train_path, skiprows=1)
        test_data = np.loadtxt(test_path, skiprows=1)

        # Separação dos dados que já foram "vistos" e os "não vistos" e limita ao tamanho da amostra
        self.X_members = train_data[:sample_size, :-1]
        self.X_non_members = test_data[:sample_size, :-1]

        self.y_attack = np.concatenate((
            np.ones(len(self.X_members)), 
            np.zeros(len(self.X_non_members))
        ) )

        logger.info(f"Dados carregados: {len(self.X_members)} membros e {len(self.X_non_members)} não-membros.")
    
    def initialize_model(self):
        """Faz o health check e ordena o treinamento do modelo alvo no servidor"""
        if self.model_manager:
            # Use model_manager directly (API internal mode)
            self.model_manager.train(
                dataset_name=self.dataset_name,
                num_trees=100,
                max_depth=15,
                min_split_size=10,
                k=10
            )
            logger.info("Modelo inicializado e treinado diretamente.")
        else:
            # Use HTTP API (external mode)
            self._communicate_api("GET", "/health")

            payload_train = {
                "dataset": self.dataset_name,
                "num_trees": 100,
                "max_depth": 15,
                "min_split_size": 10,
                "k": 10
            }

            self._communicate_api("POST", "/model/train", payload_train)
            logger.info("Modelo inicializado e treinado no servidor.")

    def get_attack_features(self):
        """Recebe as probabilidades da API para formar o dataset do atacante"""
        if self.model_manager:
            # Use model_manager directly (API internal mode)
            probs_members = self.model_manager.predict_batch(self.X_members)
            probs_non_members = self.model_manager.predict_batch(self.X_non_members)
            X_attack = np.vstack((probs_members, probs_non_members))
        else:
            # Use HTTP API (external mode)
            resp_members = self._communicate_api("POST", "/model/predict", {"data" : self.X_members.tolist()})
            resp_non_members = self._communicate_api("POST", "/model/predict", {"data" : self.X_non_members.tolist()})

            # Concatena as saídas da API em uma matriz única
            # O atacante vai usar somente as distribuições de probabilidade para aprender a separar as classes
            X_attack = np.vstack((resp_members['probabilities'], resp_non_members['probabilities']))
        return X_attack

    def execute_unlearning(self, count=500):
        """Solicita ao servidor a remoção de N amostras do modelo"""
        if self.model_manager:
            # Use model_manager directly (API internal mode)
            self.model_manager.unlearn_random(unlearn_count=count)
            logger.info(f"Modelo atualizado após unlearning de {count} amostras")
        else:
            # Use HTTP API (external mode)
            payload_unlearn = {
                "unlearn_count": count
            }

            self._communicate_api("POST", "/model/unlearn", payload_unlearn)
            logger.info(f"Modelo atualizado após unlearning de {count} amostras")

    def run(self):
        """Orquestra a execução completa do experimento do ataque"""
        #Preparação dos dados e do modelo
        self.load_data()
        self.initialize_model()
        
        #Parte inicial do ataque e análise de vulnerabilidades
        X_attack_pre = self.get_attack_features()

        self.attacker_model.fit(X_attack_pre, self.y_attack)

        predict_pre = self.attacker_model.predict(X_attack_pre)
        accuracy_pre = accuracy_score(self.y_attack, predict_pre)
        print(f"Acurácia do vazamento: {accuracy_pre * 100:.2f}%")

        #Execução do unlearning de N amostras
        self.execute_unlearning(count=500)

        #Reavaliação dos resultados
        X_attack_pos = self.get_attack_features()

        predict_pos = self.attacker_model.predict(X_attack_pos)
        accuracy_pos = accuracy_score(self.y_attack, predict_pos)
        print(f"Acurácia do Ataque: {accuracy_pos * 100:.2f}%")


if __name__ == "__main__":
    attack = MembershipInferenceAttack(dataset_name = "Adult")
    attack.run()
