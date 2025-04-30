import os
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
#from qdrant_client.http.models import Filter

from src.config import EmbeddingConfig, VectorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorSearch:
    def __init__(self):
        """Инициализация эмбеддера LaBSE и клиента Qdrant"""
        # Загрузка модели LaBSE
        model_path = EmbeddingConfig.get_model_path()
        device = EmbeddingConfig.get_device()
        try:
            if os.path.exists(model_path):
                self.model = SentenceTransformer(model_path, device=device)
                logger.info(f"Модель LaBSE загружена из {model_path}")
            else:
                self.model = SentenceTransformer('sentence-transformers/LaBSE', device=device)
                logger.info("Модель LaBSE загружена из Hugging Face")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели LaBSE: {e}")
            raise
        
        # Подключение к Qdrant
        qdrant_host = VectorConfig.get_qdrant_host()
        qdrant_port = VectorConfig.get_qdrant_port()
        try:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            logger.info(f"Подключено к Qdrant: {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {e}")
            raise
        
        self.collection_name = VectorConfig.get_collection_name()

    def find_nearest(self, query: str, limit: int = 1):
        """Векторизация строки и поиск ближайшей точки в коллекции links"""
        try:
            # Векторизация запроса
            vector = self.model.encode(query).tolist()
            logger.debug(f"Вектор для запроса '{query}' сгенерирован: {vector[:5]}...")

            # Поиск ближайшей точки
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            if not search_result:
                logger.warning(f"Для запроса '{query}' не найдено точек")
                return None

            # Возвращаем payload ближайшей точки
            nearest = search_result[0]
            logger.info(f"Найдена ближайшая точка: ID={nearest.id}, Score={nearest.score}, Payload={nearest.payload}")
            return {
                "id": nearest.id,
                "score": nearest.score,
                "payload": nearest.payload
            }

        except Exception as e:
            logger.error(f"Ошибка при поиске ближайшей точки: {e}")
            return None

vector_search = VectorSearch()