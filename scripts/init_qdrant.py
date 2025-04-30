import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_links(file_path):
    """Чтение JSON-файла с ссылками"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Загружено {len(data)} записей из {file_path}")
            return data
    except Exception as e:
        logger.error(f"Ошибка при чтении {file_path}: {e}")
        return []

def initialize_qdrant():
    """Инициализация коллекции links в Qdrant"""
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
    
    # Подключение к Qdrant без API-ключа
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    logger.info(f"Подключено к Qdrant: {qdrant_host}:{qdrant_port}")
    
    # Загрузка модели LaBSE
    model_path = '/app/models/LaBSE'
    try:
        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
            logger.info(f"Модель LaBSE загружена из {model_path}")
        else:
            model = SentenceTransformer('sentence-transformers/LaBSE')
            logger.info("Модель LaBSE загружена из Hugging Face")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели LaBSE: {e}")
        raise
    
    vector_size = 768  # Размер вектора LaBSE
    collection_name = "links"
    
    # Загрузка данных из links.json
    links = load_links('/app/data/links.json')
    
    if client.collection_exists(collection_name):
        logger.info(f"Удаление старой коллекции {collection_name}")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Старая коллекция {collection_name} удалена")
    
    logger.info(f"Создание новой коллекции {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    logger.info(f"Коллекция {collection_name} создана")
    
    # Проверка данных
    if not links:
        logger.warning("Нет данных для загрузки в Qdrant")
        return
    
    # Векторизация и загрузка
    points = []
    for idx, item in enumerate(links):
        if 'name' not in item or 'link' not in item:
            logger.warning(f"Пропущен элемент без name или link: {item}")
            continue
        # Векторизация имени
        vector = model.encode(item['name']).tolist()
        # Создание точки
        point = PointStruct(
            id=idx,
            vector=vector,
            payload=item  # Весь объект как payload
        )
        points.append(point)
    
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Загружено {len(points)} точек в коллекцию {collection_name}")
    else:
        logger.warning("Нет точек для загрузки")

if __name__ == "__main__":
    initialize_qdrant()