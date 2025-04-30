import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = "models/LaBSE"

if not os.path.exists(model_path):
    logger.info("Модель LaBSE не найдена")
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/LaBSE"

    os.makedirs(model_path, exist_ok=True)

    # Download and save the model
    logger.info("Загрузка модели LaBSE")
    model = SentenceTransformer(model_name)
    logger.info("Сохранение модели LaBSE")
    model.save(model_path)
else:
    logger.info(f"Модель LaBSE обнаружена по пути: {model_path}")