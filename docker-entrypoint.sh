#!/usr/bin/env bash

# Проверка файла с ссылками
python scripts/init_links.py

# Загрузка модели LaBSE  
python scripts/init_models.py

# Инициализация коллекции links в Qdrant
python scripts/init_qdrant.py

# Запуск FastAPI сервера
uvicorn src.server:app --host 0.0.0.0 --port 8000