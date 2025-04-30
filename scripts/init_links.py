import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

links_path = "data/links.json"

if not os.path.exists(links_path):
    logger.info("Файл с ссылками не найден. Создаётся файл по умолчанию")
    default_data = [
        {
            "name": "Общая информация",
            "link": "https://istokmw.ru/about-us/general-information/"
        },
        {
            "name": "Ценности",
            "link": "https://istokmw.ru/values/"
        },
        {
            "name": "Противодействие коррупции",
            "link": "https://istokmw.ru/anti-corruption/"
        },
        {
            "name": "Патенты и изобретения",
            "link": "https://istokmw.ru/patents-and-inventions/"
        },
        {
            "name": "История предприятия",
            "link": "https://istokmw.ru/history/"
        },
    ]
    
    with open(links_path, "w", encoding="utf-8") as file:
        json.dump(default_data, file, indent=4, ensure_ascii=False)
    
    logger.info("Файл {links_path} записан")
