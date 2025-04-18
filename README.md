# OCR API с поддержкой GigaChat

Этот проект представляет собой API-сервис для распознавания текста с изображений чеков и интеллектуального анализа с помощью GigaChat.

## 🛠️ Технологии / Стек

- **Python 3.10+**
- **FastAPI** — фреймворк для создания REST API
- **Uvicorn** — ASGI-сервер для запуска приложения
- **Pytesseract** — OCR-движок для распознавания текста
- **Pillow** — работа с изображениями
- **GigaChat SDK** — генерация ответов с помощью модели ИИ от Сбера
- **python-dotenv** — загрузка конфигураций из `.env` файла
- **Swagger UI / ReDoc** — авто-документация API

## 📦 Возможности

- Распознавание текста с изображения чека (OCR)
- Корректировка и нормализация названий блюд
- Выделение цен и количества
- Формирование итоговой суммы
- Генерация возможных комбинаций обедов
- Интерактивная документация API

## 🚀 Быстрый старт

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
