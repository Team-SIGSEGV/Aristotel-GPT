# Aristotle Neural Network Project

Проект для обучения и развертывания нейросети на основе текстов Аристотеля.

## Структура проекта

```
aristotle-nn/
├── src/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── augmenter.py
│   │   └── utils.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── api.py
│   └── __init__.py
├── tests/
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile.training
│   ├── Dockerfile.inference
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── model-training.yml
├── configs/
│   ├── training_config.yaml
│   └── inference_config.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── deploy.py
├── requirements/
│   ├── base.txt
│   ├── training.txt
│   └── inference.txt
├── .gitignore
├── pyproject.toml
├── Makefile
└── README.md
```

## Быстрый старт

### Локальная разработка

```bash
# Клонирование репозитория
git clone <repository-url>
cd aristotle-nn

# Установка зависимостей
make install

# Запуск тестов
make test

# Обучение модели
make train

# Запуск API
make serve
```

### Docker

```bash
# Сборка образов
make docker-build

# Запуск обучения в контейнере
make docker-train

# Запуск API в контейнере
make docker-serve
```

## CI/CD Pipeline

Проект использует GitHub Actions для автоматизации:

- **CI Pipeline**: Тестирование кода при каждом push/PR
- **CD Pipeline**: Автоматическое развертывание при merge в main
- **Model Training Pipeline**: Переобучение модели по расписанию

## API Endpoints

- `POST /predict` - Генерация текста
- `GET /health` - Проверка состояния сервиса
- `GET /metrics` - Метрики модели

## Мониторинг

- Логирование через structlog
- Метрики через Prometheus
- Трейсинг через OpenTelemetry