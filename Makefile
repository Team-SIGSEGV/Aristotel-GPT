# Makefile для проекта Aristotle Neural Network

.PHONY: install install-dev test lint format clean train evaluate serve docker-build docker-train docker-serve

# Переменные
PYTHON := python
PIP := pip
PYTEST := pytest
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Пути
SRC_DIR := src
TESTS_DIR := tests
CONFIG_DIR := configs
SCRIPTS_DIR := scripts
DOCKER_DIR := docker
MODEL_DIR := models
DEPLOY_DIR := deployment

# Конфигурационные файлы
TRAINING_CONFIG := $(CONFIG_DIR)/training_config.yaml
INFERENCE_CONFIG := $(CONFIG_DIR)/inference_config.yaml

# Установка зависимостей
install:
	$(PIP) install -r requirements/base.txt

install-dev: install
	$(PIP) install -r requirements/training.txt
	$(PIP) install -r requirements/inference.txt
	$(PIP) install -e .

# Тестирование
test:
	$(PYTEST) $(TESTS_DIR)

test-training:
	$(PYTEST) $(TESTS_DIR)/test_training.py

test-inference:
	$(PYTEST) $(TESTS_DIR)/test_inference.py

test-api:
	$(PYTEST) $(TESTS_DIR)/test_api.py

# Линтинг и форматирование
lint:
	flake8 $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	isort --check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	black --check $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

format:
	isort $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)
	black $(SRC_DIR) $(TESTS_DIR) $(SCRIPTS_DIR)

# Очистка временных файлов
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Обучение модели
train:
	$(PYTHON) $(SCRIPTS_DIR)/train.py --config $(TRAINING_CONFIG)

train-quick:
	$(PYTHON) $(SCRIPTS_DIR)/train.py --config $(TRAINING_CONFIG) --skip-optuna

# Оценка модели
evaluate:
	$(PYTHON) $(SCRIPTS_DIR)/evaluate.py --model-path $(MODEL_DIR)

# Запуск API
serve:
	$(PYTHON) -m src.inference.api

# Docker команды
docker-build:
	$(DOCKER) build -f $(DOCKER_DIR)/Dockerfile.training -t aristotle-training:latest .
	$(DOCKER) build -f $(DOCKER_DIR)/Dockerfile.inference -t aristotle-inference:latest .

docker-train:
	$(DOCKER) run --rm -v $(PWD)/$(MODEL_DIR):/app/$(MODEL_DIR) aristotle-training:latest

docker-serve:
	$(DOCKER) run -d -p 8000:8000 -v $(PWD)/$(MODEL_DIR):/app/$(MODEL_DIR) --name aristotle-inference aristotle-inference:latest

docker-compose-up:
	$(DOCKER_COMPOSE) -f $(DOCKER_DIR)/docker-compose.yml up -d

docker-compose-down:
	$(DOCKER_COMPOSE) -f $(DOCKER_DIR)/docker-compose.yml down

# Развертывание
deploy:
	$(PYTHON) $(SCRIPTS_DIR)/deploy.py --model-path $(MODEL_DIR) --deploy-dir $(DEPLOY_DIR)

deploy-docker:
	$(PYTHON) $(SCRIPTS_DIR)/deploy.py --model-path $(MODEL_DIR) --deploy-dir $(DEPLOY_DIR) --docker