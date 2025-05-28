from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import logging
import time
import os
from pathlib import Path
import yaml
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import uvicorn

from .predictor import AristotlePredictor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Создаем отдельный реестр для метрик, чтобы избежать конфликтов
registry = CollectorRegistry()

# Метрики Prometheus
REQUEST_COUNT = Counter('aristotle_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'],
                        registry=registry)
REQUEST_LATENCY = Histogram('aristotle_api_request_duration_seconds', 'Request latency in seconds',
                            ['method', 'endpoint'], registry=registry)
GENERATION_TOKENS = Histogram('aristotle_generation_tokens', 'Number of tokens in generated text', registry=registry)
MODEL_TEMPERATURE = Gauge('aristotle_model_temperature', 'Current model temperature setting', registry=registry)


# Модели данных
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Начальный текст для генерации")
    max_length: Optional[int] = Field(200, ge=10, le=1000, description="Максимальная длина генерируемого текста")
    temperature: Optional[float] = Field(0.8, ge=0.1, le=2.0, description="Температура генерации (разнообразие)")
    num_return_sequences: Optional[int] = Field(1, ge=1, le=5, description="Количество вариантов генерации")

    @field_validator('prompt')
    @classmethod
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Промпт не может быть пустым')
        return v


class GenerationResponse(BaseModel):
    generated_texts: List[str]
    prompt: str
    generation_params: Dict[str, Any]
    execution_time: float


# Инициализация FastAPI
app = FastAPI(
    title="Aristotle Text Generation API",
    description="API для генерации философских текстов в стиле Аристотеля",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware для логирования и метрик
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    # Обновляем метрики
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(time.time() - start_time)

    # Логируем запрос
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {time.time() - start_time:.4f}s"
    )

    return response


# Зависимость для получения предиктора
def get_predictor():
    # Загружаем конфигурацию
    config_path = os.environ.get("CONFIG_PATH", "configs/inference_config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        config = {"model_path": "models/aristotle-gpt"}

    # Создаем предиктор
    try:
        predictor = AristotlePredictor(config.get("model_path", "models/aristotle-gpt"))
        return predictor
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize model")


@app.get("/")
async def root():
    """Корневой эндпоинт для проверки работоспособности API"""
    return {"message": "Aristotle Text Generation API is running"}


@app.get("/health")
async def health_check():
    """Эндпоинт для проверки здоровья сервиса"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/metrics")
async def metrics():
    """Эндпоинт для получения метрик Prometheus"""
    return JSONResponse(content=prometheus_client.generate_latest(registry).decode("utf-8"))


@app.post("/predict", response_model=GenerationResponse)
async def generate_text(
        request: GenerationRequest,
        background_tasks: BackgroundTasks,
        predictor: AristotlePredictor = Depends(get_predictor)
):
    """Генерация текста на основе промпта"""
    start_time = time.time()

    try:
        # Обновляем метрику температуры
        MODEL_TEMPERATURE.set(request.temperature)

        # Генерация текста
        generation_params = {
            "max_length": request.max_length,
            "temperature": request.temperature,
            "num_return_sequences": request.num_return_sequences
        }

        generated_texts = predictor.generate_text(
            request.prompt,
            **generation_params
        )

        # Обновляем метрику токенов
        for text in generated_texts:
            GENERATION_TOKENS.observe(len(text.split()))

        # Логирование в фоне
        background_tasks.add_task(
            logger.info,
            f"Generated {len(generated_texts)} texts for prompt: {request.prompt[:50]}..."
        )

        return GenerationResponse(
            generated_texts=generated_texts,
            prompt=request.prompt,
            generation_params=generation_params,
            execution_time=time.time() - start_time
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server():
    """Запуск сервера"""
    # Получаем порт из переменной окружения или используем порт по умолчанию
    port = int(os.environ.get("PORT", 8000))

    # Запускаем сервер
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()