import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import structlog
from datetime import datetime


def setup_logging(log_level: str = "INFO"):
    """Настройка структурированного логирования"""
    
    # Настройка structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Настройка стандартного логгера
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )


def save_metrics(metrics: Dict[str, Any], output_dir: str):
    """Сохранение метрик обучения"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_with_timestamp = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    
    metrics_file = output_path / "training_metrics.json"
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_with_timestamp, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Metrics saved to {metrics_file}")


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Загрузка сохраненных метрик"""
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_model_size(model_path: str) -> Dict[str, Any]:
    """Вычисление размера модели"""
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        return {"error": "Model directory not found"}
    
    total_size = 0
    file_count = 0
    
    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
            file_count += 1
    
    return {
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "file_count": file_count,
        "model_path": str(model_dir)
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """Валидация конфигурации обучения"""
    required_fields = [
        "model_name", "dataset_name", "max_length", 
        "test_size", "output_dir"
    ]
    
    for field in required_fields:
        if field not in config:
            logging.error(f"Missing required config field: {field}")
            return False
    
    # Проверка типов и диапазонов
    if not isinstance(config["max_length"], int) or config["max_length"] <= 0:
        logging.error("max_length must be a positive integer")
        return False
        
    if not isinstance(config["test_size"], float) or not 0 < config["test_size"] < 1:
        logging.error("test_size must be a float between 0 and 1")
        return False
    
    return True