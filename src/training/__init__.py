"""
Модуль для обучения нейросети Аристотеля.

Включает классы и функции для загрузки данных, обучения модели,
оптимизации гиперпараметров и аугментации данных.
"""

from .trainer import AristotleTrainer
from .augmenter import TextAugmenter
from .utils import setup_logging, save_metrics, load_metrics, calculate_model_size, validate_config

__all__ = [
    'AristotleTrainer',
    'TextAugmenter',
    'setup_logging',
    'save_metrics',
    'load_metrics',
    'calculate_model_size',
    'validate_config'
]