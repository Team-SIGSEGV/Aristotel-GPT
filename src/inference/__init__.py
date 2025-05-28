"""
Модуль для инференса нейросети Аристотеля.

Включает классы и функции для загрузки обученной модели,
генерации текста и предоставления API.
"""

from .predictor import AristotlePredictor

__all__ = ['AristotlePredictor']