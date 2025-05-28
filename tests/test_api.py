import json
import unittest
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.inference.api import app, get_predictor


class TestAristotleAPI(unittest.TestCase):
    """Тесты для API нейросети Аристотеля"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.client = TestClient(app)
        
        # Создаем мок для предиктора
        self.mock_predictor = MagicMock()
        self.mock_predictor.generate_text.return_value = ["Сгенерированный философский текст."]
        self.mock_predictor.get_model_info.return_value = {
            "model_path": "./models",
            "device": "cpu",
            "parameters": 124000000,
            "generation_config": {
                "max_length": 200,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.2
            }
        }
    
    @patch('src.inference.api.get_predictor')
    def test_generate_text_endpoint(self, mock_get_predictor):
        """Тест эндпоинта генерации текста"""
        # Настройка мока
        mock_get_predictor.return_value = self.mock_predictor
        
        # Отправляем запрос
        request_data = {
            "prompt": "Что такое мудрость?",
            "max_length": 150,
            "temperature": 0.7,
            "num_return_sequences": 1
        }
        response = self.client.post("/predict", json=request_data)
        
        # Проверки
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("generated_texts", response_data)
        self.assertIn("prompt", response_data)
        self.assertIn("generation_params", response_data)
        self.assertIn("execution_time", response_data)
        self.assertEqual(response_data["prompt"], request_data["prompt"])
        self.assertEqual(len(response_data["generated_texts"]), 1)
        
        # Проверяем, что предиктор был вызван с правильными параметрами
        self.mock_predictor.generate_text.assert_called_once_with(
            request_data["prompt"],
            max_length=request_data["max_length"],
            temperature=request_data["temperature"],
            num_return_sequences=request_data["num_return_sequences"]
        )
    
    @patch('src.inference.api.get_predictor')
    def test_health_endpoint(self, mock_get_predictor):
        """Тест эндпоинта проверки состояния"""
        # Настройка мока
        mock_get_predictor.return_value = self.mock_predictor
        
        # Отправляем запрос
        response = self.client.get("/health")
        
        # Проверки
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["status"], "ok")
        self.assertIn("version", response_data)
        self.assertIn("model_info", response_data)
        
        # Проверяем информацию о модели
        model_info = response_data["model_info"]
        self.assertEqual(model_info["model_path"], "./models")
        self.assertEqual(model_info["device"], "cpu")
        self.assertIn("parameters", model_info)
        self.assertIn("generation_config", model_info)
    
    def test_metrics_endpoint(self):
        """Тест эндпоинта метрик"""
        # Отправляем запрос
        response = self.client.get("/metrics")
        
        # Проверки
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), str)  # Prometheus метрики возвращаются как строка
    
    @patch('src.inference.api.get_predictor')
    def test_invalid_request(self, mock_get_predictor):
        """Тест обработки невалидного запроса"""
        # Настройка мока
        mock_get_predictor.return_value = self.mock_predictor
        
        # Отправляем запрос с пустым промптом
        request_data = {
            "prompt": "",  # Пустой промпт
            "max_length": 150,
            "temperature": 0.7,
            "num_return_sequences": 1
        }
        response = self.client.post("/predict", json=request_data)
        
        # Проверки
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        response_data = response.json()
        self.assertIn("detail", response_data)
    
    @patch('src.inference.api.get_predictor')
    def test_invalid_parameters(self, mock_get_predictor):
        """Тест обработки невалидных параметров"""
        # Настройка мока
        mock_get_predictor.return_value = self.mock_predictor
        
        # Отправляем запрос с невалидными параметрами
        request_data = {
            "prompt": "Что такое мудрость?",
            "max_length": 5000,  # Слишком большое значение
            "temperature": 3.0,   # Слишком большое значение
            "num_return_sequences": 10  # Слишком большое значение
        }
        response = self.client.post("/predict", json=request_data)
        
        # Проверки
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        response_data = response.json()
        self.assertIn("detail", response_data)
    
    @patch('src.inference.api.get_predictor')
    def test_server_error(self, mock_get_predictor):
        """Тест обработки серверной ошибки"""
        # Настройка мока для имитации ошибки
        mock_get_predictor.return_value = self.mock_predictor
        self.mock_predictor.generate_text.side_effect = Exception("Ошибка генерации")
        
        # Отправляем запрос
        request_data = {
            "prompt": "Что такое мудрость?",
            "max_length": 150,
            "temperature": 0.7,
            "num_return_sequences": 1
        }
        response = self.client.post("/predict", json=request_data)
        
        # Проверки
        self.assertEqual(response.status_code, 500)  # Internal Server Error
        response_data = response.json()
        self.assertIn("detail", response_data)
        self.assertEqual(response_data["detail"], "Ошибка генерации")


if __name__ == '__main__':
    unittest.main()