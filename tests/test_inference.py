import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pytest
import torch
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.inference.predictor import AristotlePredictor


class TestAristotlePredictor(unittest.TestCase):
    """Тесты для класса AristotlePredictor"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        # Создаем временную директорию для модели
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = self.temp_dir.name
        
        # Создаем временный конфиг
        self.config = {
            'max_length': 100,
            'num_return_sequences': 1,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.2,
            'device': 'cpu'
        }
        
        # Сохраняем конфиг во временный файл
        self.config_path = os.path.join(self.temp_dir.name, 'config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        """Очистка после каждого теста"""
        self.temp_dir.cleanup()
    
    @patch('src.inference.predictor.GPT2Tokenizer.from_pretrained')
    @patch('src.inference.predictor.GPT2LMHeadModel.from_pretrained')
    def test_init_and_load_model(self, mock_model, mock_tokenizer):
        """Тест инициализации и загрузки модели"""
        # Настройка моков
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Создаем предиктор
        predictor = AristotlePredictor(self.model_path, self.config_path)
        
        # Проверки
        mock_tokenizer.assert_called_once_with(self.model_path)
        mock_model.assert_called_once_with(self.model_path)
        self.assertEqual(predictor.tokenizer, mock_tokenizer_instance)
        self.assertEqual(predictor.model, mock_model_instance)
        self.assertEqual(predictor.config['max_length'], self.config['max_length'])
        self.assertEqual(predictor.config['temperature'], self.config['temperature'])
    
    @patch('src.inference.predictor.GPT2Tokenizer.from_pretrained')
    @patch('src.inference.predictor.GPT2LMHeadModel.from_pretrained')
    def test_generate_text(self, mock_model, mock_tokenizer):
        """Тест генерации текста"""
        # Настройка моков
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
        mock_tokenizer_instance.decode.return_value = "Это сгенерированный текст."
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        # Создаем предиктор
        predictor = AristotlePredictor(self.model_path, self.config_path)
        
        # Генерируем текст
        prompt = "Начало текста"
        generated_texts = predictor.generate_text(prompt)
        
        # Проверки
        mock_tokenizer_instance.assert_called_once_with(prompt, return_tensors="pt")
        mock_model_instance.generate.assert_called_once()
        mock_tokenizer_instance.decode.assert_called_once()
        self.assertEqual(len(generated_texts), 1)
        self.assertEqual(generated_texts[0], "Это сгенерированный текст.")
    
    @patch('src.inference.predictor.GPT2Tokenizer.from_pretrained')
    @patch('src.inference.predictor.GPT2LMHeadModel.from_pretrained')
    def test_generate_text_with_custom_params(self, mock_model, mock_tokenizer):
        """Тест генерации текста с пользовательскими параметрами"""
        # Настройка моков
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
        mock_tokenizer_instance.decode.return_value = "Это сгенерированный текст."
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        # Создаем предиктор
        predictor = AristotlePredictor(self.model_path, self.config_path)
        
        # Генерируем текст с пользовательскими параметрами
        prompt = "Начало текста"
        custom_params = {
            'max_length': 200,
            'temperature': 1.0,
            'num_return_sequences': 2
        }
        predictor.generate_text(prompt, **custom_params)
        
        # Проверяем, что параметры были переданы в generate
        args, kwargs = mock_model_instance.generate.call_args
        self.assertEqual(kwargs['max_length'], custom_params['max_length'])
        self.assertEqual(kwargs['temperature'], custom_params['temperature'])
        self.assertEqual(kwargs['num_return_sequences'], custom_params['num_return_sequences'])
    
    @patch('src.inference.predictor.GPT2Tokenizer.from_pretrained')
    @patch('src.inference.predictor.GPT2LMHeadModel.from_pretrained')
    def test_get_model_info(self, mock_model, mock_tokenizer):
        """Тест получения информации о модели"""
        # Настройка моков
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.parameters.return_value = [torch.ones(10, 10) for _ in range(5)]
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        # Создаем предиктор
        predictor = AristotlePredictor(self.model_path, self.config_path)
        
        # Получаем информацию о модели
        model_info = predictor.get_model_info()
        
        # Проверки
        self.assertEqual(model_info['model_path'], self.model_path)
        self.assertEqual(model_info['device'], 'cpu')
        self.assertIn('parameters', model_info)
        self.assertIn('generation_config', model_info)
        self.assertEqual(model_info['generation_config']['max_length'], self.config['max_length'])
        self.assertEqual(model_info['generation_config']['temperature'], self.config['temperature'])
    
    @patch('src.inference.predictor.GPT2Tokenizer.from_pretrained')
    @patch('src.inference.predictor.GPT2LMHeadModel.from_pretrained')
    def test_adjust_generation_params(self, mock_model, mock_tokenizer):
        """Тест настройки параметров генерации"""
        # Настройка моков
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        # Создаем предиктор
        predictor = AristotlePredictor(self.model_path, self.config_path)
        
        # Исходные значения
        original_max_length = predictor.config['max_length']
        original_temperature = predictor.config['temperature']
        
        # Настраиваем параметры
        new_params = {
            'max_length': 200,
            'temperature': 1.0,
            'unknown_param': 'value'  # Неизвестный параметр
        }
        updated_config = predictor.adjust_generation_params(**new_params)
        
        # Проверки
        self.assertEqual(updated_config['max_length'], new_params['max_length'])
        self.assertEqual(updated_config['temperature'], new_params['temperature'])
        self.assertNotEqual(updated_config['max_length'], original_max_length)
        self.assertNotEqual(updated_config['temperature'], original_temperature)
        self.assertNotIn('unknown_param', updated_config)  # Неизвестный параметр не должен быть добавлен


if __name__ == '__main__':
    unittest.main()