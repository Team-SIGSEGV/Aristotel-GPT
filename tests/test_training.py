import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.training.trainer import AristotleTrainer
from src.training.augmenter import TextAugmenter
from src.training.utils import validate_config, calculate_model_size


class TestAristotleTrainer(unittest.TestCase):
    """Тесты для класса AristotleTrainer"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        # Создаем временный конфиг для тестов
        self.test_config = {
            'model_name': 'sberbank-ai/rugpt3small_based_on_gpt2',
            'dataset_name': 'DmitryYarov/aristotle-russian',
            'max_length': 128,  # Меньше для быстрых тестов
            'test_size': 0.1,
            'optuna_trials': 1,  # Минимум для тестов
            'optuna_timeout': 60,  # Короткий таймаут для тестов
            'output_dir': './test_models',
            'log_level': 'INFO'
        }
        
        # Создаем временную директорию для выходных данных
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_config['output_dir'] = self.temp_dir.name
    
    def tearDown(self):
        """Очистка после каждого теста"""
        self.temp_dir.cleanup()
    
    @patch('src.training.trainer.load_dataset')
    def test_load_data(self, mock_load_dataset):
        """Тест загрузки данных"""
        # Настройка мока
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = mock_dataset
        mock_dataset.train_test_split.return_value = {
            'train': MagicMock(spec=[]),
            'test': MagicMock(spec=[])
        }
        mock_load_dataset.return_value = mock_dataset
        
        # Создаем тренера и вызываем метод
        trainer = AristotleTrainer()
        trainer.config = self.test_config
        dataset = trainer.load_data()
        
        # Проверки
        mock_load_dataset.assert_called_once_with(
            self.test_config['dataset_name'], split="train"
        )
        mock_dataset.train_test_split.assert_called_once_with(
            test_size=self.test_config['test_size']
        )
        self.assertIn('train', dataset)
        self.assertIn('test', dataset)
    
    @patch('src.training.trainer.GPT2Tokenizer.from_pretrained')
    @patch('src.training.trainer.GPT2LMHeadModel.from_pretrained')
    def test_setup_model_and_tokenizer(self, mock_model, mock_tokenizer):
        """Тест инициализации модели и токенизатора"""
        # Настройка моков
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Создаем тренера и вызываем метод
        trainer = AristotleTrainer()
        trainer.config = self.test_config
        trainer.setup_model_and_tokenizer()
        
        # Проверки
        mock_tokenizer.assert_called_once_with(self.test_config['model_name'])
        mock_model.assert_called_once_with(self.test_config['model_name'])
        self.assertEqual(trainer.tokenizer, mock_tokenizer_instance)
        self.assertEqual(trainer.model, mock_model_instance)
        self.assertEqual(trainer.tokenizer.pad_token, trainer.tokenizer.eos_token)
    
    def test_augment_dataset(self):
        """Тест аугментации данных"""
        # Создаем мок датасета
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = MagicMock()
        
        # Создаем тренера и вызываем метод
        trainer = AristotleTrainer()
        trainer.augmenter = TextAugmenter()
        result = trainer.augment_dataset(mock_dataset)
        
        # Проверки
        mock_dataset.map.assert_called_once()
        self.assertIsNotNone(result)
    
    @patch('src.training.trainer.optuna.create_study')
    def test_optimize_hyperparams(self, mock_create_study):
        """Тест оптимизации гиперпараметров"""
        # Настройка моков
        mock_study = MagicMock()
        mock_study.best_params = {'lr': 5e-5, 'batch_size': 16, 'weight_decay': 0.01}
        mock_create_study.return_value = mock_study
        
        # Создаем тренера и мок-объекты
        trainer = AristotleTrainer()
        trainer.config = self.test_config
        trainer.model = MagicMock()
        trainer.tokenizer = MagicMock()
        
        # Мок для токенизированных данных
        tokenized_datasets = {
            'train': MagicMock(),
            'test': MagicMock()
        }
        
        # Вызываем метод
        best_params = trainer.optimize_hyperparams(tokenized_datasets)
        
        # Проверки
        mock_create_study.assert_called_once()
        mock_study.optimize.assert_called_once()
        self.assertEqual(best_params, mock_study.best_params)
        self.assertIn('lr', best_params)
        self.assertIn('batch_size', best_params)
        self.assertIn('weight_decay', best_params)


class TestTextAugmenter(unittest.TestCase):
    """Тесты для класса TextAugmenter"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.augmenter = TextAugmenter()
        self.test_text = "Добродетель есть середина между двумя крайностями."
    
    def test_augment_text(self):
        """Тест основного метода аугментации"""
        augmented = self.augmenter.augment_text(self.test_text)
        
        # Проверяем, что текст изменился, но не пустой
        self.assertIsNotNone(augmented)
        self.assertGreater(len(augmented), 0)
    
    def test_synonym_replacement(self):
        """Тест замены синонимов"""
        # Подготавливаем текст с известным словом из словаря синонимов
        test_text = "Мудрость - это главное качество философа."
        
        # Патчим random.random, чтобы всегда возвращал 0.1 (меньше порога 0.3)
        with patch('random.random', return_value=0.1):
            augmented = self.augmenter._synonym_replacement(test_text)
            
            # Проверяем, что слово "мудрость" было заменено на синоним
            self.assertNotEqual(augmented, test_text)
            self.assertNotIn("мудрость", augmented.lower())
    
    def test_add_philosophical_connectors(self):
        """Тест добавления философских связок"""
        # Патчим random.random и random.randint
        with patch('random.random', return_value=0.1), \
             patch('random.randint', return_value=1), \
             patch('random.choice', return_value="следовательно"):
            
            augmented = self.augmenter._add_philosophical_connectors(self.test_text)
            
            # Проверяем, что связка добавлена
            self.assertIn("следовательно", augmented)
    
    def test_batch_augment(self):
        """Тест пакетной аугментации"""
        texts = [self.test_text, "Счастье есть деятельность души в полноте добродетели."]
        augmented_texts = self.augmenter.batch_augment(texts, augment_factor=2)
        
        # Проверяем, что количество текстов увеличилось в augment_factor раз
        self.assertEqual(len(augmented_texts), len(texts) * 2)


class TestUtils(unittest.TestCase):
    """Тесты для утилит обучения"""
    
    def test_validate_config(self):
        """Тест валидации конфигурации"""
        # Валидная конфигурация
        valid_config = {
            'model_name': 'sberbank-ai/rugpt3small_based_on_gpt2',
            'dataset_name': 'DmitryYarov/aristotle-russian',
            'max_length': 512,
            'test_size': 0.1,
            'output_dir': './models'
        }
        
        # Невалидные конфигурации
        invalid_config1 = {
            'model_name': 'sberbank-ai/rugpt3small_based_on_gpt2',
            # Отсутствует dataset_name
            'max_length': 512,
            'test_size': 0.1,
            'output_dir': './models'
        }
        
        invalid_config2 = {
            'model_name': 'sberbank-ai/rugpt3small_based_on_gpt2',
            'dataset_name': 'DmitryYarov/aristotle-russian',
            'max_length': -10,  # Отрицательное значение
            'test_size': 0.1,
            'output_dir': './models'
        }
        
        # Проверки
        self.assertTrue(validate_config(valid_config))
        self.assertFalse(validate_config(invalid_config1))
        self.assertFalse(validate_config(invalid_config2))
    
    def test_calculate_model_size(self):
        """Тест вычисления размера модели"""
        # Создаем временную директорию с файлами
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем тестовые файлы
            with open(os.path.join(temp_dir, 'model.bin'), 'wb') as f:
                f.write(b'0' * 1024 * 1024)  # 1 MB
            
            with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
                f.write('{"test": "config"}')
            
            # Вычисляем размер
            size_info = calculate_model_size(temp_dir)
            
            # Проверки
            self.assertIn('total_size_mb', size_info)
            self.assertIn('file_count', size_info)
            self.assertEqual(size_info['file_count'], 2)
            self.assertGreaterEqual(size_info['total_size_mb'], 1.0)  # Минимум 1 MB


if __name__ == '__main__':
    unittest.main()
