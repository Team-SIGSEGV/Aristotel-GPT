import logging
import torch
from typing import Dict, Any, List, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class AristotlePredictor:
    """Класс для генерации текста с использованием обученной модели Аристотеля"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Инициализация предиктора
        
        Args:
            model_path: Путь к сохраненной модели
            config_path: Путь к конфигурационному файлу
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Конфигурация по умолчанию
        return {
            'max_length': 200,
            'num_return_sequences': 1,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.2,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def _load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            
            device = self.config.get('device', 'cpu')
            self.model.to(device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_text(self, prompt: str, **kwargs) -> List[str]:
        """
        Генерация текста на основе промпта
        
        Args:
            prompt: Начальный текст для генерации
            **kwargs: Дополнительные параметры для генерации
                
        Returns:
            List[str]: Список сгенерированных текстов
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        # Объединяем параметры из конфига и переданные аргументы
        generation_params = {
            'max_length': self.config['max_length'],
            'num_return_sequences': self.config['num_return_sequences'],
            'temperature': self.config['temperature'],
            'top_k': self.config['top_k'],
            'top_p': self.config['top_p'],
            'repetition_penalty': self.config['repetition_penalty'],
            'do_sample': True,
            'no_repeat_ngram_size': 2,
            'pad_token_id': self.tokenizer.eos_token_id
        }
        
        # Обновляем параметры из kwargs
        generation_params.update(kwargs)
        
        try:
            # Токенизация промпта
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self.config['device'])
            
            # Генерация текста
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    **generation_params
                )
            
            # Декодирование результатов
            generated_texts = []
            for sequence in output_sequences:
                text = self.tokenizer.decode(sequence, skip_special_tokens=True)
                generated_texts.append(text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return [f"Ошибка генерации: {str(e)}"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "device": self.config['device'],
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "generation_config": {
                k: v for k, v in self.config.items() 
                if k in ['max_length', 'temperature', 'top_k', 'top_p', 'repetition_penalty']
            }
        }
    
    def adjust_generation_params(self, **params):
        """Настройка параметров генерации"""
        for key, value in params.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Updated parameter {key} to {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")
        
        return self.config