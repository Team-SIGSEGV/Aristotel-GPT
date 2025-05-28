import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import optuna
from optuna.pruners import MedianPruner
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
# Исправляем импорт evaluate
import evaluate
from evaluate import load as evaluate_load, load
import yaml

from .augmenter import TextAugmenter
from .utils import setup_logging, save_metrics

logger = logging.getLogger(__name__)


class AristotleTrainer:
    """Класс для обучения модели Аристотеля"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.augmenter = TextAugmenter()
        setup_logging(self.config.get('log_level', 'INFO'))
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Конфигурация по умолчанию
        return {
            'model_name': 'sberbank-ai/rugpt3small_based_on_gpt2',
            'dataset_name': 'DmitryYarov/aristotle-russian',
            'max_length': 512,
            'test_size': 0.1,
            'optuna_trials': 5,
            'optuna_timeout': 3600,
            'output_dir': './models',
            'log_level': 'INFO'
        }
    
    def load_data(self) -> Dataset:
        """Загрузка и подготовка данных"""
        logger.info(f"Loading dataset: {self.config['dataset_name']}")
        
        try:
            dataset = load_dataset(self.config['dataset_name'], split="train")
            dataset = dataset.train_test_split(test_size=self.config['test_size'])
            
            logger.info(f"Dataset loaded. Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def setup_model_and_tokenizer(self):
        """Инициализация модели и токенизатора"""
        logger.info(f"Loading model: {self.config['model_name']}")
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config['model_name'])
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = GPT2LMHeadModel.from_pretrained(self.config['model_name'])
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def augment_dataset(self, dataset: Dataset) -> Dataset:
        """Аугментация данных"""
        logger.info("Starting data augmentation")
        
        def augment_fn(examples):
            return {
                "text": [self.augmenter.augment_text(text) for text in examples["text"]]
            }
        
        augmented = dataset.map(augment_fn, batched=True)
        logger.info("Data augmentation completed")
        
        return augmented
    
    def tokenize_data(self, dataset: Dataset) -> Dataset:
        """Токенизация данных"""
        logger.info("Tokenizing dataset")
        
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=self.config['max_length']
            )
        
        tokenized = dataset.map(tokenize_fn, batched=True)
        logger.info("Tokenization completed")
        
        return tokenized
    
    def optimize_hyperparams(self, tokenized_datasets: Dataset) -> Dict[str, Any]:
        """Оптимизация гиперпараметров с помощью Optuna"""
        logger.info("Starting hyperparameter optimization")
        
        def objective(trial):
            args = TrainingArguments(
                output_dir="./optuna_trials",
                learning_rate=trial.suggest_float("lr", 1e-5, 5e-4, log=True),
                per_device_train_batch_size=trial.suggest_categorical("batch_size", [8, 16]),
                num_train_epochs=3,
                weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
                evaluation_strategy="epoch",
                logging_steps=50,
                report_to="none",
                save_strategy="no"
            )
            
            trainer = Trainer(
                model=self.model,
                args=args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            )
            
            trainer.train()
            eval_result = trainer.evaluate()
            return eval_result["eval_loss"]
        
        study = optuna.create_study(direction="minimize", pruner=MedianPruner())
        study.optimize(
            objective,
            n_trials=self.config['optuna_trials'],
            timeout=self.config['optuna_timeout']
        )
        
        logger.info(f"Best hyperparams: {study.best_params}")
        return study.best_params
    
    def train_final_model(self, tokenized_datasets: Dataset, best_params: Dict[str, Any]) -> Trainer:
        """Финальное обучение модели"""
        logger.info("Starting final model training")
        
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            learning_rate=best_params["lr"],
            per_device_train_batch_size=best_params["batch_size"],
            num_train_epochs=5,
            weight_decay=best_params["weight_decay"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=100,
            gradient_accumulation_steps=2,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        trainer.train()
        logger.info("Final training completed")
        
        return trainer
    
    def evaluate_model(self, dataset: Dataset) -> Dict[str, float]:
        """Оценка модели"""
        logger.info("Evaluating model")
        
        try:
            perplexity = load("perplexity")
            results = perplexity.compute(
                model=self.model,
                add_start_token=True,
                texts=dataset["test"]["text"][:100]
            )
            
            metrics = {
                "perplexity": results['mean_perplexity']
            }
            
            logger.info(f"Evaluation results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"perplexity": float('inf')}
    
    def save_model(self, output_dir: str):
        """Сохранение модели"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Model saved to {output_path}")
    
    def train_pipeline(self) -> Dict[str, Any]:
        """Полный пайплайн обучения"""
        logger.info("Starting training pipeline")
        
        # Загрузка данных
        dataset = self.load_data()
        
        # Инициализация модели
        self.setup_model_and_tokenizer()
        
        # Аугментация данных
        augmented_dataset = self.augment_dataset(dataset)
        
        # Токенизация
        tokenized_datasets = self.tokenize_data(augmented_dataset)
        
        # Оптимизация гиперпараметров
        best_params = self.optimize_hyperparams(tokenized_datasets)
        
        # Финальное обучение
        trainer = self.train_final_model(tokenized_datasets, best_params)
        
        # Сохранение модели
        self.save_model(self.config['output_dir'])
        
        # Оценка
        metrics = self.evaluate_model(dataset)
        
        # Сохранение метрик
        save_metrics(metrics, self.config['output_dir'])
        
        logger.info("Training pipeline completed successfully")
        
        return {
            "best_params": best_params,
            "metrics": metrics,
            "model_path": self.config['output_dir']
        }