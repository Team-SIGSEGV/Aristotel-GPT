#!/usr/bin/env python3
import argparse
import logging
import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import AristotleTrainer
from src.training.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение нейросети Аристотеля")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Путь к конфигурационному файлу"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Директория для сохранения модели (переопределяет значение из конфига)"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Уровень логирования"
    )
    
    parser.add_argument(
        "--skip-optuna", 
        action="store_true",
        help="Пропустить оптимизацию гиперпараметров"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Настройка логирования
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training script")
    logger.info(f"Using config: {args.config}")
    
    try:
        # Инициализация тренера
        trainer = AristotleTrainer(args.config)
        
        # Переопределение директории для сохранения модели, если указана
        if args.output_dir:
            trainer.config['output_dir'] = args.output_dir
            logger.info(f"Output directory overridden: {args.output_dir}")
        
        # Создание директории для сохранения модели
        os.makedirs(trainer.config['output_dir'], exist_ok=True)
        
        # Запуск пайплайна обучения
        if args.skip_optuna:
            logger.info("Skipping hyperparameter optimization")
            # Загрузка данных
            dataset = trainer.load_data()
            
            # Инициализация модели
            trainer.setup_model_and_tokenizer()
            
            # Аугментация и токенизация
            augmented_dataset = trainer.augment_dataset(dataset)
            tokenized_datasets = trainer.tokenize_data(augmented_dataset)
            
            # Используем дефолтные гиперпараметры
            best_params = {
                "lr": 5e-5,
                "batch_size": 16,
                "weight_decay": 0.01
            }
            
            # Обучение и оценка
            trainer.train_final_model(tokenized_datasets, best_params)
            metrics = trainer.evaluate_model(dataset)
            
            # Сохранение модели и метрик
            trainer.save_model(trainer.config['output_dir'])
            trainer.save_metrics(metrics, trainer.config['output_dir'])
            
        else:
            # Полный пайплайн с оптимизацией
            results = trainer.train_pipeline()
            logger.info(f"Training completed with metrics: {results['metrics']}")
        
        logger.info(f"Model saved to {trainer.config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Training script completed successfully")


if __name__ == "__main__":
    main()