#!/usr/bin/env python3
import argparse
import logging
import sys
import json
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import AristotlePredictor
from src.training.utils import setup_logging
from datasets import load_dataset
import model_evaluation
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Оценка модели Аристотеля")
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Путь к обученной модели"
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="DmitryYarov/aristotle-russian",
        help="Имя или путь к тестовому датасету"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="evaluation_results.json",
        help="Путь для сохранения результатов оценки"
    )
    
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=100,
        help="Количество примеров для оценки"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Уровень логирования"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Настройка логирования
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting evaluation script")
    logger.info(f"Using model: {args.model_path}")
    
    try:
        # Загрузка модели
        predictor = AristotlePredictor(args.model_path)
        
        # Загрузка датасета
        logger.info(f"Loading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset, split="train")
        
        # Разделение на тестовую выборку, если нужно
        if len(dataset) > args.num_samples:
            dataset = dataset.select(range(args.num_samples))
        
        logger.info(f"Evaluating on {len(dataset)} samples")
        
        # Инициализация метрик
        perplexity = evaluate.load("perplexity")
        rouge = evaluate.load("rouge")
        
        # Оценка перплексии
        logger.info("Calculating perplexity...")
        perplexity_results = perplexity.compute(
            model=predictor.model,
            add_start_token=True,
            texts=dataset["text"]
        )
        
        # Генерация текстов для оценки ROUGE
        logger.info("Generating texts for ROUGE evaluation...")
        generated_texts = []
        reference_texts = []
        
        for i, example in enumerate(dataset):
            # Берем первые 50 слов как промпт
            prompt_words = example["text"].split()[:50]
            prompt = " ".join(prompt_words)
            
            # Генерируем текст
            generated = predictor.generate_text(prompt, max_length=200)[0]
            
            # Добавляем в списки для оценки
            generated_texts.append(generated)
            reference_texts.append(example["text"])
            
            if i % 10 == 0:
                logger.info(f"Generated {i}/{len(dataset)} texts")
        
        # Оценка ROUGE
        logger.info("Calculating ROUGE scores...")
        rouge_results = rouge.compute(
            predictions=generated_texts,
            references=reference_texts
        )
        
        # Сбор всех метрик
        evaluation_results = {
            "perplexity": perplexity_results["mean_perplexity"],
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "model_path": args.model_path,
            "dataset": args.dataset,
            "num_samples": len(dataset)
        }
        
        # Сохранение результатов
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {args.output}")
        logger.info(f"Perplexity: {evaluation_results['perplexity']}")
        logger.info(f"ROUGE-1: {evaluation_results['rouge1']}")
        logger.info(f"ROUGE-2: {evaluation_results['rouge2']}")
        logger.info(f"ROUGE-L: {evaluation_results['rougeL']}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Evaluation script completed successfully")


if __name__ == "__main__":
    main()