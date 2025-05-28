#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import shutil
import json
import subprocess
from pathlib import Path
import yaml

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Развертывание модели Аристотеля")
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Путь к обученной модели"
    )
    
    parser.add_argument(
        "--deploy-dir", 
        type=str, 
        default="deployment",
        help="Директория для развертывания"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/inference_config.yaml",
        help="Путь к конфигурационному файлу для инференса"
    )
    
    parser.add_argument(
        "--docker", 
        action="store_true",
        help="Собрать и запустить Docker-контейнер"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Порт для API"
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
    
    logger.info("Starting deployment script")
    logger.info(f"Using model: {args.model_path}")
    
    try:
        # Проверка существования модели
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model path does not exist: {args.model_path}")
            sys.exit(1)
        
        # Создание директории для развертывания
        deploy_dir = Path(args.deploy_dir)
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Копирование модели
        model_deploy_dir = deploy_dir / "model"
        if model_deploy_dir.exists():
            logger.info(f"Removing existing model directory: {model_deploy_dir}")
            shutil.rmtree(model_deploy_dir)
        
        logger.info(f"Copying model from {args.model_path} to {model_deploy_dir}")
        shutil.copytree(model_path, model_deploy_dir)
        
        # Копирование конфигурации
        config_path = Path(args.config)
        if config_path.exists():
            # Загрузка и модификация конфигурации
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Обновление пути к модели
            config["model_path"] = str(model_deploy_dir)
            config["port"] = args.port
            
            # Сохранение обновленной конфигурации
            deploy_config_path = deploy_dir / "inference_config.yaml"
            with open(deploy_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {deploy_config_path}")
        else:
            logger.warning(f"Config file not found: {args.config}")
        
        # Создание файла с информацией о развертывании
        deployment_info = {
            "model_path": str(model_deploy_dir),
            "config_path": str(deploy_config_path) if config_path.exists() else None,
            "timestamp": Path(model_path / "training_metrics.json").stat().st_mtime if Path(model_path / "training_metrics.json").exists() else None,
            "port": args.port
        }
        
        with open(deploy_dir / "deployment_info.json", 'w', encoding='utf-8') as f:
            json.dump(deployment_info, f, indent=2, ensure_ascii=False)
        
        # Если указан флаг --docker, собираем и запускаем контейнер
        if args.docker:
            logger.info("Building Docker container")
            
            # Копирование Dockerfile
            dockerfile_src = Path("docker/Dockerfile.inference")
            if not dockerfile_src.exists():
                logger.error(f"Dockerfile not found: {dockerfile_src}")
                sys.exit(1)
            
            shutil.copy(dockerfile_src, deploy_dir / "Dockerfile")
            
            # Сборка образа
            docker_build_cmd = [
                "docker", "build", 
                "-t", "aristotle-inference:latest",
                "-f", str(deploy_dir / "Dockerfile"),
                str(deploy_dir)
            ]
            
            logger.info(f"Running command: {' '.join(docker_build_cmd)}")
            subprocess.run(docker_build_cmd, check=True)
            
            # Запуск контейнера
            docker_run_cmd = [
                "docker", "run", 
                "-d",
                "-p", f"{args.port}:{args.port}",
                "--name", "aristotle-inference",
                "-v", f"{os.path.abspath(deploy_dir)}/model:/app/model",
                "-v", f"{os.path.abspath(deploy_dir)}/inference_config.yaml:/app/config.yaml",
                "-e", "CONFIG_PATH=/app/config.yaml",
                "-e", "MODEL_PATH=/app/model",
                "-e", f"API_PORT={args.port}",
                "aristotle-inference:latest"
            ]
            
            logger.info(f"Running command: {' '.join(docker_run_cmd)}")
            subprocess.run(docker_run_cmd, check=True)
            
            logger.info(f"Docker container started. API available at http://localhost:{args.port}")
        else:
            logger.info(f"Model deployed to {deploy_dir}")
            logger.info(f"To start the API server, run: python -m src.inference.api --config {deploy_config_path}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Deployment script completed successfully")


if __name__ == "__main__":
    main()