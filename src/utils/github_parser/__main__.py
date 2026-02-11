import logging
from time import time

from src.utils.github_parser.config import load_config_file
from src.utils.github_parser.cst_parser import CSTCodeParser
from src.utils.github_parser.dataset_builder import DatasetBuilder
from src.utils.github_parser.github_repo_loader import TorchGitHubLoader
from src.utils.github_parser.text_file_parsers import MarkdownParser, RSTParser


logger = logging.getLogger(__file__)


if __name__ == "__main__":
    """
    Точка входа для запуска процесса сборки датасета.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                filename="github_repo_processor.log",
                encoding="utf-8",
            ),
        ],
    )  # TODO: перед PR удалить, чтобы не перезаписать конфигурацию!
    config = load_config_file()
    loader = TorchGitHubLoader(config.config_for_github_loader)
    code_parser = CSTCodeParser()
    md_parser = MarkdownParser()
    rst_parser = RSTParser()
    pipeline = DatasetBuilder(loader, code_parser, md_parser, rst_parser)
    try:
        logger.info("Репозиторий: %s", config.repo_for_analyzing)
        logger.info(
            "Целевая папка: %s",
            config.specific_folder_for_analyzing or "весь репозиторий",
        )
        logger.info(
            "Расширения файлов для парсинга: %s",
            config.config_for_github_loader.allowed_file_extensions,
        )
        logger.info(
            "Пропускаемые паттерны: %s",
            config.config_for_github_loader.skip_file_patterns,
        )
        start_time = time()
        dataset = pipeline.run(
            config.repo_for_analyzing, config.specific_folder_for_analyzing
        )
        processing_time = round(time() - start_time, 2)
        logger.info("=" * 70)
        logger.info("АНАЛИЗ ЗАВЕРШЕН")
        logger.info("=" * 70)
        logger.info("Время анализа в секундах: %s", processing_time)
        if dataset:
            for example in dataset:
                logger.info("-" * 50)
                logger.info("Файл: %s", example.source_object_path)
                logger.info("Объект: %s", example.source_object_name)
                logger.info("Тип объекта: %s", example.source_object_type)
                logger.info("Описание задачи:\n%s", example.task_description)
                logger.info("Код:\n%s", example.solution_code)
                logger.info(
                    "Метаданные (исходный код):\n%s",
                    example.metadata_source_code,
                )
                logger.info("Ссылки: %s", example.references)
    except Exception as e:
        logger.critical(
            "Ошибка при выполнении пайплайна! %s", e, exc_info=True
        )
