import logging
from time import perf_counter

from src.utils.github_parser.config import load_config_file
from src.utils.github_parser.cst_parser import CSTCodeParser
from src.utils.github_parser.dataset_builder import DatasetBuilder
from src.utils.github_parser.github_repo_loader import TorchGitHubLoader
from src.utils.github_parser.text_file_parsers import MarkdownParser, RSTParser


logger = logging.getLogger(__file__)


if __name__ == "__main__":
    """
    Точка входа для запуска процесса сборки датасета пар "задача-решение"
    с репозитория torchmetrics.
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
    code_parser = CSTCodeParser(
        config.ignore_internal_functions, config.min_code_length_for_analyzing
    )
    md_parser = MarkdownParser(config.min_code_length_for_analyzing)
    rst_parser = RSTParser(
        config.rst_skip_code_lines_patterns,
        config.min_code_length_for_analyzing,
    )
    pipeline = DatasetBuilder(
        loader,
        code_parser,
        md_parser,
        rst_parser,
    )
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
        logger.info(
            "Целевые секции для парсинга в корневом README файле: %s",
            config.readme_root_target_sections,
        )
        start_time = perf_counter()
        dataset = pipeline.run(
            config.repo_for_analyzing,
            config.specific_folder_for_analyzing,
            config.extract_examples_from_root_readme,
            config.readme_root_target_sections,
        )
        processing_time = round(perf_counter() - start_time, 2)
        logger.info("=" * 70)
        logger.info("АНАЛИЗ ЗАВЕРШЕН")
        logger.info("=" * 70)
        logger.info("Время анализа в секундах: %s", processing_time)
        # TODO: добавить LLM для фильтрации
        if dataset:
            for example in dataset:
                logger.info("-" * 50)
                logger.info("Файл: %s", example.source_object_path)
                logger.info("Объект: %s", example.source_object_name)
                logger.info("Тип объекта: %s", example.source_object_type)
                logger.info("Описание задачи:\n%s", example.task_description)
                logger.info("Код:\n%s", example.solution_code)
                # logger.info(
                #     "Метаданные (исходный код):\n%s",
                #     example.metadata_source_code,
                # )
                logger.info("Ссылки: %s", example.references)
    except Exception as e:
        logger.error("Ошибка при выполнении пайплайна! %s", e, exc_info=True)
