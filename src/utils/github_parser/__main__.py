import logging
from time import perf_counter
import hashlib

from src.utils.github_parser.config import load_config_file
from src.utils.github_parser.cst_parser import CSTCodeParser
from src.utils.github_parser.dataset_builder import DatasetBuilder
from src.utils.github_parser.github_repo_loader import TorchGitHubLoader
from src.utils.github_parser.text_file_parsers import MarkdownParser, RSTParser
from src.agent_constructor.core import Chunk


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
        
        logger.info("Время анализа в секундах: %s", processing_time)
        
        # ========== СОХРАНЕНИЕ ДАННЫХ В БАЗУ ДАННЫХ ==========
        if dataset:
           
            logger.info("Всего извлечено примеров: %s", len(dataset))
            
            try:
                # Импортируем необходимые классы
                from src.utils.github_parser.github_parser_db.github_docs_db import GitHubDocsDB
                from src.agents.general.embedding_agents import EmbeddingAgent
                
                # Создаём эмбеддер
                logger.info("Создание EmbeddingAgent...")
                embedder = EmbeddingAgent(
                    url="http://localhost:7216/v1",
                    model_name="Qwen/Qwen3-Embedding-4B",
                )
                logger.info("EmbeddingAgent создан")
                
                # Создаём подключение к БД
                logger.info("Подключение к базе данных...")
                db = GitHubDocsDB(
                    embedder=embedder,
                    db_path="data/github_example.db",
                    vector_db_path="data/github_vector_db",
                    collection_name="docs"
                )
                logger.info("GitHubDocsDB создана")
                
                # Конвертируем примеры в чанки с проверкой дубликатов
                logger.info("Конвертация примеров в чанки...")
                chunks = []
                seen_ids = set()  # множество для отслеживания уникальных ID
                duplicates = 0
                
                for i, example in enumerate(dataset):
                    # Генерируем уникальный ID
                    chunk_id = hashlib.md5(
                        f"{config.repo_for_analyzing}:{example.source_object_path}:{example.source_object_name}:{example.solution_code}".encode()
                    ).hexdigest()
                    
                    # Проверяем на дубликаты
                    if chunk_id in seen_ids:
                        duplicates += 1
                        continue  # пропускаем дубликат
                    
                    seen_ids.add(chunk_id)
                    
                    # Создаём чанк
                    chunk = Chunk(
                        id=chunk_id,
                        doc_id=chunk_id,
                        text=example.solution_code,
                        metadata={
                            'name': example.source_object_name,
                            'type': example.source_object_type,
                            'path': example.source_object_path,
                            'description': example.task_description[:200] if example.task_description else '',
                            'repo': config.repo_for_analyzing
                        }
                    )
                    chunks.append(chunk)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"   Обработано {i+1}/{len(dataset)} примеров")
                
                logger.info("Создано %s уникальных чанков", len(chunks))
                if duplicates > 0:
                    logger.info("Пропущено дубликатов: %s", duplicates)
                
                # Добавляем чанки в векторную БД
                if chunks:
                    logger.info("Добавление чанков в векторную БД...")
                    db.add_chunks(chunks)
                    logger.info("Добавлено %s чанков в БД", len(chunks))
                else:
                    logger.warning("Нет чанков для добавления")
                
                # Статистика
                #stats = db.get_stats()
                #logger.info("Статистика БД:")
                #logger.info("Всего чанков: %s", stats.get('total_chunks', 0))
                
            except Exception as e:
                logger.error("Ошибка при сохранении в БД: %s", e, exc_info=True)
        else:
            logger.warning("Нет данных для сохранения в БД")
        
        # ========== ВЫВОД ПРИМЕРОВ В ЛОГ ==========
        if dataset:
            for i, example in enumerate(dataset[:5]):  # Показываем первые 5 примеров
                logger.info("-" * 50)
                logger.info("Пример %d:", i+1)
                logger.info("Файл: %s", example.source_object_path)
                logger.info("Объект: %s", example.source_object_name)
                logger.info("Тип объекта: %s", example.source_object_type)
                logger.info("Код:\n%s", example.solution_code[:200] + "..." if len(example.solution_code) > 200 else example.solution_code)
    except Exception as e:
        logger.error("Ошибка при выполнении пайплайна! %s", e, exc_info=True)